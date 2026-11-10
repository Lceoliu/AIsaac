// B58d: the lane service's state machine (abp_lanes.h) with the native lineage under -fsanitize=thread, random
// interleavings. No CUDA, no torch: the GPU side (restore, the jobs' steps) is a fake that writes page rows and
// computes each answer as a function of (request sequence, lane).
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_lanes.cpp -o test_abp_lanes && ./test_abp_lanes 3000 <seed>
//
// Threads: workers (one per lane: a branch begins at one of its own captured points, a few continuations, then the
// point released through the status array; each request written into the lane arrays, then its message into the
// lane thread's intake queue (the pipe), then the answer awaited and checked: the echoed sequence, the actions);
// the lane thread (LaneCore: intake, control, window, submit / begins / jobs, answers, replies); the driver (main rows'
// logs and snapshots, CF captures into the point table, status 2); a Python-like reader (contexts / injury snapshots
// pinned, their parts read, the K/V rows' bytes checked). The GC runs with the lane thread paused in production (the
// hand-off) and is not part of this test. "replyfirst" (the control): the reply before the answer is written: the
// workers must see wrong answers (and TSAN races). Exit code 0 when no check failed.
#include "../abp_lanes.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <random>
#include <thread>

using namespace abpc;

static uint8_t pat(i64 dec, int b) { return (uint8_t)((dec*131+b*7+(dec >> 8)) & 0xff); }
static i64 act(i64 seq, i64 lane, i64 h) { return (seq*7+lane*3+h) % 5; }

int run(int iters, unsigned seed, bool reply_first) {
  const i64 L = 4, NH = 2, BS = 4, KV = 16, BANK = 4, TW = 6, ROWSZ = 64;
  PageCore pc(32, KV, 2, 2, BS, NH, 2048, 512);
  Native N;
  LineageCfg lcfg; lcfg.window = 6; lcfg.block_steps = BS; lcfg.memory_blocks = 2; lcfg.nh = NH;
  N.lin.configure(lcfg, 3*L);   // (rows: generation 0 lanes, generation 1 lanes, main rows)
  N.inj.configure(L, {1});
  std::vector<i64> status(L*BANK, 0), mode(L, 0), kept(L*2, 0), cf_snap(L, 0);
  std::vector<int32_t> io(L*12, 0); std::vector<double> out(L*6, 0); std::vector<float> lt(L*TW, 0);
  std::vector<uint8_t> rows(L*ROWSZ, 0);
  N.status = status.data();
  std::vector<i64> jobs(L*2*2, -1), epoch(L, 0);   // (the reclamation: no injury-job pins, no teacher rings)
  N.tr.jobs = jobs.data(); N.tr.jobs_k = 2; N.tr.epoch = epoch.data(); N.tr.ring = 0; N.tr_on = true;
  { std::lock_guard<std::mutex> l(N.lin.mu); N.pts.published(0, 0); N.pts.published(1, 1); }
  LaneCore C;
  LaneCfg cfg; cfg.lanes = L; cfg.nh = NH; cfg.tw = TW; cfg.wait = 0; cfg.quiet = 0; cfg.reclaim_s = 0.005; cfg.pid = 5;
  LaneShared sh; sh.rows = rows.data(); sh.row_size = ROWSZ; sh.o_episode = 0; sh.s_episode = 8; sh.io = io.data(); sh.io_cols = 12;
  sh.out = out.data(); sh.out_cols = 6; sh.t = lt.data(); sh.mode = mode.data(); sh.status = status.data(); sh.pts_per = BANK;
  sh.kept = kept.data(); sh.kept_cols = 2; sh.cf_snap = cf_snap.data();
  C.configure(cfg, sh, &N);
  C.reply_first = reply_first;
  // the intake queue (the lanes' pipes) and the replies
  std::mutex qm; std::deque<i64> q;
  std::mutex rm; std::condition_variable rc; std::vector<i64> replied(L, 0);
  std::atomic<bool> done{false}; std::atomic<i64> errors{0}, answers{0}, reads{0}, workers_left{L};
  auto write_rows = [&](i64 n, const std::vector<i64> &rws, std::vector<i64> &dec, std::vector<i64> &kvs, std::vector<i64> &bid,
                        std::vector<i64> &steps) {
    uint8_t frame[8] = {1};
    std::vector<uint8_t> kv(n*KV), first(n, 0); std::vector<float> z(n*2, 0); std::vector<i64> pend(n*NH, 0), nb(n, 8);
    std::vector<int32_t> sch(n, 1); std::vector<const uint8_t *> fr(n, frame);
    i64 next; { std::lock_guard<std::mutex> l(pc.mu); next = pc.next_decision; }
    for (i64 j = 0; j < n; j++) for (int b = 0; b < KV; b++) kv[j*KV+b] = pat(next+j, b);
    dec.resize(n); kvs.resize(n); bid.assign(n, -1);
    pc.write_decisions_raw(n, kv.data(), KV, z.data(), 2, steps.data(), 0, first.data(), sch.data(), fr.data(), nb.data(), pend.data(),
                           dec.data(), kvs.data());
    for (i64 j = 0; j < n; j++)
      if ((steps[j]+1) % BS == 0) {
        std::vector<float> v(2, 1.f); std::vector<i64> src(BS, -1), sv(BS, -1); i64 e = steps[j];
        pc.write_blocks_raw(1, v.data(), 2, &e, 0, src.data(), sv.data(), &bid[j]);
      }
    (void)rws;
  };
  C.ops.base = [&](i64 g) { return g*L; };
  C.ops.gen_version = [&](i64 g) { std::lock_guard<std::mutex> l(N.lin.mu); return N.pts.gens.version[g]; };
  C.ops.main_version = [&]() { return (i64)0; };
  C.ops.restore = [&](i64 g, i64 lane, i64 snap) { std::lock_guard<std::mutex> l(N.lin.mu); N.lin.set_head_locked(g*L+lane, N.lin.branch_locked(snap)); };
  C.ops.reply = [&](const std::vector<i64> &lanes) {
    std::lock_guard<std::mutex> l(rm);
    for (i64 x : lanes) replied[x]++;
    rc.notify_all();
  };
  C.ops.submit = [&](i64 job, i64 g, std::vector<uint8_t> rws, std::vector<i64> pend, std::vector<i64> lanes, std::vector<i64> seq,
                     std::vector<uint8_t> replay) {
    (void)rws; (void)pend; (void)replay;
    i64 k = (i64)lanes.size();
    std::vector<i64> steps(k), rowids(k);
    {
      std::lock_guard<std::mutex> l(N.lin.mu);
      for (i64 q_ = 0; q_ < k; q_++) { rowids[q_] = g*L+lanes[q_]; i64 h = N.lin.head[rowids[q_]]; steps[q_] = N.lin.logs[h].end(); }
    }
    LaneJobOut o;
    write_rows(k, rowids, o.dec, o.kvs, o.bid, steps);
    o.step_before = steps; o.first.assign(k, 0); o.version = g;
    for (i64 q_ = 0; q_ < k; q_++) o.complete.push_back((steps[q_]+1) % BS == 0);
    o.out.assign(k*(NH+3), 0.f);
    for (i64 q_ = 0; q_ < k; q_++) {
      for (i64 h = 0; h < NH; h++) o.out[q_*(NH+3)+h] = (float)act(seq[q_], lanes[q_], h);
      o.out[q_*(NH+3)+NH] = (float)seq[q_];
    }
    C.job_done(job, o);
  };
  auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  // the driver: main rows' logs, snapshots, CF captures (points of every worker's bank rows)
  std::thread driver([&] {
    std::mt19937 r(seed);
    for (i64 w = 0; w < L; w++) { std::lock_guard<std::mutex> l(N.lin.mu); N.lin.set_head_locked(2*L+w, N.lin.new_log_locked(9, w, 1, 0)); }
    i64 s = 0;
    while (workers_left.load() > 0) {
      i64 w = r()%L;
      pc.set_stamp(++s);
      std::vector<i64> rw{2*L+w}, dec, kvs, bid, steps(1);
      { std::lock_guard<std::mutex> l(N.lin.mu); steps[0] = N.lin.logs[N.lin.head[2*L+w]].end(); }
      write_rows(1, rw, dec, kvs, bid, steps);
      {
        std::lock_guard<std::mutex> l(N.lin.mu);
        i64 pend[NH] = {0, 0};
        i64 sn = N.lin.snap_row_locked(2*L+w, pend, w % 2);
        for (i64 b = 0; b < BANK; b++) {   // (a free bank row of this worker: captured)
          i64 p = w*BANK+b;
          if (__atomic_load_n(&status[p], __ATOMIC_ACQUIRE) == 0 && !N.pts.table.count(p) && r()%3 == 0) {
            i64 c = N.pts.capture(N.lin, p, sn, w % 2);
            if (c == 2) __atomic_store_n(&status[p], (i64)2, __ATOMIC_RELEASE);
            break;
          }
        }
        N.inj.record(N.lin, w, 1, steps[0], sn);
        N.lin.push_locked(2*L+w, dec[0], kvs[0]);
        if (bid[0] >= 0) N.lin.push_block_locked(2*L+w, bid[0], steps[0]);
        N.lin.release_locked(sn);
        N.lin.published = s;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20+r()%100));
    }
  });
  // the workers (one per lane)
  auto worker = [&](i64 w) {
    std::mt19937 r(seed*31+w);
    i64 seq = 1000*(w+1);
    for (int b = 0; b < iters; b++) {
      i64 point = -1;
      while (point < 0) {   // (one of its captured points)
        for (i64 bb = 0; bb < BANK; bb++) if (__atomic_load_n(&status[w*BANK+bb], __ATOMIC_ACQUIRE) == 2) { point = w*BANK+bb; break; }
        if (point < 0) std::this_thread::yield();
      }
      int steps = 1+r()%4;
      for (int s = 0; s < steps; s++) {
        seq++;
        io[w*12+10] = (int32_t)seq; io[w*12+0] = (int32_t)(seq % 3); io[w*12+1] = 0;
        io[w*12+11] = s == 0 ? (int32_t)(point+1) : 0;
        rows[w*ROWSZ] = (uint8_t)b;
        mode[w] = r()%5 == 0;
        i64 before;
        { std::lock_guard<std::mutex> l(rm); before = replied[w]; }
        { std::lock_guard<std::mutex> l(qm); q.push_back(w); }
        std::unique_lock<std::mutex> lk(rm);
        rc.wait(lk, [&] { return replied[w] > before; });
        lk.unlock();
        answers++;
        if ((i64)out[w*6+4] != seq) { errors++; continue; }
        if (out[w*6+5] == 0)
          for (i64 h = 0; h < NH; h++) if (io[w*12+5+h] != act(seq, w, h)) { errors++; break; }
      }
      __atomic_store_n(&status[point], (i64)3, __ATOMIC_RELEASE);   // (the point released: the control acknowledges with 0)
      while (__atomic_load_n(&status[point], __ATOMIC_ACQUIRE) != 0) std::this_thread::yield();
    }
    workers_left--;
  };
  std::vector<std::thread> ws;
  for (i64 w = 0; w < L; w++) ws.emplace_back(worker, w);
  std::thread lane([&] {
    i64 passes = 0;
    while (workers_left.load() > 0) {
      if (++passes % 200 == 0) collect(pc, N.lin, -1);   // (the GC between rounds; the driver's unpublished rows by the stamp)
      std::deque<i64> got;
      { std::lock_guard<std::mutex> l(qm); got.swap(q); }
      double t = now();
      for (i64 x : got) C.intake(x, t);
      C.control(t);
      if (C.fly && C.fly->ready) C.finish(t);
      if (!C.ready.empty() && !C.fly && C.due(t)) { C.submit(t); if (C.fly && C.fly->ready) C.finish(now()); }
      if (got.empty()) std::this_thread::yield();
    }
    done = true;
  });
  std::thread reader([&] {
    std::mt19937 r(seed+7);
    while (!done) {
      i64 snap = -1; Parts p;
      {
        std::lock_guard<std::mutex> l(N.lin.mu);
        std::vector<i64> cand;
        for (auto &e : N.lin.ctx) cand.push_back(e.second.second);
        for (auto &e : N.inj.sources) cand.push_back(e.second);
        if (cand.empty()) continue;
        snap = cand[r()%cand.size()];
        N.lin.retain_locked(snap);
        p = N.lin.parts_locked(snap);
      }
      for (size_t i = 0; i < p.kvs.size(); i++) {
        const uint8_t *row = pc.kv_page(p.kvs[i]/pc.page)+(p.kvs[i] % pc.page)*KV;
        i64 d;
        { std::lock_guard<std::mutex> l(N.lin.mu); const LogData &g = N.lin.logs[N.lin.snaps[snap].log]; d = g.dec[p.kv_steps[i]-g.offset]; }
        for (int b = 0; b < KV; b++) if (row[b] != pat(d, b)) { errors++; break; }
        reads++;
      }
      { std::lock_guard<std::mutex> l(N.lin.mu); N.lin.release_locked(snap); }
    }
  });
  for (auto &t : ws) t.join();
  lane.join(); driver.join(); reader.join();
  std::printf("lanes (reply %s): %lld answers, %lld rounds, %lld jobs, %lld begins, %lld row reads, %lld contexts, %lld points left, "
              "%lld wrong\n", reply_first ? "before the answer (control)" : "after the answer", (long long)answers.load(),
              (long long)C.n_rounds, (long long)C.n_jobs, (long long)C.n_begins, (long long)reads.load(), (long long)N.lin.ctx.size(),
              (long long)N.pts.table.size(), (long long)errors.load());
  return errors.load() == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 3000;
  unsigned seed = argc > 2 ? (unsigned)atoi(argv[2]) : 1;
  bool rf = argc > 3 && std::string(argv[3]) == "replyfirst";
  int rc = run(iters, seed, rf);
  std::printf("%s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
