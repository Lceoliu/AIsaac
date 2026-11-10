// B58c: the native lineage (abp_lineage.h) with the page store (abp_core.h) under -fsanitize=thread, random
// interleavings. No CUDA, no torch.
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_lineage.cpp -o test_abp_lineage && ./test_abp_lineage 20000 <seed>
//
// The driver: per step, for some workers, writes decision rows (and every block_steps a block) stamped with its step,
// then under the lineage lock: a new log at an episode start, the snapshot before the step, an injury source, a
// context now and then, the log pushes, the sidecar's chunk starts; then publishes the stamp. Every few hundred steps
// a hand-off: the sidecar carried into the other slot.
// A Python-like reader: pins a random live snapshot (sidecar / context / injury), takes its parts under the lock and
// reads every K/V row of its window outside it (each row's bytes are a function of its decision id).
// A retirer: completes owners and retires contexts with random live sets (the teacher builds).
// A collector: collect() with the stamp rule (limit -1: the lineage's published stamp + 1, read with the scan).
// Without the stamp rule ("nostamp": limit 0) a row written but not yet in the lineage can be freed and reused while
// a log will name it: wrong bytes in the reader (and TSAN races on the page rows). Exit code 0 when no check failed.
#include "../abp_lineage.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <thread>

using namespace abpc;

static uint8_t pat(i64 dec, int b) { return (uint8_t)((dec*131+b*7+(dec >> 8)) & 0xff); }

int run(int iters, unsigned seed, bool stamp) {
  const i64 KV = 16, W = 4, NH = 2, BS = 4;
  PageCore pc(32, KV, 2, 2, BS, NH, 256, 64);
  Native N;
  LineageCfg c; c.window = 6; c.block_steps = BS; c.memory_blocks = 2; c.nh = NH;
  N.lin.configure(c, W);
  N.inj.configure(W, {1, 2});
  N.lin.side[0].chunk_steps = N.lin.side[1].chunk_steps = 8;
  std::vector<i64> jobs(W*2*2, -1), epoch(W, 0);   // (no injury-job pins, no teacher rings: recent keys and candidates only)
  N.tr.jobs = jobs.data(); N.tr.jobs_k = 2; N.tr.epoch = epoch.data(); N.tr.ring = 0; N.tr_on = true;
  std::atomic<bool> done{false};
  std::atomic<i64> errors{0}, reads{0}, gcs{0}, retired{0};
  std::thread driver([&] {
    std::mt19937 r(seed);
    i64 ep[W] = {0}, t[W] = {0}, idx[W] = {0};
    int sl = 0;
    uint8_t frame[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (i64 s = 1; s <= iters; s++) {
      pc.set_stamp(s);
      std::vector<i64> ws;
      for (i64 w = 0; w < W; w++) if (r()%2) ws.push_back(w);
      if (ws.empty()) ws.push_back(r()%W);
      i64 n = (i64)ws.size();
      std::vector<uint8_t> first(n);
      for (i64 j = 0; j < n; j++) { first[j] = t[ws[j]] == 0 || r()%40 == 0; if (first[j]) { ep[ws[j]]++; t[ws[j]] = 0; } }
      // the rows (decision ids are sequential: the K/V bytes are a function of the id the store will give)
      std::vector<uint8_t> kv(n*KV); std::vector<float> z(n*2, 0);
      std::vector<i64> steps(n), dec(n), kvs(n), pend(n*NH, 0), nb(n, 8);
      std::vector<int32_t> sch(n, 1); std::vector<const uint8_t *> fr(n, frame);
      i64 next;
      { std::lock_guard<std::mutex> l(pc.mu); next = pc.next_decision; }
      {
        std::lock_guard<std::mutex> l(N.lin.mu);
        for (i64 j = 0; j < n; j++) {
          i64 h = N.lin.head[ws[j]];
          steps[j] = first[j] || h < 0 ? 0 : N.lin.logs[h].end();
        }
      }
      for (i64 j = 0; j < n; j++) for (int b = 0; b < KV; b++) kv[j*KV+b] = pat(next+j, b);
      pc.write_decisions_raw(n, kv.data(), KV, z.data(), 2, steps.data(), s, first.data(), sch.data(), fr.data(), nb.data(), pend.data(),
                             dec.data(), kvs.data());
      std::vector<i64> bid(n, -1);
      for (i64 j = 0; j < n; j++)
        if ((steps[j]+1) % BS == 0) {
          std::vector<float> vals(2, 1.f); std::vector<i64> src(BS, -1), srcver(BS, -1); i64 end = steps[j];
          pc.write_blocks_raw(1, vals.data(), 2, &end, s, src.data(), srcver.data(), &bid[j]);
        }
      if (r()%32 == 0) std::this_thread::yield();   // (a GC may scan here: the rows are not in the lineage yet)
      {
        std::lock_guard<std::mutex> l(N.lin.mu);
        Lineage &L = N.lin;
        for (i64 j = 0; j < n; j++) {
          i64 w = ws[j];
          N.inj.observe(w, first[j], ep[w], t[w], r()%5 == 0);
          if (first[j]) L.set_head_locked(w, L.new_log_locked(7, w, ep[w], 0));
          i64 sn = L.snap_row_locked(w, pend.data()+j*NH, s);
          N.inj.record(L, w, ep[w], t[w], sn);
          if (r()%6 == 0) L.ctx_put_locked(r()%3, sn);
          L.push_locked(w, dec[j], kvs[j]);
          if (bid[j] >= 0) L.push_block_locked(w, bid[j], steps[j]);
          L.side_record_locked(sl, w, idx[w]++, first[j], sn);
          L.release_locked(sn);
          t[w]++;
        }
        if (s % 300 == 0) {   // a hand-off (the driver paused in production; here the driver itself)
          std::vector<i64> cw, ci;
          for (auto &o : L.side[sl].open) { cw.push_back(o.first); ci.push_back(o.second.first); }
          L.side_carry_locked(1-sl, sl, cw.data(), ci.data(), (i64)cw.size());
          sl = 1-sl;
          for (i64 w = 0; w < W; w++) idx[w] = 0;
          for (i64 w : cw) idx[w] = 1;
        }
        L.published = s;
      }
    }
    done = true;
  });
  auto reader = [&](unsigned sd) {
    std::mt19937 r(sd);
    while (!done) {
      i64 snap = -1; Parts p;
      {
        std::lock_guard<std::mutex> l(N.lin.mu);
        Lineage &L = N.lin;
        std::vector<i64> cand;
        for (auto &e : L.ctx) cand.push_back(e.second.second);
        for (auto &w : L.side[r()%2].starts) for (auto &e : w.second) cand.push_back(e.second);
        if (cand.empty()) continue;
        snap = cand[r()%cand.size()];
        L.retain_locked(snap);   // (a SnapPin)
        p = L.parts_locked(snap);
      }
      for (size_t q = 0; q < p.kvs.size(); q++) {
        i64 id = p.kvs[q];
        const uint8_t *row = pc.kv_page(id/pc.page)+(id % pc.page)*KV;
        i64 d;
        {   // the decision of this K/V row: the snapshot's log names it
          std::lock_guard<std::mutex> l(N.lin.mu);
          const LogData &g = N.lin.logs[N.lin.snaps[snap].log];
          d = g.dec[p.kv_steps[q]-g.offset];
        }
        for (int b = 0; b < KV; b++) if (row[b] != pat(d, b)) { errors++; break; }
        reads++;
      }
      { std::lock_guard<std::mutex> l(N.lin.mu); N.lin.release_locked(snap); }
    }
  };
  std::thread r1(reader, seed+2), r2(reader, seed+3);
  std::thread retirer([&] {
    std::mt19937 r(seed+5);
    while (!done) {
      {
        std::lock_guard<std::mutex> l(N.lin.mu);
        N.lin.ctx_complete_locked(r()%3);
        std::unordered_set<i64> live;
        for (auto &e : N.lin.ctx) if (r()%3 == 0) live.insert(e.first);
        retired += N.lin.ctx_retire_locked(live, r()%2 == 0);
        i64 f; N.inj.reclaim(N.lin, N.tr, f);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200+r()%800));
    }
  });
  std::thread collector([&] {
    std::mt19937 r(seed+4);
    while (!done) {
      collect(pc, N.lin, stamp ? -1 : 0);
      gcs++;
      std::this_thread::sleep_for(std::chrono::microseconds(50+r()%300));
    }
  });
  driver.join(); r1.join(); r2.join(); retirer.join(); collector.join();
  Lineage &L = N.lin;
  std::printf("lineage (stamp rule %s): %d steps, %lld row reads, %lld collections, %lld freed rows, %lld kept by the stamp, "
              "%lld contexts retired, %lld logs / %lld snapshots live, %lld reallocs, %lld wrong bytes\n", stamp ? "on" : "off", iters,
              (long long)reads.load(), (long long)gcs.load(), (long long)pc.freed_kv, (long long)pc.stamp_kept, (long long)retired.load(),
              (long long)L.n_logs, (long long)L.n_snaps, (long long)pc.reallocs, (long long)errors.load());
  return errors.load() == 0 && pc.reallocs == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20000;
  unsigned seed = argc > 2 ? (unsigned)atoi(argv[2]) : 1;
  bool stamp = !(argc > 3 && std::string(argv[3]) == "nostamp");
  int rc = run(iters, seed, stamp);
  std::printf("%s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
