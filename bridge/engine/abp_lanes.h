// B58d: the lane service's state machine in C++ (torch free; abp_actor.cpp runs it on the lane thread and gives it the
// GPU work through LaneOps). What block_lanes.BlockLaneService (B54 driver mode) and tok_lanes.LaneService did in
// Python, request by request:
//
//   intake       a lane's 5-byte request message -> ready (with the oldest / newest read time)
//   due          tok_lanes.window_due with the --cf-lane-wait / --cf-lane-quiet constants and ACTIVE_S
//   control      service_control: status == 3 (a worker released a point): the lane's pending request of that point
//                cancelled, cf_snap cleared, the point released (its contexts' owner completes), status 0; every
//                RECLAIM_S the transient contexts no pin / lane_t / v2_t ring names retired, the injury sources reclaimed
//   submit       one round at a time: the requests taken (distinct, ascending), a branch's first request whose point is
//                not captured yet deferred (DEFER_S, then cancelled as an orphan), begins (restore from the point's
//                snapshot: LaneOps.restore), repeated sequence numbers answered from last_answers, cancelled ones
//                answered as such, the rest grouped per teacher generation (the main policy's version first) into jobs
//   job done     the rows' lineage (a log at an episode's first row, the snapshot before the step, the pushes), the
//                answers (actions, logp, value, value_global, the generation's version), the lane steps
//   finish       the contexts registered (ident halves into lane_t), last_answers, lane_io / lane_out written, replies
//
// Shared memory with the workers is used as the Python path used it: a worker writes its request (lane_rows,
// lane_io [0, nh) / [10] / [11], block mode) before its message; the answer (lane_io [5, 5 + nh), lane_out, lane_t)
// is written before the reply. Threads: only the lane thread runs a LaneCore; the lineage it touches is under
// Lineage::mu (abp_lineage.h).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "abp_lineage.h"

namespace abpc {

struct LaneCfg {
  i64 lanes = 0, nh = 0, tw = 0;            // lanes (= workers), action heads, lane_t width
  double wait = .002, quiet = .0005, active_s = .05, defer_s = 2.0, reclaim_s = 0.1;
  i64 pid = 0;                              // the lane logs' episode key's first part (os.getpid())
};

struct LaneShared {   // the lanes' shared memory (tok_teacher2 / tok_cf / block_protocol layouts)
  const uint8_t *rows = nullptr; i64 row_size = 0; i64 o_episode = 0; int s_episode = 8;   // lane_rows [L] ROW
  int32_t *io = nullptr; i64 io_cols = 12;   // lane_io [L, 12] int32
  double *out = nullptr; i64 out_cols = 6;   // lane_out [L, 6] float64
  float *t = nullptr;                        // lane_t [L, tw] float32
  const float *v2_t = nullptr; i64 v2_ring = 0;   // v2_t [L, V2_RING, tw]
  const i64 *mode = nullptr;                 // block_arrays[1] [L] (replay)
  i64 *status = nullptr; i64 pts_per = 0;    // block_arrays[0] [L, bank rows]
  const i64 *kept = nullptr; i64 kept_cols = 0;   // block_arrays[4] [L, keep]
  i64 *cf_snap = nullptr;                    // [L]
};

struct LaneAnswer {
  i64 seq = 0;
  std::vector<i64> a;   // [nh]
  float lp = 0, v = 0, vg = 0;
  i64 version = -1;
  std::vector<float> refs;   // [tw]
  bool cancelled = false;
};

struct LaneJobOut {   // a job's rows after their step (the torch side fills it)
  std::vector<float> out;          // [k, nh + 3]
  std::vector<i64> step_before, dec, kvs, bid;
  std::vector<uint8_t> first, complete;
  i64 version = 0;
};

// the GPU / lineage-store side a LaneCore needs (abp_actor.cpp; fakes in the tests)
struct LaneOps {
  std::function<void(i64 gen, i64 lane, i64 snap)> restore;   // state.load + the lane row's head (Lineage::branch) + ring
  // a job for generation gen's rows: (job id, gen, rows [k * row_size] copied, pending [k, nh], lanes, CRN keys (seq,
  // lane), replay); the torch side steps it (merged or lane-only) and calls LaneCore::job_done
  std::function<void(i64 job, i64 gen, std::vector<uint8_t> rows, std::vector<i64> pend, std::vector<i64> lanes, std::vector<i64> seq,
                     std::vector<uint8_t> replay)> submit;
  std::function<void(const std::vector<i64> &lanes)> reply;   // the reply bytes to the lanes' pipes
  std::function<i64(i64 gen)> gen_version;                     // the generation actor's version (its points' version)
  std::function<i64()> main_version;
  std::function<i64(i64 gen)> base;                            // the generation's first arena row
};

struct Round {
  std::vector<i64> idx, seq;
  std::vector<i64> a;              // [n, nh]
  std::vector<float> lp, v, vg;
  std::vector<i64> version;
  std::vector<uint8_t> cancelled;
  std::vector<float> refs;         // [n, tw]
  std::set<i64> jobs;
  bool ready = false;
  double t0 = 0;
  struct Put { i64 j, token, snap; };
  std::vector<Put> puts;           // (registered in finish, right before lane_t)
};

struct LaneJobInfo {
  i64 gen, version;
  std::vector<i64> pos, lanes, tokens;
  std::vector<uint8_t> replay;
  std::vector<i64> pend;           // [k, nh]
  std::vector<i64> episode;        // the rows' episodes (a first row's new log)
};

struct LaneCore {
  LaneCfg cfg;
  LaneShared sh;
  LaneOps ops;
  Native *nat = nullptr;
  // state (only the lane thread)
  std::vector<i64> ready;
  double first_t = 0, last_t = 0, reclaimed_t = -1e9;
  std::vector<double> seen;
  std::unordered_map<i64, i64> cancelled_req;   // lane -> seq
  std::unordered_map<i64, LaneAnswer> last;
  std::unordered_map<i64, i64> lane_point, lane_steps;
  std::unordered_map<i64, double> deferred_begin;
  std::unique_ptr<Round> fly;
  std::unordered_map<i64, LaneJobInfo> jobs;
  i64 job_serial = 0;
  bool reply_first = false;   // (tests: the control without the answer-before-reply order; must fail)
  // counters
  i64 n_rounds = 0, n_rows = 0, n_jobs = 0, n_deferred = 0, n_orphan = 0, n_cancel_scans = 0, n_retired = 0, n_begins = 0;
  double held_s = 0, fly_s = 0;
  // (D1 / B58d) per answered request: the window's wait (read -> the window rule's due time), the wait behind the
  // caller (due -> taken: d1 the driver's main steps), the GPU (its restore + its job), read -> reply
  std::unordered_map<i64, double> t_read;
  std::unordered_map<i64, double> gpu_lane;   // (this round: a lane's restore + job seconds)
  std::vector<double> win_s, drv_s, gpu_s, rt_s;
  struct Taken { i64 lane; double read, due, take; };
  std::vector<Taken> taken_now;

  void configure(const LaneCfg &c, const LaneShared &s, Native *n) {
    cfg = c; sh = s; nat = n;
    seen.assign(cfg.lanes, -1e9);
  }

  // ---- intake / window
  void intake(i64 lane, double now) {
    t_read.emplace(lane, now);   // (the first read of the lane's pending request)
    if (ready.empty()) first_t = now;
    last_t = now;
    ready.push_back(lane);
  }
  static bool window_due(i64 n_ready, double first, double lastt, double now, double wait, double quiet, i64 active) {
    if (n_ready <= 0) return false;
    if (wait <= 0 || now-first >= wait) return true;
    if (quiet > 0 && now-lastt >= quiet) return true;
    return active >= 0 && n_ready >= active;
  }
  bool due(double now) {
    if (ready.empty()) return false;
    if (window_due(1, first_t, last_t, now, cfg.wait, cfg.quiet, -1)) return true;
    std::set<i64> rd(ready.begin(), ready.end());
    i64 active = 0;
    for (i64 l = 0; l < cfg.lanes; l++) active += seen[l] >= now-cfg.active_s || rd.count(l);
    return window_due((i64)rd.size(), first_t, last_t, now, cfg.wait, cfg.quiet, active);
  }
  // the time until the window may end (the lane thread's poll timeout), -1: nothing waits
  double wait_left(double now) const {
    if (ready.empty()) return -1;
    double w = std::max(0.0, cfg.wait-(now-first_t));
    if (cfg.quiet > 0) w = std::min(w, std::max(0.0, cfg.quiet-(now-last_t)));
    return w;
  }

  // ---- service_control
  void control(double now) {
    n_cancel_scans++;
    Lineage &L = nat->lin;
    i64 np = cfg.lanes*sh.pts_per;
    for (i64 point = 0; point < np; point++) {
      if (__atomic_load_n(&sh.status[point], __ATOMIC_ACQUIRE) != 3) continue;
      i64 worker = point/sh.pts_per;
      i64 begin = sh.io[worker*sh.io_cols+11];
      i64 requested = begin > 0 ? begin-1 : (lane_point.count(worker) ? lane_point[worker] : -1);
      if (requested == point) cancelled_req[worker] = sh.io[worker*sh.io_cols+10];
      if (sh.cf_snap && sh.cf_snap[worker] == point+1) sh.cf_snap[worker] = 0;
      {
        std::lock_guard<std::mutex> l(L.mu);
        i64 tok = nat->pts.release(L, point, ops.main_version ? ops.main_version() : -1);
        if (tok >= 0) L.ctx_complete_locked(tok);
      }
      __atomic_store_n(&sh.status[point], (i64)0, __ATOMIC_RELEASE);
    }
    if (now-reclaimed_t < cfg.reclaim_s) return;
    reclaimed_t = now;
    std::unordered_set<i64> live;
    if (sh.kept) for (i64 i = 0; i < cfg.lanes*sh.kept_cols; i++) live.insert(sh.kept[i]);
    auto ref = [&](const float *r) { return (i64)(uint32_t)r[0] | ((i64)(uint32_t)r[1] << 16); };
    if (sh.t) for (i64 l = 0; l < cfg.lanes; l++) live.insert(ref(sh.t+l*cfg.tw));
    if (sh.v2_t) for (i64 i = 0; i < cfg.lanes*sh.v2_ring; i++) live.insert(ref(sh.v2_t+i*cfg.tw));
    std::lock_guard<std::mutex> l(L.mu);
    n_retired += L.ctx_retire_locked(live, true);
    if (nat->tr_on) { i64 f; nat->inj.reclaim(L, nat->tr, f); }
  }

  // ---- a round
  bool point_known(i64 point) { std::lock_guard<std::mutex> l(nat->lin.mu); return nat->pts.table.count(point) > 0; }
  abpc::Point point_of(i64 point) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    auto it = nat->pts.table.find(point);
    if (it == nat->pts.table.end()) throw std::runtime_error("lane: a request of point " + std::to_string(point) + " that is not captured");
    return it->second;
  }
  bool submit(double now) {
    if (fly) return false;   // (one round in flight at a time)
    std::set<i64> taken(ready.begin(), ready.end());
    double first0 = first_t;
    double due_th = cfg.wait <= 0 ? first_t : first_t+cfg.wait;   // (when the window rule would have let this round go)
    if (cfg.quiet > 0) due_th = std::min(due_th, last_t+cfg.quiet);
    {
      i64 active = 0;
      for (i64 l = 0; l < cfg.lanes; l++) active += seen[l] >= now-cfg.active_s || taken.count(l);
      if ((i64)taken.size() >= active) due_th = std::min(due_th, last_t);
    }
    ready.clear();
    if (taken.empty()) return false;
    std::vector<i64> keep, back;
    for (i64 lane : taken) {
      i64 begin = sh.io[lane*sh.io_cols+11], seq = sh.io[lane*sh.io_cols+10];
      auto c = cancelled_req.find(lane);
      bool canc = c != cancelled_req.end() && c->second == seq;
      if (begin > 0 && !point_known(begin-1) && !canc) {
        auto d = deferred_begin.emplace(lane, now).first;
        if (now-d->second < cfg.defer_s) { back.push_back(lane); continue; }
        cancelled_req[lane] = seq; n_orphan++;
      }
      deferred_begin.erase(lane);
      keep.push_back(lane);
    }
    if (!back.empty()) {
      first_t = now; last_t = now;   // (Python: lane_first_t set when lane_ready was empty, lane_last_t = t0)
      ready = back; n_deferred += (i64)back.size();
    }
    if (keep.empty()) return false;
    i64 n = (i64)keep.size(), nh = cfg.nh;
    held_s += std::max(0.0, now-(back.empty() ? first0 : first_t));   // (Python reads lane_first_t after the deferral)
    taken_now.clear(); gpu_lane.clear();
    for (i64 l : keep) {
      auto it = t_read.find(l);
      double rd = it != t_read.end() ? it->second : now;
      if (it != t_read.end()) t_read.erase(it);
      taken_now.push_back(Taken{l, rd, std::max(due_th, rd), now});
    }
    for (i64 l : keep) seen[l] = now;
    auto f = std::make_unique<Round>();
    f->idx = keep; f->t0 = now;
    f->seq.resize(n); f->a.resize(n*nh); f->lp.assign(n, 0); f->v.assign(n, 0); f->vg.assign(n, 0); f->version.assign(n, -1);
    f->cancelled.assign(n, 0); f->refs.assign(n*cfg.tw, 0.f);
    std::vector<uint8_t> rows(n*sh.row_size);
    std::vector<i64> pend(n*nh);
    for (i64 j = 0; j < n; j++) {
      i64 l = keep[j];
      memcpy(&rows[j*sh.row_size], sh.rows+l*sh.row_size, sh.row_size);
      for (i64 h = 0; h < nh; h++) pend[j*nh+h] = sh.io[l*sh.io_cols+h];
      f->seq[j] = sh.io[l*sh.io_cols+10];
    }
    fly = std::move(f);
    Round &r = *fly;
    // _begin
    for (i64 j = 0; j < n; j++) {
      i64 lane = keep[j], seq = r.seq[j];
      auto c = cancelled_req.find(lane);
      if (c != cancelled_req.end() && c->second == seq) { sh.io[lane*sh.io_cols+11] = 0; continue; }
      auto la = last.find(lane);
      if (la != last.end() && la->second.seq == seq) continue;
      i64 begin = sh.io[lane*sh.io_cols+11];
      if (begin == 0) continue;
      if (begin < 0) throw std::runtime_error("block-memory search requires an explicit point snapshot");
      i64 point = begin-1;
      abpc::Point p = point_of(point);
      double tr = mono_now();
      ops.restore(p.gen, lane, p.snap);
      gpu_lane[lane] += mono_now()-tr;
      lane_point[lane] = point; lane_steps[lane] = 0;
      sh.io[lane*sh.io_cols+11] = 0;
      n_begins++;
    }
    // _submit_jobs: a = pending, the groups per generation
    for (i64 j = 0; j < n; j++) for (i64 h = 0; h < nh; h++) r.a[j*nh+h] = pend[j*nh+h];
    std::map<i64, std::vector<i64>> groups; std::vector<i64> order;   // (gen -> positions, in first-appearance order)
    std::vector<uint8_t> replay(n, 0);
    for (i64 j = 0; j < n; j++) {
      i64 lane = keep[j], seq = r.seq[j];
      auto c = cancelled_req.find(lane);
      if (c != cancelled_req.end() && c->second == seq) { r.cancelled[j] = 1; continue; }
      auto la = last.find(lane);
      if (la != last.end() && la->second.seq == seq) {
        const LaneAnswer &A = la->second;
        for (i64 h = 0; h < nh; h++) r.a[j*nh+h] = A.a[h];
        r.lp[j] = A.lp; r.v[j] = A.v; r.vg[j] = A.vg; r.version[j] = A.version;
        std::copy(A.refs.begin(), A.refs.end(), r.refs.begin()+j*cfg.tw); r.cancelled[j] = A.cancelled;
        continue;
      }
      auto lp_ = lane_point.find(lane);
      if (lp_ == lane_point.end()) throw std::runtime_error("lane: a request without a branch begin");
      i64 gen = point_of(lp_->second).gen;
      replay[j] = sh.mode ? sh.mode[lane] != 0 : 0;
      if (!groups.count(gen)) order.push_back(gen);
      groups[gen].push_back(j);
    }
    i64 mv = ops.main_version();
    std::stable_sort(order.begin(), order.end(), [&](i64 a, i64 b) { return (ops.gen_version(a) != mv) < (ops.gen_version(b) != mv); });
    for (i64 gen : order) {
      LaneJobInfo ji; ji.gen = gen; ji.version = ops.gen_version(gen);
      std::vector<uint8_t> sub; std::vector<i64> seqs;
      for (i64 j : groups[gen]) {
        i64 lane = keep[j];
        ji.pos.push_back(j); ji.lanes.push_back(lane); ji.replay.push_back(replay[j]);
        ji.tokens.push_back(point_of(lane_point[lane]).token);
        for (i64 h = 0; h < nh; h++) ji.pend.push_back(pend[j*nh+h]);
        ji.episode.push_back(rd_int(&rows[j*sh.row_size]+sh.o_episode, sh.s_episode));
        sub.insert(sub.end(), rows.begin()+j*sh.row_size, rows.begin()+(j+1)*sh.row_size);
        seqs.push_back(r.seq[j]);
      }
      i64 job = ++job_serial;
      r.jobs.insert(job); n_jobs++;
      std::vector<i64> lanes = ji.lanes, jp = ji.pend; std::vector<uint8_t> rp = ji.replay;
      jobs.emplace(job, std::move(ji));
      double tj = mono_now();
      std::vector<i64> jl = lanes;
      ops.submit(job, gen, std::move(sub), std::move(jp), std::move(lanes), std::move(seqs), std::move(rp));
      double dj = mono_now()-tj;
      for (i64 l : jl) gpu_lane[l] += dj;
    }
    r.ready = r.jobs.empty();
    n_rounds++; n_rows += n;
    return true;
  }
  // a job's rows stepped (on_lane_record + _answer_rows): the lane rows' lineage, the answers, the lane steps
  void job_done(i64 job, const LaneJobOut &o) {
    auto it = jobs.find(job);
    if (it == jobs.end()) throw std::runtime_error("lane: a record of an unknown job");
    LaneJobInfo ji = std::move(it->second); jobs.erase(it);
    Round &r = *fly;
    i64 k = (i64)ji.lanes.size(), nh = cfg.nh, ow = nh+3;
    Lineage &L = nat->lin;
    {
      std::lock_guard<std::mutex> l(L.mu);
      for (i64 q = 0; q < k; q++) {
        i64 lane = ji.lanes[q], row = ops.base(ji.gen)+lane;
        if (o.first[q]) L.set_head_locked(row, L.new_log_locked(cfg.pid, lane, ji.episode[q], 0));
        i64 g = L.head.at(row);
        if (g < 0 || L.logs[g].end() != o.step_before[q]) throw std::runtime_error("lane: a row's log end differs from its step");
        if (!ji.replay[q]) {
          i64 sn = L.snap_row_locked(row, &ji.pend[q*nh], o.version);
          r.puts.push_back(Round::Put{ji.pos[q], ji.tokens[q], sn});
        }
        L.push_locked(row, o.dec[q], o.kvs[q]);
        if (o.complete[q]) L.push_block_locked(row, o.bid[q], o.step_before[q]);
      }
    }
    {
      std::lock_guard<std::mutex> l(L.mu);
      double tn = mono_now();
      for (i64 q = 0; q < k; q++) { auto lp_ = lane_point.find(ji.lanes[q]); if (lp_ != lane_point.end()) nat->pts.first_step(lp_->second, tn); }
    }
    for (i64 q = 0; q < k; q++) {
      i64 j = ji.pos[q];
      for (i64 h = 0; h < nh; h++) r.a[j*nh+h] = (i64)o.out[q*ow+h];
      r.lp[j] = o.out[q*ow+nh]; r.v[j] = o.out[q*ow+nh+1]; r.vg[j] = o.out[q*ow+nh+2];
      r.version[j] = ji.version;
      lane_steps[ji.lanes[q]]++;
    }
    r.jobs.erase(job);
    if (r.jobs.empty()) r.ready = true;
  }
  // _finish + LaneService.finish: contexts registered right before lane_t, last_answers, the shared answers, replies
  void finish(double now) {
    if (!fly || !fly->ready) return;
    std::unique_ptr<Round> f = std::move(fly);
    i64 n = (i64)f->idx.size(), nh = cfg.nh;
    {
      std::lock_guard<std::mutex> l(nat->lin.mu);
      for (auto &p : f->puts) {
        i64 ident = nat->lin.ctx_put_locked(p.token, p.snap);
        nat->lin.release_locked(p.snap);
        float *rf = &f->refs[p.j*cfg.tw];
        rf[0] = (float)(ident & 65535); rf[1] = (float)(ident >> 16); rf[cfg.tw-1] = 1.0f;
      }
    }
    for (i64 j = 0; j < n; j++) {
      LaneAnswer A; A.seq = f->seq[j]; A.a.assign(&f->a[j*nh], &f->a[j*nh]+nh); A.lp = f->lp[j]; A.v = f->v[j]; A.vg = f->vg[j];
      A.version = f->version[j]; A.refs.assign(&f->refs[j*cfg.tw], &f->refs[j*cfg.tw]+cfg.tw); A.cancelled = f->cancelled[j];
      last[f->idx[j]] = std::move(A);
    }
    if (reply_first) ops.reply(f->idx);
    for (i64 j = 0; j < n; j++) {
      i64 l = f->idx[j];
      if (sh.t) memcpy(sh.t+l*cfg.tw, &f->refs[j*cfg.tw], cfg.tw*sizeof(float));
      sh.out[l*sh.out_cols+5] = -(double)f->cancelled[j];
      for (i64 h = 0; h < nh; h++) sh.io[l*sh.io_cols+5+h] = (int32_t)f->a[j*nh+h];
      sh.out[l*sh.out_cols+0] = f->v[j]; sh.out[l*sh.out_cols+1] = f->lp[j]; sh.out[l*sh.out_cols+2] = (double)f->version[j];
      sh.out[l*sh.out_cols+3] = f->vg[j];
    }
    for (i64 j = 0; j < n; j++) sh.out[f->idx[j]*sh.out_cols+4] = (double)f->seq[j];   // (the requests' numbers, written last)
    __atomic_thread_fence(__ATOMIC_RELEASE);
    if (!reply_first) ops.reply(f->idx);
    fly_s += now-f->t0;
    double tr = mono_now();
    for (auto &tk : taken_now) {
      win_s.push_back(std::max(0.0, tk.due-tk.read)); drv_s.push_back(std::max(0.0, tk.take-tk.due));
      auto g = gpu_lane.find(tk.lane); gpu_s.push_back(g != gpu_lane.end() ? g->second : 0.0); rt_s.push_back(tr-tk.read);
    }
    taken_now.clear();
  }
  // tick (B54 driver mode): control, a ready round finished, a due round submitted (and finished when it needed no job)
  bool tick(double now) {
    control(now);
    bool did = false;
    if (fly) {
      if (!fly->ready) return false;
      finish(now); did = true;
    }
    if (!ready.empty() && due(now)) {
      submit(now);
      if (fly && fly->ready) finish(now);
      did = true;
    }
    return did;
  }
};

}  // namespace abpc
