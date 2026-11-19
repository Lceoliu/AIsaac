// P package phase 1, milestone B (2026-11-19): the thin TokPolicy driver's thread (P-driver-design 11.1-11.6).
// Included by abp_tok_driver.cpp after TokMiddle. One C++ thread owns the main workers' fds: it reads their messages
// (abptok::MsgReader), takes the ready records (worker order, at most n a call, one per worker), copies each into a
// staging block (fixed fields + the frame prefix), decides go (Episodes.arrive's rule), replies at once to an episode's
// end that needs no action, runs the call's middle (TokMiddle::run_raw) for the others, writes their actions / values
// into the sampler and replies, and queues one record per record taken for the Python consumer (book, store), in
// sequence order. Credits -- a staging block, queue room, output-ring rows -- are taken for a whole call before it is
// accepted: when one is short the records wait (so do their workers); nothing is overwritten or dropped. A stop and a
// fatal wake every waiter (one mutex, one condition variable: every waiter re-checks every condition).
#pragma once
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <poll.h>
#include <unistd.h>

struct Rec {
  i64 seq, call, worker, slot, block, brow, kind, go, ring, version, e_bucket, n_call;
  i64 act[8], pend[8];   // (pend: a go record's under_way, after Episodes.arrive's resets; others: before them)
  double logp, value, vg;
  i64 tc[4];
  i64 replied;     // (Codex 17-3) 1: the reply was written whole; 0: the write failed (the worker gone)
  double origin;   // (Codex 17-5) the worker's ep_archive column when its record was taken (before the reply)
};
enum RecKind { K_REC = 0, K_END = 1, K_GONE = 2 };   // a record (go or not), a worker's 'x', a worker gone (EOF)

struct TokDriver {
  TokMiddle *mid;
  py::object mid_keep;   // (the Python object owning mid)
  // the sampler
  i64 W = 0, row_bytes = 0, nh = 0, act_stride = 0;
  uint8_t *rows_base = nullptr; int32_t *actions = nullptr; double *main_value = nullptr, *main_vg = nullptr;
  int64_t *cf_snap = nullptr; std::vector<int> fds;
  // ROW fields for go (Episodes.arrive's rule)
  i64 off_done = 0, off_first = 0, off_t = 0, done_branch = 4, cap_t = 0, trunc = 0;
  // the staging blocks [M, n, row_bytes] (Python owns them: numpy ROW arrays the consumer reads in place)
  uint8_t *blocks = nullptr; i64 M = 0, n = 0;
  // (tok_priv) each slot's PRIV record beside its row (sampler.priv [2 W]) into priv blocks [M, n] at the same rows
  uint8_t *priv_base = nullptr, *priv_blocks = nullptr; i64 priv_size = 0;
  std::vector<int> block_free;
  // the output ring: K rows, absolute sequence numbers (row = seq mod K; head: the next; released: below it free)
  i64 K = 0, ring_head = 0, ring_released = 0;
  // the queue
  std::deque<Rec> q; i64 q_cap = 0, seq = 0, call_id = 0, version = 0;
  std::vector<i64> pending;      // [W, nh]: the action under way per worker (C++ owns it, P-driver-design 11.6)
  std::vector<i64> ready_slot;   // [W]: -1, or the slot of the worker's record waiting to be taken
  std::vector<char> gone;
  // (Codex 17-3) a worker's end / EOF read while its record waits to be taken: queued after that record (the consumer
  // sees a worker's records before its end); 0 none, else the RecKind
  std::vector<i64> gone_after;
  abptok::MsgReader reader;
  std::vector<double> f_at;      // an 'F' waiting for its 'R' since (seconds)
  std::vector<double> ready_t;   // when each worker's waiting record was read (the ready -> reply clock)
  double rr_s = 0; i64 rr_n = 0;  // ready -> reply, summed over the replies (ends included)
  // control
  std::mutex mu; std::condition_variable cv;
  std::thread th; bool running = false, stop_req = false, pause_req = false, paused = false;
  // (Codex 17-2) each pause() a new generation; the thread acknowledges the generation it paused in (ack_gen), so a
  // resume() then a pause() before the thread woke cannot leave the consumer waiting on an acknowledgement never given
  i64 pause_gen = 0, ack_gen = -1;
  int fatal_code = 0; std::string fatal_text; i64 fatal_worker = -1;
  i64 accepted = 0, answered = 0, ended_records = 0, replies = 0, calls_n = 0, credit_waits = 0;
  i64 reply_fail = 0, cancelled = 0, cf_snaps = 0;
  // (Codex 17-5) the sampler's ep_archive column (stats[:, ep_archive], float64, a row stride apart)
  const uint8_t *origin_base = nullptr; i64 origin_stride = 0;
  double t_credit = 0, t_call = 0, t_read = 0;
  double r_timeout_s = 2.0;
  // fault injection (tests): the consumer's lag is the test's; here, a delay before each call
  double test_call_delay = 0;
  // (the gates' tool p_chain: each record's sampling uniforms from (worker + 1, its ROW 't') -- B36's ukeys -- so a
  //  Python replay of the same calls draws the same numbers; off in training: the graphs' own draws)
  bool test_ukeys = false;
  std::vector<i64> uk_c, ut_c;

  explicit TokDriver(py::object middle) : mid_keep(middle) { mid = middle.cast<TokMiddle *>(); }
  ~TokDriver() { halt(); }

  void set_sampler(i64 rows_ptr, i64 workers, std::vector<int> fds_, i64 actions_ptr, i64 act_stride_, i64 main_value_ptr,
                   i64 main_vg_ptr, i64 cf_snap_ptr) {
    rows_base = (uint8_t *)rows_ptr; W = workers; fds = fds_; actions = (int32_t *)actions_ptr; act_stride = act_stride_;
    main_value = (double *)main_value_ptr; main_vg = (double *)main_vg_ptr; cf_snap = (int64_t *)cf_snap_ptr;
    row_bytes = mid->L.row_bytes; nh = mid->L.nh;
    TORCH_CHECK((i64)fds.size() == W && nh <= 8, "set_sampler: one fd per worker, at most 8 heads");
    pending.assign((size_t)(W*nh), 0); ready_slot.assign((size_t)W, -1); gone.assign((size_t)W, 0);
    gone_after.assign((size_t)W, 0);
    reader.reset(W); f_at.assign((size_t)W, -1.0); ready_t.assign((size_t)W, 0.0);
  }
  void set_rules(i64 off_done_, i64 off_first_, i64 off_t_, i64 done_branch_, i64 cap_t_, i64 trunc_) {
    off_done = off_done_; off_first = off_first_; off_t = off_t_; done_branch = done_branch_; cap_t = cap_t_; trunc = trunc_;
  }
  void set_blocks(i64 ptr, i64 M_, i64 n_, i64 q_cap_) {
    blocks = (uint8_t *)ptr; M = M_; n = n_;
    TORCH_CHECK(n_ >= 1 && n_ <= mid->L.n, "set_blocks: a block holds 1 .. the graphs' rows");
    block_free.clear(); for (i64 b = M-1; b >= 0; b--) block_free.push_back((int)b);
    q_cap = q_cap_ > 0 ? q_cap_ : M*n;
  }
  void set_priv(i64 src_ptr, i64 itemsize, i64 blocks_ptr) {
    priv_base = (uint8_t *)src_ptr; priv_size = itemsize; priv_blocks = (uint8_t *)blocks_ptr;
  }
  void set_origin(i64 ptr, i64 stride) { origin_base = (const uint8_t *)ptr; origin_stride = stride; }
  void set_ring_rows(i64 K_) { K = K_; ring_head = ring_released = 0; }   // (K >= n: checked at start)
  void set_version(i64 v) { std::lock_guard<std::mutex> g(mu); version = v; }
  void set_pending(i64 w, std::vector<i64> p) {
    std::lock_guard<std::mutex> g(mu);
    for (i64 j = 0; j < nh; j++) pending[w*nh+j] = p[(size_t)j];
  }

  // ---- control (Python)
  void start() {
    std::lock_guard<std::mutex> g(mu);
    if (running) return;
    TORCH_CHECK(blocks && rows_base && K > 0, "TokDriver.start: sampler, blocks and ring first");
    TORCH_CHECK(K >= n, "TokDriver.start: the ring (", K, " rows) must hold a call (", n, " rows)");
    stop_req = pause_req = paused = false; running = true;
    th = std::thread([this] { loop(); });
  }
  void halt() {
    { std::lock_guard<std::mutex> g(mu); if (!running) return; stop_req = true; }
    cv.notify_all();
    if (th.joinable()) th.join();
    std::lock_guard<std::mutex> g(mu); running = false;
  }
  void stop() { py::gil_scoped_release nogil; halt(); }
  void pause() { { std::lock_guard<std::mutex> g(mu); pause_req = true; pause_gen++; } cv.notify_all(); }
  void resume() { { std::lock_guard<std::mutex> g(mu); pause_req = false; paused = false; } cv.notify_all(); }
  bool is_paused() { std::lock_guard<std::mutex> g(mu); return paused_ack(); }
  // (Codex 17-1) one consistent view for the consumer's drain: (paused and acknowledged, queue length, next seq)
  py::tuple drain_state() {
    std::lock_guard<std::mutex> g(mu);
    return py::make_tuple(paused_ack(), (i64)q.size(), seq);
  }
  // (Codex 17-2) after stop(): every record already read (and every one still arriving) answered without an action
  // -- a cancel, the workers told to close first (sampler.control[0]) -- until each worker has ended or the time is up;
  // a half-read message is read on to its end. Their ends / EOFs are queued for the consumer (take). Returns how many
  // were cancelled and how many workers are still open.
  py::tuple close_serve(double timeout_s) {
    i64 n_c = 0, open_w = 0;
    {
      py::gil_scoped_release nogil;
      TORCH_CHECK(!running, "close_serve: stop() first");
      const double end = now()+timeout_s;
      while (true) {
        std::vector<i64> ws;
        {
          std::lock_guard<std::mutex> g(mu);
          for (i64 w = 0; w < W; w++) if (ready_slot[w] >= 0) { ws.push_back(w); ready_slot[w] = -1; }
        }
        for (i64 w : ws) {
          reply(w); n_c++;
          std::lock_guard<std::mutex> g(mu);
          cancelled++;
          push_gone_after(w);
        }
        open_w = 0;
        for (i64 w = 0; w < W; w++) if (!gone[w] && !reader.closed[w]) open_w++;
        if (!open_w || now() > end) break;
        read_fds(20);
      }
    }
    cv.notify_all();
    return py::make_tuple(n_c, open_w);
  }
  py::tuple fatal() { std::lock_guard<std::mutex> g(mu); return py::make_tuple(fatal_code, fatal_worker, py::bytes(fatal_text)); }
  i64 last_seq() { std::lock_guard<std::mutex> g(mu); return seq; }
  std::vector<i64> pending_of(i64 w) {
    std::lock_guard<std::mutex> g(mu);
    return std::vector<i64>(pending.begin()+w*nh, pending.begin()+(w+1)*nh);
  }

  // ---- the consumer (Python): records in sequence order; then the credits back
  py::dict take(i64 max_n, double timeout) {
    std::vector<Rec> out;
    {
      py::gil_scoped_release nogil;
      std::unique_lock<std::mutex> lk(mu);
      if (q.empty() && timeout > 0)
        cv.wait_for(lk, std::chrono::duration<double>(timeout),
                    [&] { return !q.empty() || stop_req || fatal_code || paused_ack(); });
      while (!q.empty() && (i64)out.size() < max_n) { out.push_back(q.front()); q.pop_front(); }
    }
    cv.notify_all();   // (queue room)
    const i64 m = (i64)out.size();
    py::dict d;
    auto col = [&](const char *name, i64 Rec::*f) {
      py::array_t<i64> a(m); i64 *p = a.mutable_data(); for (i64 i = 0; i < m; i++) p[i] = out[i].*f; d[name] = a;
    };
    col("seq", &Rec::seq); col("call", &Rec::call); col("worker", &Rec::worker); col("slot", &Rec::slot);
    col("block", &Rec::block); col("brow", &Rec::brow); col("kind", &Rec::kind); col("go", &Rec::go);
    col("ring", &Rec::ring); col("version", &Rec::version); col("e_bucket", &Rec::e_bucket); col("n_call", &Rec::n_call);
    col("replied", &Rec::replied);
    py::array_t<double> origin(m);
    for (i64 i = 0; i < m; i++) origin.mutable_data()[i] = out[i].origin;
    d["origin"] = origin;
    py::array_t<i64> act({m, nh}), pend({m, nh}), tc({m, (i64)4});
    py::array_t<double> logp(m), value(m), vg(m);
    for (i64 i = 0; i < m; i++) {
      for (i64 j = 0; j < nh; j++) { act.mutable_data()[i*nh+j] = out[i].act[j]; pend.mutable_data()[i*nh+j] = out[i].pend[j]; }
      for (int j = 0; j < 4; j++) tc.mutable_data()[i*4+j] = out[i].tc[j];
      logp.mutable_data()[i] = out[i].logp; value.mutable_data()[i] = out[i].value; vg.mutable_data()[i] = out[i].vg;
    }
    d["actions"] = act; d["pending"] = pend; d["tc"] = tc; d["logp"] = logp; d["value"] = value; d["vg"] = vg;
    return d;
  }
  void release_block(i64 b) {
    { std::lock_guard<std::mutex> g(mu);
      TORCH_CHECK(b >= 0 && b < M, "release_block: ", b);
      for (int f : block_free) TORCH_CHECK(f != b, "release_block: block ", b, " released twice");
      block_free.push_back((int)b); }
    cv.notify_all();
  }
  void release_ring(i64 upto) {
    { std::lock_guard<std::mutex> g(mu); TORCH_CHECK(upto <= ring_head, "release_ring: past the head"); if (upto > ring_released) ring_released = upto; }
    cv.notify_all();
  }
  py::dict counters() {
    std::lock_guard<std::mutex> g(mu);
    py::dict d;
    d["accepted"] = accepted; d["answered"] = answered; d["ended_records"] = ended_records; d["replies"] = replies;
    d["calls"] = calls_n; d["credit_waits"] = credit_waits; d["queue"] = (i64)q.size(); d["blocks_free"] = (i64)block_free.size();
    d["ring_used"] = ring_head-ring_released; d["ring_head"] = ring_head; d["t_credit_s"] = t_credit; d["t_call_s"] = t_call;
    d["t_read_s"] = t_read; d["rr_s"] = rr_s; d["rr_n"] = rr_n;
    d["reply_fail"] = reply_fail; d["cancelled"] = cancelled; d["cf_snaps"] = cf_snaps; d["pause_gen"] = pause_gen;
    return d;
  }

 private:
  static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
  void set_fatal(int code, i64 w, const std::string &text) {
    { std::lock_guard<std::mutex> g(mu); if (!fatal_code) { fatal_code = code; fatal_worker = w; fatal_text = text; } }
    cv.notify_all();
  }
  bool paused_ack() const { return !running || (pause_req && paused && ack_gen == pause_gen); }
  // (Codex 17-3) the reply written whole (EINTR retried, a short write continued); false: the write failed (the
  // worker's end gone). Counters under the lock (counters() reads them while the thread runs).
  bool reply(i64 w) {
    static const uint8_t R[5] = {0, 0, 0, 1, 'a'};
    size_t off = 0;
    while (off < 5) {
      const ssize_t k = ::write(fds[(size_t)w], R+off, 5-off);
      if (k > 0) off += (size_t)k;
      else if (k < 0 && errno == EINTR) continue;
      else break;
    }
    const bool ok = off == 5;
    const double dt = now()-ready_t[(size_t)w];
    std::lock_guard<std::mutex> g(mu);
    if (ok) replies++; else reply_fail++;
    rr_s += dt; rr_n++;
    return ok;
  }
  // (under mu) a worker's end / EOF held back behind its record, queued now
  void push_gone_after(i64 w) {
    if (!gone_after[(size_t)w]) return;
    Rec r{}; r.seq = seq++; r.call = -1; r.worker = w; r.slot = -1; r.block = -1; r.brow = -1;
    r.kind = gone_after[(size_t)w]; r.ring = -1; r.version = version; r.replied = 1;
    q.push_back(r);
    gone_after[(size_t)w] = 0;
  }
  // read every readable fd: records into ready_slot, ends / gone into the queue, fatal / bad into the fatal state
  void read_fds(int timeout_ms) {
    std::vector<struct pollfd> p; std::vector<i64> who;
    for (i64 w = 0; w < W; w++) if (!gone[w] && !reader.closed[w]) { p.push_back({fds[(size_t)w], POLLIN, 0}); who.push_back(w); }
    if (p.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms)); return; }
    int r = ::poll(p.data(), p.size(), timeout_ms);
    std::vector<abptok::MsgEvent> ev;
    if (r > 0) {
      uint8_t buf[4096];
      for (size_t j = 0; j < p.size(); j++) {
        if (!(p[j].revents & (POLLIN | POLLHUP | POLLERR))) continue;
        ssize_t got = ::read(p[j].fd, buf, sizeof buf);
        if (got > 0) reader.feed(who[j], buf, (size_t)got, ev);
        else if (got == 0 || (errno != EAGAIN && errno != EINTR)) reader.eof(who[j], ev);
      }
    }
    const double t = now();
    for (i64 w = 0; w < W; w++)
      if (reader.await_r[w] && !reader.closed[w]) {
        if (f_at[w] < 0) f_at[w] = t;
        else if (t-f_at[w] > r_timeout_s) reader.r_timeout(w, ev);
      }
    if (ev.empty()) return;
    {
      std::lock_guard<std::mutex> g(mu);
      for (auto &e : ev) {
        if (e.kind == abptok::MsgEvent::RECORD) {
          if (ready_slot[e.worker] >= 0) {
            if (!fatal_code) { fatal_code = 3; fatal_worker = e.worker; fatal_text = "two records of one worker before its answer"; }
            continue;
          }
          ready_slot[e.worker] = e.slot;
          ready_t[(size_t)e.worker] = t;
        } else if (e.kind == abptok::MsgEvent::END || e.kind == abptok::MsgEvent::GONE) {
          gone[e.worker] = 1;
          const i64 kind = e.kind == abptok::MsgEvent::END ? K_END : K_GONE;
          if (ready_slot[e.worker] >= 0) { gone_after[(size_t)e.worker] = kind; continue; }   // (after its record)
          Rec r{}; r.seq = seq++; r.call = -1; r.worker = e.worker; r.slot = -1; r.block = -1; r.brow = -1;
          r.kind = kind; r.ring = -1; r.version = version; r.replied = 1;
          q.push_back(r);
        } else if (!fatal_code) {   // FATAL (86: a visible-v3 protocol error, the worker's reason) / BAD (2: malformed)
          fatal_code = e.kind == abptok::MsgEvent::FATAL ? 86 : 2; fatal_worker = e.worker; fatal_text = e.text;
        }
      }
    }
    cv.notify_all();
  }
  bool is_go(const uint8_t *r) const {
    const i64 done = r[off_done];
    if (done == 0) return true;
    int32_t t; std::memcpy(&t, r+off_t, 4);
    bool trunc_end = false;
    if (cap_t > 0 && trunc && done == 3) {
      const bool at_cap = (i64)t >= cap_t;
      trunc_end = (at_cap && (trunc & 1)) || (!at_cap && (trunc & 2));
    }
    return done == done_branch || trunc_end;
  }

  void loop() {
    try {
      loop_body();
    } catch (const std::exception &ex) {
      set_fatal(5, -1, std::string("TokDriver thread: ") + ex.what());
    } catch (...) {
      set_fatal(5, -1, "TokDriver thread: an unknown exception");
    }
    std::lock_guard<std::mutex> g(mu);
    paused = true; ack_gen = pause_gen;
    cv.notify_all();
  }
  void loop_body() {
    std::vector<i64> take_w, slots, pend_c, widx_c, ring_c, cs, cd, gi;
    std::vector<char> rep;
    std::vector<double> org;
    std::vector<float> out_c;
    std::vector<uint8_t> callrows;
    while (true) {
      {
        std::unique_lock<std::mutex> lk(mu);
        if (stop_req || fatal_code) break;
        if (pause_req) {
          if (!paused || ack_gen != pause_gen) { paused = true; ack_gen = pause_gen; cv.notify_all(); }
          const i64 gen = pause_gen;
          cv.wait_for(lk, std::chrono::milliseconds(20),
                      [&] { return !pause_req || stop_req || fatal_code || pause_gen != gen; });
          if (pause_req && !stop_req && !fatal_code && pause_gen == gen) { lk.unlock(); read_fds(0); }
          continue;
        }
      }
      double t0 = now();
      read_fds(1);
      {
        std::lock_guard<std::mutex> g(mu);
        t_read += now()-t0;
      }
      take_w.clear();
      {
        std::lock_guard<std::mutex> g(mu);
        if (fatal_code || stop_req) break;
        if (pause_req) continue;
        for (i64 w = 0; w < W && (i64)take_w.size() < n; w++) if (ready_slot[w] >= 0) take_w.push_back(w);
      }
      if (take_w.empty()) continue;
      const i64 k = (i64)take_w.size();
      // credits for the whole call: a block, queue room for k records, k contiguous ring rows
      i64 b = -1, r0 = 0;
      {
        const double tc0 = now();
        std::unique_lock<std::mutex> lk(mu);
        auto ok = [&] {   // (ring rows wrap: a call's rows are (r0 + j) mod K, the consumer gathers them by row)
          return !block_free.empty() && (i64)q.size()+k <= q_cap && ring_head+k-ring_released <= K;
        };
        auto halt_ = [&] { return stop_req || fatal_code || pause_req; };   // (Codex 17-2: before any credit)
        if (halt_()) continue;
        if (!ok()) {
          credit_waits++;
          while (!ok() && !halt_()) {   // (Codex 17-3: the fds read on while short -- F / R, timeouts, EOFs)
            cv.wait_for(lk, std::chrono::milliseconds(20), [&] { return ok() || halt_(); });
            if (!ok() && !halt_()) { lk.unlock(); read_fds(0); lk.lock(); }
          }
          t_credit += now()-tc0;
          if (!ok() || halt_()) continue;   // (a stop / a fatal / a pause: the records stay ready)
        }
        b = block_free.back(); block_free.pop_back();
        r0 = ring_head; ring_head += k;
        slots.assign((size_t)k, -1);
        for (i64 i = 0; i < k; i++) { slots[i] = ready_slot[take_w[i]]; ready_slot[take_w[i]] = -1; }
      }
      // the records into the block (fixed fields + the frame prefix: A-4's take_prefix rule), go, pending, cf_snap
      uint8_t *blk = blocks+b*n*row_bytes;
      const i64 off_frame = mid->off_frame, off_bytes = mid->off_bytes, cap = row_bytes-off_frame;
      std::vector<char> go((size_t)k);
      std::vector<std::array<i64, 8>> before((size_t)k);
      rep.assign((size_t)k, 1); org.assign((size_t)k, 0.0);
      if (origin_base)
        for (i64 i = 0; i < k; i++) std::memcpy(&org[i], origin_base+take_w[i]*origin_stride, 8);
      for (i64 i = 0; i < k; i++) {
        const uint8_t *src = rows_base+slots[i]*row_bytes;
        int32_t vb; std::memcpy(&vb, src+off_bytes, 4);
        const i64 e = (vb >= 0 && vb <= cap) ? off_frame+(((i64)vb+3) & ~(i64)3) : row_bytes;
        std::memcpy(blk+i*row_bytes, src, (size_t)e);
        if (priv_base)   // (the worker may overwrite its slot once answered: its PRIV record copied with the row)
          std::memcpy(priv_blocks+(b*n+i)*priv_size, priv_base+slots[i]*priv_size, (size_t)priv_size);
        go[i] = is_go(blk+i*row_bytes);
      }
      gi.clear();
      std::unique_lock<std::mutex> lk_p(mu);   // (Codex 17-6: pending_of() reads it)
      for (i64 i = 0; i < k; i++) {
        const i64 w = take_w[i];
        const uint8_t *r = blk+i*row_bytes;
        const bool first = r[off_first] != 0, ended = r[off_done] != 0;
        for (i64 j = 0; j < nh; j++) before[i][j] = pending[w*nh+j];
        if (ended && !go[i]) for (i64 j = 0; j < nh; j++) pending[w*nh+j] = 0;   // (Episodes.arrive: an end's)
        if (first && go[i]) for (i64 j = 0; j < nh; j++) pending[w*nh+j] = 0;    // (an episode's first)
        if (go[i]) gi.push_back(i);
      }
      lk_p.unlock();
      const i64 kg = (i64)gi.size();
      {   // (the call's ring rows: its go records only -- the rows reserved for its ends given back at once, else the
          //  consumer, releasing up to its last go record's row + 1, would never free them: a full ring, a stall)
        std::lock_guard<std::mutex> g(mu);
        ring_head = r0+kg;
      }
      const uint8_t *call_p = blk;
      bool front = true; for (i64 j = 0; j < kg; j++) if (gi[j] != j) front = false;
      if (!front) {   // (the go records contiguous for the call)
        callrows.resize((size_t)(kg*row_bytes));
        for (i64 j = 0; j < kg; j++) std::memcpy(&callrows[j*row_bytes], blk+gi[j]*row_bytes, (size_t)row_bytes);
        call_p = callrows.data();
      }
      pend_c.assign((size_t)(std::max<i64>(kg, 1)*nh), 0); widx_c.clear(); ring_c.clear(); cs.clear(); cd.clear();
      uk_c.clear(); ut_c.clear();
      for (i64 j = 0; j < kg; j++) {
        const i64 w = take_w[gi[j]];
        for (i64 h = 0; h < nh; h++) pend_c[j*nh+h] = pending[w*nh+h];
        widx_c.push_back(w); ring_c.push_back((r0+j) % K);
        if (test_ukeys) {
          int32_t t_; std::memcpy(&t_, blk+gi[j]*row_bytes+off_t, 4);
          uk_c.push_back(w+1); ut_c.push_back((i64)t_);
        }
        if (cf_snap && cf_snap[w] > 0) { cs.push_back(w); cd.push_back(cf_snap[w]-1); }
      }
      for (i64 i = 0; i < k; i++) if (!go[i]) rep[i] = reply(take_w[i]);   // (an episode's end that needs no action)
      i64 rec = -1, eb = -1, graph = -1;
      out_c.assign((size_t)(std::max<i64>(kg, 1)*mid->cols), 0.0f);
      if (kg > 0) {
        if (test_call_delay > 0) std::this_thread::sleep_for(std::chrono::duration<double>(test_call_delay));
        const double tc0 = now();
        int code = 0;
        try {
          code = mid->run_raw(call_p, kg, pend_c.data(), widx_c.data(), mid->L.t_on ? ring_c.data() : nullptr, cs, cd,
                              out_c.data(), &rec, &eb, &graph, test_ukeys ? uk_c.data() : nullptr,
                              test_ukeys ? ut_c.data() : nullptr);
        } catch (const std::exception &ex) {
          set_fatal(5, take_w[gi[0]], std::string("TokMiddle: ") + ex.what());
          break;
        }
        {
          std::lock_guard<std::mutex> g(mu);
          t_call += now()-tc0; cf_snaps += (i64)cs.size();
        }
        if (code) {
          set_fatal(86, take_w[gi[(size_t)std::max<i64>(rec, 0)]], "visible-v3: the check refused the call (code " +
                    std::to_string(code) + ", record " + std::to_string(rec) + ")");
          break;
        }
        for (i64 v : cs) cf_snap[v] = 0;   // (after the copy: the stream was synchronized in run_raw)
        {
          std::lock_guard<std::mutex> g(mu);
          for (i64 j = 0; j < kg; j++) {
            const i64 w = take_w[gi[j]];
            const float *o = out_c.data()+j*mid->cols;
            for (i64 h = 0; h < nh; h++) pending[w*nh+h] = (i64)o[h];
          }
        }
        for (i64 j = 0; j < kg; j++) {
          const i64 w = take_w[gi[j]];
          const float *o = out_c.data()+j*mid->cols;
          for (i64 h = 0; h < nh; h++) actions[w*act_stride+h] = (int32_t)(i64)o[h];
          if (main_value) main_value[w] = (double)o[nh+1];
          if (main_vg) main_vg[w] = mid->cols >= nh+7 ? (double)o[nh+2] : (double)o[nh+1];
          rep[gi[j]] = reply(w);
        }
      }
      {
        std::lock_guard<std::mutex> g(mu);
        const i64 cid = call_id++;
        calls_n += kg > 0; accepted += k;
        const bool vg_on = mid->cols >= nh+7;
        for (i64 i = 0, j = 0; i < k; i++) {
          Rec r{}; r.seq = seq++; r.call = cid; r.worker = take_w[i]; r.slot = slots[i]; r.block = b; r.brow = i;
          r.kind = K_REC; r.go = go[i]; r.version = version; r.e_bucket = eb; r.n_call = k; r.ring = -1;
          r.replied = rep[i]; r.origin = org[i];
          for (i64 h = 0; h < nh; h++) r.pend[h] = before[i][h];
          if (go[i]) {
            for (i64 h = 0; h < nh; h++) r.pend[h] = pend_c[j*nh+h];   // (the pending the call used: after the resets)
            const float *o = out_c.data()+j*mid->cols;
            for (i64 h = 0; h < nh; h++) r.act[h] = (i64)o[h];
            r.logp = o[nh]; r.value = o[nh+1]; r.vg = vg_on ? o[nh+2] : o[nh+1];
            const i64 at = nh+2+(vg_on ? 1 : 0);
            for (int c = 0; c < 4; c++) r.tc[c] = (i64)std::llround(o[at+c]);
            r.ring = r0+j; j++; answered++;
          } else {
            ended_records++;
          }
          q.push_back(r);
        }
        for (i64 i = 0; i < k; i++) push_gone_after(take_w[i]);   // (a worker's end after its last record)
      }
      cv.notify_all();
    }
  }
};
