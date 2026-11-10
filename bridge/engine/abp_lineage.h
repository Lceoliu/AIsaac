// B58c: the actor's lineage in C++ (torch free, like abp_core.h): what block_paged.EpisodeLog / PagedSnapshot,
// BlockRollout (the rollout's memory sidecar), block_lanes.ContextRegistry and the store's collect() were in Python.
//
//   Lineage      logs (one per episode of an arena row, append-only decisions / K/V rows / blocks, an offset for a
//                branch's copied prefix) and snapshots (log, step, pending, version) with reference counts; a head per
//                arena row (the log that row writes, its step = the log's end); branch (a restored lane's child log),
//                parts (snapshot_parts: what a restore / the learner reads), the live rows of the GC
//   Sidecar      a rollout's chunk-start snapshots and each worker's open decision (BlockRollout.record / carry)
//   Contexts     ident -> (owner token, snapshot): ContextRegistry (put / complete / retire, idents from 1, < 2^32)
//   collect()    PagedStore.collect over the native lineage: every row a live snapshot or head can read stays
//
// Threads: the driver creates logs and snapshots and appends (its steps); Python materializes snapshots (hand-off,
// teacher records), pins / unpins them and runs the GC between collects (B59: with the record stamps of abp_core.h
// while the driver runs). Every structure here is under Lineage::mu.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "abp_core.h"

namespace abpc {

constexpr int MAX_NH = 8;   // action heads a snapshot's pending row holds

struct LineageCfg {
  i64 window = 0, block_steps = 0, memory_blocks = 0, nh = 0;
};

struct LogData {
  i64 key[3] = {0, 0, 0};   // the episode (EpisodeLog.episode: pid / tag, row or worker, episode)
  i64 offset = 0;           // the first decision's step (a branch's copied prefix starts there)
  std::vector<i64> dec, kvs, blk, blk_end;
  i64 refs = 0;             // snapshots + heads naming it
  bool used = false;
  i64 end() const { return offset+(i64)dec.size(); }
};

struct SnapData {
  i64 log = -1, step = 0, version = 0;
  i64 pend[MAX_NH] = {0};
  i64 refs = 0;
  bool used = false;
};

// the parts of one snapshot (PagedStore.snapshot_parts): decisions [lo, step) with their K/V rows, the open block's
// decisions [step - step % bs, step), the newest memory_blocks complete blocks ending before step
struct Parts {
  i64 step = 0, lo = 0;
  std::vector<i64> kv_steps, kvs, pdec, pkvs, bid, ends;
};

struct Sidecar {   // BlockRollout: per worker its chunk starts (index -> snapshot) and its open decision
  i64 chunk_steps = 128;
  std::map<i64, std::map<i64, i64>> starts;   // worker -> index -> snapshot (ordered: payload order is insertion order in
  std::map<i64, std::vector<i64>> order;      // Python, kept in `order`: worker -> indices in insertion order)
  std::vector<i64> worker_order;              // workers in first-insertion order
  std::map<i64, std::pair<i64, i64>> open;    // worker -> (index, snapshot)
};

struct Lineage {
  LineageCfg cfg;
  std::vector<LogData> logs;
  std::vector<i64> free_logs;
  std::vector<SnapData> snaps;
  std::vector<i64> free_snaps;
  std::vector<i64> head;   // per arena row: its log (-1: none)
  Sidecar side[2];
  // contexts (ContextRegistry)
  i64 next_ident = 1;
  std::unordered_map<i64, std::pair<i64, i64>> ctx;   // ident -> (owner token, snapshot)
  std::unordered_set<i64> completed;                  // owner tokens whose point was released
  // counters
  i64 n_logs = 0, n_snaps = 0, made_logs = 0, made_snaps = 0, gc_runs = 0;
  i64 published = 0;   // the record stamp up to which every row the driver wrote is in this lineage (the GC stamp rule)
  std::mutex mu;

  void configure(const LineageCfg &c, i64 rows) {
    std::lock_guard<std::mutex> l(mu);
    cfg = c;
    if (cfg.nh > MAX_NH) throw std::runtime_error("Lineage: more action heads than MAX_NH");
    head.assign(rows, -1);
  }

  // ---- logs and snapshots (callers hold mu: the *_locked functions)
  i64 new_log_locked(i64 k0, i64 k1, i64 k2, i64 offset) {
    i64 id;
    if (!free_logs.empty()) { id = free_logs.back(); free_logs.pop_back(); }
    else { id = (i64)logs.size(); logs.emplace_back(); }
    LogData &g = logs[id];
    g.key[0] = k0; g.key[1] = k1; g.key[2] = k2; g.offset = offset;
    g.dec.clear(); g.kvs.clear(); g.blk.clear(); g.blk_end.clear();
    g.refs = 0; g.used = true;
    n_logs++; made_logs++;
    return id;
  }
  void log_ref_locked(i64 id) { logs.at(id).refs++; }
  void log_unref_locked(i64 id) {
    LogData &g = logs.at(id);
    if (--g.refs > 0) return;
    g.used = false;
    std::vector<i64>().swap(g.dec); std::vector<i64>().swap(g.kvs);
    std::vector<i64>().swap(g.blk); std::vector<i64>().swap(g.blk_end);
    free_logs.push_back(id); n_logs--;
  }
  void set_head_locked(i64 row, i64 log) {
    i64 old = head.at(row);
    if (log >= 0) log_ref_locked(log);
    head[row] = log;
    if (old >= 0) log_unref_locked(old);
  }
  i64 new_snap_locked(i64 log, i64 step, const i64 *pend, i64 version) {
    i64 id;
    if (!free_snaps.empty()) { id = free_snaps.back(); free_snaps.pop_back(); }
    else { id = (i64)snaps.size(); snaps.emplace_back(); }
    SnapData &s = snaps[id];
    s.log = log; s.step = step; s.version = version; s.refs = 1; s.used = true;
    for (i64 h = 0; h < MAX_NH; h++) s.pend[h] = h < cfg.nh && pend ? pend[h] : 0;
    log_ref_locked(log);
    n_snaps++; made_snaps++;
    return id;
  }
  void retain_locked(i64 s) {
    SnapData &d = snaps.at(s);
    if (!d.used) throw std::runtime_error("Lineage: retain of a freed snapshot");
    d.refs++;
  }
  void release_locked(i64 s) {
    SnapData &d = snaps.at(s);
    if (!d.used || d.refs <= 0) throw std::runtime_error("Lineage: release of a freed snapshot");
    if (--d.refs > 0) return;
    d.used = false;
    i64 lg = d.log; d.log = -1;
    free_snaps.push_back(s); n_snaps--;
    log_unref_locked(lg);
  }
  // a snapshot of an arena row before its step: its head log at the head's end (PagedSnapshot(store, logs[slot],
  // step_host[row], pending, version)); the caller checks the step is the log's end
  i64 snap_row_locked(i64 row, const i64 *pend, i64 version) {
    i64 lg = head.at(row);
    if (lg < 0) throw std::runtime_error("Lineage: a snapshot of a row without a log");
    return new_snap_locked(lg, logs[lg].end(), pend, version);
  }
  void push_locked(i64 row, i64 dec, i64 kvs) {
    LogData &g = logs.at(head.at(row));
    g.dec.push_back(dec); g.kvs.push_back(kvs);
  }
  void push_block_locked(i64 row, i64 bid, i64 end) {
    LogData &g = logs.at(head.at(row));
    g.blk.push_back(bid); g.blk_end.push_back(end);
  }
  // EpisodeLog.decisions(lo, hi) as indices into the log's arrays
  void check_range(const LogData &g, i64 lo, i64 hi) const {
    if (lo < g.offset || hi > g.end() || lo > hi) throw std::runtime_error("snapshot reads outside its retained log prefix");
  }
  // EpisodeLog.blocks_before(step, limit): [lo, at) of the log's blocks
  static void blocks_before(const LogData &g, i64 step, i64 limit, i64 &lo, i64 &at) {
    at = (i64)(std::lower_bound(g.blk_end.begin(), g.blk_end.end(), step)-g.blk_end.begin());
    lo = std::max<i64>(0, at-limit);
  }
  // EpisodeLog.branch(step): a self-contained child log (decisions [max(offset, step - max(window, bs)), step), the
  // newest memory_blocks blocks before step), the parent's episode key
  i64 branch_locked(i64 s) {
    SnapData d = snaps.at(s);
    LogData g = logs.at(d.log);   // (a copy: new_log_locked may grow the vector)
    i64 lo = std::max(g.offset, d.step-std::max(cfg.window, cfg.block_steps));
    check_range(g, lo, d.step);
    i64 blo, bat; blocks_before(g, d.step, cfg.memory_blocks, blo, bat);
    i64 id = new_log_locked(g.key[0], g.key[1], g.key[2], lo);
    LogData &c = logs[id];
    c.dec.assign(g.dec.begin()+(lo-g.offset), g.dec.begin()+(d.step-g.offset));
    c.kvs.assign(g.kvs.begin()+(lo-g.offset), g.kvs.begin()+(d.step-g.offset));
    c.blk.assign(g.blk.begin()+blo, g.blk.begin()+bat);
    c.blk_end.assign(g.blk_end.begin()+blo, g.blk_end.begin()+bat);
    return id;
  }
  Parts parts_locked(i64 s) const {
    const SnapData &d = snaps.at(s);
    const LogData &g = logs.at(d.log);
    Parts p; p.step = d.step;
    p.lo = std::max(g.offset, d.step-cfg.window);
    check_range(g, p.lo, d.step);
    for (i64 t = p.lo; t < d.step; t++) { p.kv_steps.push_back(t); p.kvs.push_back(g.kvs[t-g.offset]); }
    i64 ph = d.step % cfg.block_steps;
    check_range(g, d.step-ph, d.step);
    for (i64 t = d.step-ph; t < d.step; t++) { p.pdec.push_back(g.dec[t-g.offset]); p.pkvs.push_back(g.kvs[t-g.offset]); }
    i64 blo, bat; blocks_before(g, d.step, cfg.memory_blocks, blo, bat);
    p.bid.assign(g.blk.begin()+blo, g.blk.begin()+bat); p.ends.assign(g.blk_end.begin()+blo, g.blk_end.begin()+bat);
    return p;
  }

  // ---- the GC's live rows (PagedStore.collect): every (log, step) a live snapshot or a head names
  // keep_kv [cap_d], keep_b [cap_b] set to 1 for live rows; live_raw gets the live decisions (not the blocks'
  // sources: the caller adds b_src of the live blocks)
  void live_rows_locked(uint8_t *keep_kv, i64 cap_d, uint8_t *keep_b, i64 cap_b, std::vector<i64> &live_raw) {
    i64 W = std::max(cfg.window, cfg.block_steps);
    std::unordered_map<i64, std::vector<i64>> steps;   // log -> live steps
    for (i64 s = 0; s < (i64)snaps.size(); s++) if (snaps[s].used) steps[snaps[s].log].push_back(snaps[s].step);
    for (i64 r = 0; r < (i64)head.size(); r++) if (head[r] >= 0) steps[head[r]].push_back(logs[head[r]].end());
    for (auto &kv : steps) {
      const LogData &g = logs[kv.first];
      std::vector<i64> &st = kv.second;
      std::sort(st.begin(), st.end()); st.erase(std::unique(st.begin(), st.end()), st.end());
      i64 n = (i64)g.dec.size();
      if (n) {   // decisions [max(step - W, offset), step), clipped to the log
        std::vector<i64> diff(n+1, 0);
        for (i64 t : st) {
          i64 hi = std::min(std::max<i64>(t-g.offset, 0), n), lo = std::min(std::max<i64>(std::max(t-W, g.offset)-g.offset, 0), n);
          diff[lo]++; diff[hi]--;
        }
        i64 run = 0;
        for (i64 i = 0; i < n; i++) {
          run += diff[i];
          if (run > 0) {
            i64 k = g.kvs[i];
            if (k >= 0 && k < cap_d) keep_kv[k] = 1;
            live_raw.push_back(g.dec[i]);
          }
        }
      }
      i64 nb = (i64)g.blk.size();
      if (nb) {
        std::vector<i64> diff(nb+1, 0);
        for (i64 t : st) {
          i64 at = (i64)(std::lower_bound(g.blk_end.begin(), g.blk_end.end(), t)-g.blk_end.begin());
          diff[std::max<i64>(at-cfg.memory_blocks, 0)]++; diff[at]--;
        }
        i64 run = 0;
        for (i64 i = 0; i < nb; i++) {
          run += diff[i];
          if (run > 0) { i64 b = g.blk[i]; if (b >= 0 && b < cap_b) keep_b[b] = 1; }
        }
      }
    }
  }

  // ---- the rollout sidecar (BlockRollout)
  void side_record_locked(int sl, i64 worker, i64 index, bool first, i64 snap) {
    Sidecar &sc = side[sl];
    auto it = sc.starts.find(worker);
    bool fresh = it == sc.starts.end();
    std::map<i64, i64> &st = sc.starts[worker];
    if (fresh) sc.worker_order.push_back(worker);
    bool add = st.empty() || first || index-st.rbegin()->first >= sc.chunk_steps;
    // (Python: index - max(starts) with the dict's keys; indices only grow within a rollout, so max == last)
    if (add) {
      auto old = st.find(index);
      if (old != st.end()) release_locked(old->second);
      else sc.order[worker].push_back(index);
      retain_locked(snap); st[index] = snap;
    }
    auto op = sc.open.find(worker);
    retain_locked(snap);
    if (op != sc.open.end()) release_locked(op->second.second);
    sc.open[worker] = {index, snap};
  }
  void side_reset_locked(int sl) {
    Sidecar &sc = side[sl];
    for (auto &w : sc.starts) for (auto &e : w.second) release_locked(e.second);
    for (auto &o : sc.open) release_locked(o.second.second);
    sc.starts.clear(); sc.order.clear(); sc.worker_order.clear(); sc.open.clear();
  }
  // BlockRollout.carry: dst gets each worker's open decision of src as its index 0 (indices: the open decisions'
  // indices in src, checked)
  void side_carry_locked(int dst, int src, const i64 *workers, const i64 *indices, i64 n) {
    side_reset_locked(dst);
    Sidecar &d = side[dst], &s = side[src];
    for (i64 i = 0; i < n; i++) {
      auto it = s.open.find(workers[i]);
      if (it == s.open.end() || it->second.first != indices[i]) throw std::runtime_error("open decision and memory snapshot are out of sync");
      i64 snap = it->second.second;
      retain_locked(snap); retain_locked(snap);
      d.starts[workers[i]][0] = snap; d.order[workers[i]].push_back(0); d.worker_order.push_back(workers[i]);
      d.open[workers[i]] = {0, snap};
    }
  }

  // ---- contexts (ContextRegistry)
  i64 ctx_put_locked(i64 owner, i64 snap) {
    i64 ident = next_ident;
    if (ident >= ((i64)1 << 32)) throw std::runtime_error("search context reference space exhausted; restart the actor");
    next_ident++;
    retain_locked(snap);
    ctx[ident] = {owner, snap};
    return ident;
  }
  void ctx_complete_locked(i64 owner) { completed.insert(owner); }
  // retire(live, transient): every ident (transient) or every ident of a completed owner not in live; then the
  // completed owners no ident names any more are forgotten
  i64 ctx_retire_locked(const std::unordered_set<i64> &live, bool transient) {
    i64 n = 0;
    for (auto it = ctx.begin(); it != ctx.end();) {
      if ((transient || completed.count(it->second.first)) && !live.count(it->first)) {
        release_locked(it->second.second); it = ctx.erase(it); n++;
      } else ++it;
    }
    std::unordered_set<i64> owners;
    for (auto &e : ctx) owners.insert(e.second.first);
    for (auto it = completed.begin(); it != completed.end();) { if (!owners.count(*it)) it = completed.erase(it); else ++it; }
    return n;
  }
};

// ------------------------------------------------------------------------------------------- injury sources (B46)
// block_lanes.observe_rows / record_main / injury_context and the reclamation half of service_control: the main
// rows' snapshots by (worker, episode, step), each worker's recent keys and its hurt candidates; a source lives while
// a recent / candidate key, a worker's injury-job pin (jobs + the epoch seqlock) or a published teacher record names it.
struct Key3 {
  i64 w, e, t;
  bool operator==(const Key3 &o) const { return w == o.w && e == o.e && t == o.t; }
  bool operator<(const Key3 &o) const { return w != o.w ? w < o.w : e != o.e ? e < o.e : t < o.t; }
};
struct Key3Hash {
  size_t operator()(const Key3 &k) const { return std::hash<i64>()(k.w*1000003LL ^ k.e*7919LL ^ k.t); }
};
struct TeachRing {   // the workers' teacher-record rings (shared memory), for the reclamation
  const double *stats = nullptr; i64 stats_stride = 0, st_teach = 0;   // stats [W, ST] float64, column ST['teach']
  const i64 *commit = nullptr; i64 ring = 0;                        // block_arrays[5] [W, TEACH_RING]
  const uint8_t *rows = nullptr; i64 row_size = 0, o_episode = 0, o_t = 0; int s_episode = 8, s_t = 8;   // teach_rows [W, ring] ROW
  const i64 *jobs = nullptr; i64 jobs_k = 0;                        // block_arrays[2] [W, K, 2] (episode, hurt_at)
  const i64 *epoch = nullptr;                                       // block_arrays[3] [W]
};
struct Injury {
  std::vector<i64> depths;
  i64 recent_size = 2, workers = 0;
  std::unordered_map<Key3, i64, Key3Hash> sources;   // key -> snapshot (one ref)
  std::vector<std::vector<Key3>> recent;
  std::vector<std::set<Key3>> hurt;
  void configure(i64 n, std::vector<i64> d) {
    workers = n; depths = d; std::sort(depths.begin(), depths.end()); depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
    recent_size = (depths.empty() ? 0 : depths.back())+2;
    recent.assign(n, {}); hurt.assign(n, {});
  }
  void observe(i64 w, bool first, i64 e, i64 t, bool hurt_now) {
    if (first) { recent[w].clear(); hurt[w].clear(); }
    if (hurt_now)
      for (i64 d : depths) {
        Key3 k{w, e, t-1-d};
        if (k.t >= 0 && sources.count(k)) hurt[w].insert(k);
      }
  }
  void record(Lineage &lin, i64 w, i64 e, i64 t, i64 snap) {   // (lin.mu held)
    Key3 k{w, e, t};
    lin.retain_locked(snap);
    auto it = sources.find(k);
    if (it != sources.end()) { lin.release_locked(it->second); it->second = snap; } else sources.emplace(k, snap);
    std::vector<Key3> &r = recent[w];
    r.push_back(k);
    if ((i64)r.size() > recent_size) r.erase(r.begin(), r.end()-recent_size);
  }
  // the reclamation (service_control after its RECLAIM_S check): false when a worker was inside its pin update (odd
  // epoch or a changed one: Python returns before touching anything; so does this)
  bool reclaim(Lineage &lin, const TeachRing &tr, i64 &freed) {
    std::vector<i64> before(tr.epoch, tr.epoch+workers);
    for (i64 v : before) if (v % 2) return false;
    std::vector<i64> pins(tr.jobs, tr.jobs+workers*tr.jobs_k*2);
    for (i64 w = 0; w < workers; w++) if (__atomic_load_n(&tr.epoch[w], __ATOMIC_ACQUIRE) != before[w]) return false;
    std::set<Key3> live;
    for (auto &r : recent) live.insert(r.begin(), r.end());
    for (auto &h : hurt) live.insert(h.begin(), h.end());
    for (i64 w = 0; w < workers; w++) {
      for (i64 q = 0; q < tr.jobs_k; q++) {
        i64 ep = pins[(w*tr.jobs_k+q)*2], at = pins[(w*tr.jobs_k+q)*2+1];
        if (ep >= 0) for (i64 d : depths) live.insert(Key3{w, ep, at-1-d});
      }
      if (tr.ring <= 0 || !tr.rows) continue;   // (no teacher rings)
      i64 count = (i64)tr.stats[w*tr.stats_stride+tr.st_teach];
      for (i64 serial = std::max<i64>(0, count-tr.ring); serial < count; serial++) {
        i64 slot = serial % tr.ring;
        i64 c = tr.commit[w*tr.ring+slot];
        const uint8_t *row = tr.rows+(w*tr.ring+slot)*tr.row_size;
        i64 e = rd_int(row+tr.o_episode, tr.s_episode), t = rd_int(row+tr.o_t, tr.s_t);
        if (c == serial+1 && tr.commit[w*tr.ring+slot] == c) live.insert(Key3{w, e, t});
      }
    }
    freed = 0;
    for (auto it = sources.begin(); it != sources.end();) {
      if (!live.count(it->first)) { lin.release_locked(it->second); it = sources.erase(it); freed++; } else ++it;
    }
    return true;
  }
};

// ------------------------------------------------------------------------------------------- CF points (B54/B58c)
// block_lanes.capture_point / release_point and block_memory.TeacherGenerations' bookkeeping (the models stay in
// Python: a generation whose weights must be published first is handed to Python)
inline double mono_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// N teacher generations (TeacherGenerations' rule; N = 2 by default, configurable: Points::configure)
struct Generations {
  std::vector<i64> version{-1, -1}, refs{0, 0};
  double all_since = -1, all_s = 0;   // (D1 / the d2 gate) seconds every slot was pinned (no free generation)
  i64 n() const { return (i64)version.size(); }
  void configure(i64 slots) {
    if (slots < 1 || slots > 64) throw std::runtime_error("teacher generations: 1 .. 64 slots");
    version.assign(slots, -1); refs.assign(slots, 0); all_since = -1; all_s = 0;
  }
  // publish(version)'s choice: >= 0 the slot already holding it (no copy), -1 - i: slot i must be published (copy the
  // weights there), -100: none free (backpressure)
  i64 choose(i64 v) const {
    for (i64 i = 0; i < n(); i++) if (version[i] == v) return i;
    i64 best = -1;
    for (i64 i = 0; i < n(); i++) if (refs[i] == 0 && (best < 0 || version[i] < version[best])) best = i;
    return best < 0 ? -100 : -1-best;
  }
  void track(double now) {   // (after a refs change) the all-pinned interval
    bool all = true; for (i64 r : refs) all &= r > 0;
    if (all && all_since < 0) all_since = now;
    if (!all && all_since >= 0) { all_s += now-all_since; all_since = -1; }
  }
  double take_all_s(double now) {   // the seconds since the last take (an open interval counted up to now)
    if (all_since >= 0) { all_s += now-all_since; all_since = now; }
    double s = all_s; all_s = 0; return s;
  }
};
struct Point { i64 snap, gen, version, token; double t_capture = 0, t_first = -1; };
struct Points {
  Generations gens;
  std::unordered_map<i64, Point> table;
  i64 serial = 0;
  // (D1 / the d2 gate) per released point: capture -> first lane step, first lane step -> release, capture -> release,
  // policy versions crossed (the release's main version - the point's)
  std::vector<double> first_s, run_s, life_s, life_v;
  i64 refused = 0;
  // capture_point with the weights published: false (backpressure) or the point made (-1 from choose: Python publishes)
  i64 capture(Lineage &lin, i64 point, i64 snap, i64 version) {   // (lin.mu held) 2 made, -1 refused, -2 - i: publish slot i first
    if (table.count(point)) throw std::runtime_error("release a search point before reusing its bank slot");
    if (lin.snaps.at(snap).version != version) throw std::runtime_error("point snapshot and supplied actor weights have different versions");
    i64 c = gens.choose(version);
    if (c == -100) { refused++; return -1; }
    if (c < 0) return -2-(-1-c);
    gens.refs[c]++;
    double now = mono_now();
    gens.track(now);
    serial++;
    lin.retain_locked(snap);   // (snapshot.fork(): the same immutable snapshot)
    Point p{snap, c, gens.version[c], serial}; p.t_capture = now;
    table[point] = p;
    return 2;
  }
  void published(int slot, i64 v) { gens.version.at(slot) = v; }
  void first_step(i64 point, double now) {   // (the lane service: the point's first lane step answered)
    auto it = table.find(point);
    if (it != table.end() && it->second.t_first < 0) it->second.t_first = now;
  }
  // release_point: the point's token (its contexts' owner), -1 when there is no such point
  i64 release(Lineage &lin, i64 point, i64 main_version = -1) {
    auto it = table.find(point);
    if (it == table.end()) return -1;
    Point p = it->second; table.erase(it);
    if (gens.refs[p.gen] <= 0) throw std::runtime_error("teacher generation released without a matching acquire");
    gens.refs[p.gen]--;
    double now = mono_now();
    gens.track(now);
    life_s.push_back(now-p.t_capture);
    if (p.t_first >= 0) { first_s.push_back(p.t_first-p.t_capture); run_s.push_back(now-p.t_first); }
    if (main_version >= 0) life_v.push_back((double)(main_version-p.version));
    lin.release_locked(p.snap);
    return p.token;
  }
};

// PagedStore.collect over the native lineage: decisions / blocks / raw frames no live snapshot or head can read are
// freed (limit: the stamp rule of PageCore.free_rows; 0 = all; -1 = the lineage's published stamp + 1, read with the
// scan: rows the driver wrote but has not put in the lineage yet stay). extra_kv / extra_b / extra_raw: rows the Python side
// keeps (its own live snapshots and logs: lanes on the Python path). Returns the numbers freed (kv, blocks, raw).
struct Freed { i64 kv = 0, blocks = 0, raw = 0; };
inline Freed collect(PageCore &pc, Lineage &lin, i64 limit = 0, const uint8_t *extra_kv = nullptr, const uint8_t *extra_b = nullptr,
                     const i64 *extra_raw = nullptr, i64 n_extra_raw = 0) {
  std::vector<uint8_t> keep_kv(pc.cap_d, 0), keep_b(pc.cap_b, 0), used_d(pc.cap_d), used_b(pc.cap_b);
  std::vector<i64> live_raw(extra_raw, extra_raw+n_extra_raw);
  if (extra_kv) for (i64 i = 0; i < pc.cap_d; i++) keep_kv[i] = extra_kv[i];
  if (extra_b) for (i64 i = 0; i < pc.cap_b; i++) keep_b[i] = extra_b[i];
  {
    std::lock_guard<std::mutex> l(lin.mu);
    lin.live_rows_locked(keep_kv.data(), pc.cap_d, keep_b.data(), pc.cap_b, live_raw);
    lin.gc_runs++;
    if (limit < 0) limit = lin.published+1;
  }
  std::vector<i64> dead_d, dead_b, dead_r;
  std::unordered_set<i64> raw_live(live_raw.begin(), live_raw.end());
  {
    std::lock_guard<std::mutex> l(pc.mu);
    memcpy(used_d.data(), pc.d_used.get(), pc.cap_d); memcpy(used_b.data(), pc.b_used.get(), pc.cap_b);
    for (i64 b = 0; b < pc.cap_b; b++)
      if (keep_b[b]) for (i64 q = 0; q < pc.bs; q++) { i64 s = pc.b_src[b*pc.bs+q]; if (s >= 0) raw_live.insert(s); }
    for (auto &e : pc.raw) if (!raw_live.count(e.first)) dead_r.push_back(e.first);
  }
  for (i64 i = 0; i < pc.cap_d; i++) if (used_d[i] && !keep_kv[i]) dead_d.push_back(i);
  for (i64 i = 0; i < pc.cap_b; i++) if (used_b[i] && !keep_b[i]) dead_b.push_back(i);
  std::sort(dead_r.begin(), dead_r.end());
  i64 k0 = pc.freed_kv, b0 = pc.freed_blocks, r0 = pc.freed_raw;
  pc.free_rows(dead_d.data(), (i64)dead_d.size(), dead_b.data(), (i64)dead_b.size(), dead_r.data(), (i64)dead_r.size(), limit);
  Freed f; f.kv = pc.freed_kv-k0; f.blocks = pc.freed_blocks-b0; f.raw = pc.freed_raw-r0;
  return f;
}

// everything above, under one lock (Lineage::mu), shared by the page store's owners (the driver, Python)
struct Native {
  Lineage lin;
  Injury inj;
  Points pts;
  TeachRing tr;
  i64 *status = nullptr;   // block_arrays[0] (the CF points' status, flat index = point)
  bool tr_on = false;
};

}  // namespace abpc
