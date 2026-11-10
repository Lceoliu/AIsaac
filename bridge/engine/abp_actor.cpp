// B52: the block-memory actor's scheduling hot path in C++ (isaac_bridge/actor_native.py builds and loads this file
// together with abp_visible.c against the process's torch; see docs/B50_CPP_ACTOR_DESIGN.md).
//
//   PageStore    the CPU pages of the paged store (K/V, latents, blocks), their allocation, metadata and raw frames.
//                Python's PagedStore keeps its interface and delegates to this when the extension is available.
//   ActorDriver  the graph registry (graphs captured in Python, replayed here through their shared_ptr without the
//                GIL), run_graph (unpack -> H2D -> replay -> D2H -> sync, the middle of ActorStepGraph.__call__),
//                and the driver thread: poll the workers' pipes -> batch -> step -> reply -> pages -> a record for
//                the Python consumer (bookkeeping, rollout, lineage, snapshots, CF capture).
//
// Invariant (B52 condition 3): the driver replies, writes the pages, then enqueues the record, all on its thread;
// every read of a decision's pages (snapshot / restore / learner gather) is reached from a record the consumer took
// from the queue, so no reader can see a page row before it is complete.
//
// B54: the lane rounds (CF / teacher-v2 branches) run on the driver thread too. The consumer restores the lanes'
// arena rows (snapshot restore, lineage ring: set_ring) and queues one job per teacher generation (submit_lanes);
// the driver merges the jobs of the main policy's version into its next main batch (main rows first, as the Python
// path's merged call) and runs every other job as a lane-only call through that generation's lane graph. Order in a
// merged batch: main rows replied -> main pages -> main record -> lane pages -> lane record (kind 4); the consumer
// answers the lanes from the kind 4 records (LaneService.finish). Lane rows take the CRN uniforms of their (request
// sequence, lane) or 0.5 on replayed rows, never the driver's RNG, so the main rows' uniforms are the same with or
// without lanes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Sleep.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>

#include <poll.h>
#include <unistd.h>
#include <errno.h>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "abp_core.h"
#include "abp_lineage.h"
#include "abp_lanes.h"

extern "C" int abp_visible_unpack(const unsigned char *frames, long stride, int n, int wb, int we, int wn, int ws,
                                  float *room, float *map, float *body, unsigned char *body_mask, float *local,
                                  unsigned char *local_mask, float *ent, unsigned char *ent_mask, float *inv,
                                  unsigned char *inv_mask, float *slots, unsigned char *slots_mask);
extern "C" int abp_visible_version(void);

namespace py = pybind11;
using i64 = int64_t;

static inline i64 read_int(const uint8_t *p, int size) {
  switch (size) {
    case 1: return *p;
    case 2: { int16_t v; memcpy(&v, p, 2); return v; }
    case 4: { int32_t v; memcpy(&v, p, 4); return v; }
    default: { i64 v; memcpy(&v, p, 8); return v; }
  }
}

// ----------------------------------------------------------------------------------------------------- PageStore

// (B58c) the store's memory, allocation, metadata, raw frames and GC stamps live in abpc::PageCore (abp_core.h, torch
// free); PageStore gives Python and the driver tensor views of it.
using abpc::RawEntry;
struct PageStore {
  i64 page, kv_elems, z_elems, b_elems, bs, nh;
  int kv_itemsize;
  at::ScalarType kv_dtype;
  i64 cap_d, cap_b;
  std::shared_ptr<abpc::PageCore> core;   // (every view below holds a copy in its deleter: abp_core.h, Lifetimes)
  at::Tensor d_meta, d_used, b_meta, b_src, b_srcver, b_used;   // views of the core's arrays (never reallocated)
  at::Tensor view(void *p, at::IntArrayRef sizes, at::TensorOptions o) {
    std::shared_ptr<abpc::PageCore> keep = core;
    return at::from_blob(p, sizes, [keep](void *) {}, o);
  }

  PageStore(i64 page_, i64 kv_elems_, bool kv_bf16, i64 z_elems_, i64 b_elems_, i64 block_steps, i64 nh_, i64 max_pages_d,
            i64 max_pages_b)
      : page(page_), kv_elems(kv_elems_), z_elems(z_elems_), b_elems(b_elems_), bs(block_steps), nh(nh_) {
    kv_dtype = kv_bf16 ? at::kBFloat16 : at::kFloat;
    kv_itemsize = kv_bf16 ? 2 : 4;
    core.reset(new abpc::PageCore(page, kv_elems*kv_itemsize, z_elems, b_elems, bs, nh, max_pages_d, max_pages_b));
    cap_d = core->cap_d; cap_b = core->cap_b;
    auto L = at::TensorOptions().dtype(at::kLong), B = at::TensorOptions().dtype(at::kBool);
    d_meta = view(core->d_meta.get(), {cap_d, 2}, L); d_used = view(core->d_used.get(), {cap_d}, B);
    b_meta = view(core->b_meta.get(), {cap_b, 2}, L); b_used = view(core->b_used.get(), {cap_b}, B);
    b_src = view(core->b_src.get(), {cap_b, bs}, L); b_srcver = view(core->b_srcver.get(), {cap_b, bs}, L);
  }
  at::Tensor kv_page(i64 p) { return view(core->kv_page(p), {page, kv_elems}, at::TensorOptions().dtype(kv_dtype)); }
  at::Tensor z_page(i64 p) { return view(core->z_page(p), {page, z_elems}, at::TensorOptions().dtype(at::kFloat)); }
  at::Tensor b_page(i64 p) { return view(core->b_page(p), {page, b_elems}, at::TensorOptions().dtype(at::kFloat)); }
  i64 n_pages_d() { std::lock_guard<std::mutex> g(core->mu); return (i64)core->kv_pages.size(); }
  i64 n_pages_b() { std::lock_guard<std::mutex> g(core->mu); return (i64)core->b_pages.size(); }
  i64 n_free_d() { std::lock_guard<std::mutex> g(core->mu); return (i64)core->d_free.size(); }
  i64 n_free_b() { std::lock_guard<std::mutex> g(core->mu); return (i64)core->b_free.size(); }
  void grow_ahead(i64 min_free_d, i64 min_free_b) { core->grow_ahead(min_free_d, min_free_b); }
  void write_decisions_raw(i64 n, const uint8_t *kv, i64 kv_row_bytes, const float *z, i64 z_row_floats, const i64 *steps, i64 version,
                           const uint8_t *first, const int32_t *schema, const uint8_t *const *frames, const i64 *nbytes,
                           const i64 *pending, i64 *dec_out, i64 *kvs_out) {
    core->write_decisions_raw(n, kv, kv_row_bytes, z, z_row_floats, steps, version, first, schema, frames, nbytes, pending, dec_out, kvs_out);
  }
  void write_blocks_raw(i64 m, const float *values, i64 row_floats, const i64 *ends, i64 version, const i64 *src, const i64 *srcver,
                        i64 *bid_out) {
    core->write_blocks_raw(m, values, row_floats, ends, version, src, srcver, bid_out);
  }
  // Python-path writes (tensors): kv [n, ...] of kv_dtype, z [n, ...] float32 (any device -> cpu contiguous)
  std::pair<at::Tensor, at::Tensor> write_decisions(at::Tensor kv, at::Tensor z, at::Tensor steps, i64 version, at::Tensor first,
                                                    at::Tensor schema, at::Tensor frames, at::Tensor nbytes, at::Tensor pending) {
    i64 n = steps.numel();
    kv = kv.to(kv_dtype).contiguous().cpu(); z = z.to(at::kFloat).contiguous().cpu();
    TORCH_CHECK(kv.numel() == n*kv_elems && z.numel() == n*z_elems, "write_decisions: kv / z shapes");
    steps = steps.to(at::kLong).contiguous(); first = first.to(at::kByte).contiguous(); schema = schema.to(at::kInt).contiguous();
    nbytes = nbytes.to(at::kLong).contiguous(); pending = pending.to(at::kLong).contiguous();
    TORCH_CHECK(frames.dim() == 2 && frames.size(0) == n && frames.scalar_type() == at::kByte, "write_decisions: frames [n, F] uint8");
    std::vector<const uint8_t *> fp(n);
    i64 fstride = frames.stride(0);
    const uint8_t *fbase = frames.data_ptr<uint8_t>();
    for (i64 i = 0; i < n; i++) fp[i] = fbase+i*fstride;
    at::Tensor dec = at::empty({n}, at::kLong), kvs = at::empty({n}, at::kLong);
    core->write_decisions_raw(n, (const uint8_t *)kv.data_ptr(), kv_elems*kv_itemsize, z.data_ptr<float>(), z_elems, steps.data_ptr<i64>(),
                              version, first.data_ptr<uint8_t>(), schema.data_ptr<int32_t>(), fp.data(), nbytes.data_ptr<i64>(),
                              pending.data_ptr<i64>(), dec.data_ptr<i64>(), kvs.data_ptr<i64>());
    return {dec, kvs};
  }
  at::Tensor write_blocks(at::Tensor values, at::Tensor ends, i64 version, at::Tensor src, at::Tensor srcver) {
    i64 m = ends.numel();
    values = values.to(at::kFloat).contiguous().cpu();
    TORCH_CHECK(values.numel() == m*b_elems, "write_blocks: values shape");
    ends = ends.to(at::kLong).contiguous(); src = src.to(at::kLong).contiguous(); srcver = srcver.to(at::kLong).contiguous();
    TORCH_CHECK(src.numel() == m*bs && srcver.numel() == m*bs, "write_blocks: src shapes");
    at::Tensor bid = at::empty({m}, at::kLong);
    core->write_blocks_raw(m, values.data_ptr<float>(), b_elems, ends.data_ptr<i64>(), version, src.data_ptr<i64>(), srcver.data_ptr<i64>(),
                           bid.data_ptr<i64>());
    return bid;
  }
  // raw frames
  py::object raw_entry(i64 dec) {
    std::lock_guard<std::mutex> g(core->mu);
    auto it = core->raw.find(dec);
    TORCH_CHECK(it != core->raw.end(), "PageStore: no raw frame for decision ", dec);
    const RawEntry &e = it->second;
    at::Tensor pend = at::empty({nh}, at::kLong);
    memcpy(pend.data_ptr<i64>(), e.pending.data(), nh*sizeof(i64));
    return py::make_tuple((int)e.first, (int)e.schema, py::bytes(e.frame), pend);
  }
  bool raw_has(i64 dec) { std::lock_guard<std::mutex> g(core->mu); return core->raw.count(dec) > 0; }
  i64 raw_count() { std::lock_guard<std::mutex> g(core->mu); return (i64)core->raw.size(); }
  at::Tensor raw_keys() {
    std::lock_guard<std::mutex> g(core->mu);
    at::Tensor k = at::empty({(i64)core->raw.size()}, at::kLong);
    i64 *p = k.data_ptr<i64>(); i64 i = 0;
    for (auto &kv : core->raw) p[i++] = kv.first;
    return k;
  }
  double raw_mb() {
    std::lock_guard<std::mutex> g(core->mu);
    double b = 0; for (auto &kv : core->raw) b += kv.second.frame.size();
    return b/1048576.0;
  }
  // (B45 source_arena) first [n] u8, schema [n] i32, sizes [n] i64, offsets [n] i64, joined frames u8, pending [n, nh]
  py::tuple source_arena(at::Tensor decisions) {
    decisions = decisions.to(at::kLong).contiguous();
    i64 n = decisions.numel();
    const i64 *d = decisions.data_ptr<i64>();
    std::lock_guard<std::mutex> g(core->mu);
    at::Tensor first = at::empty({n}, at::kByte), schema = at::empty({n}, at::kInt), sizes = at::empty({n}, at::kLong),
               off = at::empty({n}, at::kLong), pend = at::empty({n, nh}, at::kLong);
    i64 total = 0;
    std::vector<const RawEntry *> es(n);
    for (i64 i = 0; i < n; i++) {
      auto it = core->raw.find(d[i]);
      TORCH_CHECK(it != core->raw.end(), "PageStore.source_arena: no raw frame for decision ", d[i]);
      es[i] = &it->second;
      first.data_ptr<uint8_t>()[i] = es[i]->first; schema.data_ptr<int32_t>()[i] = es[i]->schema;
      sizes.data_ptr<i64>()[i] = (i64)es[i]->frame.size(); off.data_ptr<i64>()[i] = total; total += (i64)es[i]->frame.size();
      memcpy(pend.data_ptr<i64>()+i*nh, es[i]->pending.data(), nh*sizeof(i64));
    }
    at::Tensor buf = at::empty({total}, at::kByte);
    uint8_t *b = buf.data_ptr<uint8_t>();
    for (i64 i = 0; i < n; i++) memcpy(b+off.data_ptr<i64>()[i], es[i]->frame.data(), es[i]->frame.size());
    return py::make_tuple(first, schema, sizes, off, buf, pend);
  }
  void free_rows(at::Tensor dead_d, at::Tensor dead_b, at::Tensor dead_raw, i64 limit) {
    dead_d = dead_d.to(at::kLong).contiguous(); dead_b = dead_b.to(at::kLong).contiguous(); dead_raw = dead_raw.to(at::kLong).contiguous();
    core->free_rows(dead_d.data_ptr<i64>(), dead_d.numel(), dead_b.data_ptr<i64>(), dead_b.numel(), dead_raw.data_ptr<i64>(),
                    dead_raw.numel(), limit);
  }
  void set_stamp(i64 v) { core->set_stamp(v); }
  py::dict stats() {
    std::lock_guard<std::mutex> g(core->mu);
    py::dict d;
    d["kv_pages"] = (i64)core->kv_pages.size(); d["b_pages"] = (i64)core->b_pages.size(); d["d_free"] = (i64)core->d_free.size();
    d["b_free"] = (i64)core->b_free.size(); d["raw"] = (i64)core->raw.size(); d["next_decision"] = core->next_decision;
    d["reallocs"] = core->reallocs; d["cap_pages_d"] = (i64)core->kv_pages.capacity(); d["cap_pages_b"] = (i64)core->b_pages.capacity();
    d["freed_kv"] = core->freed_kv; d["freed_blocks"] = core->freed_blocks; d["freed_raw"] = core->freed_raw;
    d["stamp"] = core->stamp; d["stamp_kept"] = core->stamp_kept;
    return d;
  }

  // ---- (B58c) the native lineage (abp_lineage.h): logs, snapshots, the rollout sidecars, contexts, injury sources,
  // CF points; shared by the driver and Python (Lineage::mu). Python-held snapshots are SnapPin objects.
  std::shared_ptr<abpc::Native> nat = std::make_shared<abpc::Native>();
  bool lineage_on = false;
  void lineage_configure(i64 window, i64 block_steps, i64 memory_blocks, i64 nh_, i64 rows, i64 workers, std::vector<i64> depths,
                         i64 chunk_steps, i64 generations) {
    abpc::LineageCfg c; c.window = window; c.block_steps = block_steps; c.memory_blocks = memory_blocks; c.nh = nh_;
    nat->lin.configure(c, rows);
    std::lock_guard<std::mutex> l(nat->lin.mu);
    nat->inj.configure(workers, depths);
    nat->lin.side[0].chunk_steps = nat->lin.side[1].chunk_steps = chunk_steps;
    nat->pts.gens.configure(generations);
    lineage_on = true;
  }
  void lineage_status(i64 status_ptr) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->status = (i64 *)status_ptr; }
  void lineage_teach(py::dict d) {
    auto I = [&](const char *k) { return d[k].cast<i64>(); };
    std::lock_guard<std::mutex> l(nat->lin.mu);
    abpc::TeachRing &t = nat->tr;
    t.stats = (const double *)I("stats"); t.stats_stride = I("stats_stride"); t.st_teach = I("st_teach");
    t.commit = (const i64 *)I("commit"); t.ring = I("ring");
    t.rows = (const uint8_t *)I("rows"); t.row_size = I("row_size"); t.o_episode = I("o_episode"); t.o_t = I("o_t");
    t.s_episode = (int)I("s_episode"); t.s_t = (int)I("s_t");
    t.jobs = (const i64 *)I("jobs"); t.jobs_k = I("jobs_k"); t.epoch = (const i64 *)I("epoch");
    nat->tr_on = true;
  }
  // logs / snapshots from Python (tests, the Python path's rows)
  void lin_new_head(i64 row, i64 k0, i64 k1, i64 k2) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    nat->lin.set_head_locked(row, nat->lin.new_log_locked(k0, k1, k2, 0));
  }
  i64 lin_head_end(i64 row) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    i64 g = nat->lin.head.at(row); return g < 0 ? -1 : nat->lin.logs[g].end();
  }
  i64 lin_snap_row(i64 row, at::Tensor pend, i64 version) {
    pend = pend.to(at::kLong).contiguous().cpu();
    std::lock_guard<std::mutex> l(nat->lin.mu);
    return nat->lin.snap_row_locked(row, pend.data_ptr<i64>(), version);
  }
  void lin_push(i64 row, i64 dec, i64 kvs) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.push_locked(row, dec, kvs); }
  void lin_push_block(i64 row, i64 bid, i64 end) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.push_block_locked(row, bid, end); }
  void lin_retain(i64 s) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.retain_locked(s); }
  void lin_release(i64 s) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.release_locked(s); }
  // a restored row's head: the snapshot's branch (EpisodeLog.branch) -> the row's new log
  void lin_restore(i64 row, i64 snap) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    nat->lin.set_head_locked(row, nat->lin.branch_locked(snap));
  }
  // the snapshots as Python reads them: per snapshot its branch window (EpisodeLog.branch's arrays); meta [n, 8 + nh]
  // (step, version, offset, decisions, blocks, episode key x3, pending), the concatenated dec / kvs / blk / blk_end
  py::tuple lin_materialize(std::vector<i64> snaps) {
    i64 n = (i64)snaps.size(), nh_ = nat->lin.cfg.nh;
    std::vector<i64> dec, kvs, blk, bend;
    at::Tensor meta = at::empty({n, 8+nh_}, at::kLong);
    i64 *m = meta.data_ptr<i64>();
    {
      std::lock_guard<std::mutex> l(nat->lin.mu);
      abpc::Lineage &L = nat->lin;
      for (i64 i = 0; i < n; i++) {
        const abpc::SnapData &d = L.snaps.at(snaps[i]);
        if (!d.used) throw std::runtime_error("Lineage: materialize of a freed snapshot");
        const abpc::LogData &g = L.logs.at(d.log);
        i64 lo = std::max(g.offset, d.step-std::max(L.cfg.window, L.cfg.block_steps));
        L.check_range(g, lo, d.step);
        i64 blo, bat; abpc::Lineage::blocks_before(g, d.step, L.cfg.memory_blocks, blo, bat);
        i64 *r = m+i*(8+nh_);
        r[0] = d.step; r[1] = d.version; r[2] = lo; r[3] = d.step-lo; r[4] = bat-blo; r[5] = g.key[0]; r[6] = g.key[1]; r[7] = g.key[2];
        for (i64 h = 0; h < nh_; h++) r[8+h] = d.pend[h];
        dec.insert(dec.end(), g.dec.begin()+(lo-g.offset), g.dec.begin()+(d.step-g.offset));
        kvs.insert(kvs.end(), g.kvs.begin()+(lo-g.offset), g.kvs.begin()+(d.step-g.offset));
        blk.insert(blk.end(), g.blk.begin()+blo, g.blk.begin()+bat);
        bend.insert(bend.end(), g.blk_end.begin()+blo, g.blk_end.begin()+bat);
      }
    }
    auto T = [](const std::vector<i64> &v) { at::Tensor t = at::empty({(i64)v.size()}, at::kLong); if (!v.empty()) memcpy(t.data_ptr<i64>(), v.data(), v.size()*8); return t; };
    return py::make_tuple(meta, T(dec), T(kvs), T(blk), T(bend));
  }
  // the rollout sidecars (BlockRollout)
  void side_record(int sl, i64 w, i64 index, bool first, i64 snap) {
    std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.side_record_locked(sl, w, index, first, snap);
  }
  void side_reset(int sl) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.side_reset_locked(sl); }
  void side_carry(int dst, int src, at::Tensor workers, at::Tensor indices) {
    workers = workers.to(at::kLong).contiguous(); indices = indices.to(at::kLong).contiguous();
    std::lock_guard<std::mutex> l(nat->lin.mu);
    nat->lin.side_carry_locked(dst, src, workers.data_ptr<i64>(), indices.data_ptr<i64>(), workers.numel());
  }
  // (keys [m, 2] (worker, index) in BlockRollout.payload's order, snapshots [m]); open: (workers, indices, snapshots)
  py::tuple side_payload(int sl) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    abpc::Sidecar &sc = nat->lin.side[sl];
    std::vector<i64> keys, snaps;
    for (i64 w : sc.worker_order)
      for (i64 idx : sc.order[w]) { keys.push_back(w); keys.push_back(idx); snaps.push_back(sc.starts[w][idx]); }
    at::Tensor k = at::empty({(i64)snaps.size(), 2}, at::kLong), s = at::empty({(i64)snaps.size()}, at::kLong);
    if (!snaps.empty()) { memcpy(k.data_ptr<i64>(), keys.data(), keys.size()*8); memcpy(s.data_ptr<i64>(), snaps.data(), snaps.size()*8); }
    std::vector<i64> ow, oi, os;
    for (auto &e : sc.open) { ow.push_back(e.first); oi.push_back(e.second.first); os.push_back(e.second.second); }
    auto T = [](const std::vector<i64> &v) { at::Tensor t = at::empty({(i64)v.size()}, at::kLong); if (!v.empty()) memcpy(t.data_ptr<i64>(), v.data(), v.size()*8); return t; };
    return py::make_tuple(k, s, T(ow), T(oi), T(os));
  }
  // contexts (ContextRegistry)
  i64 ctx_put(i64 owner, i64 snap) { std::lock_guard<std::mutex> l(nat->lin.mu); return nat->lin.ctx_put_locked(owner, snap); }
  i64 ctx_get(i64 ident) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    auto it = nat->lin.ctx.find(ident);
    if (it == nat->lin.ctx.end()) throw py::key_error("context " + std::to_string(ident));
    return it->second.second;
  }
  void ctx_complete(i64 owner) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.ctx_complete_locked(owner); }
  i64 ctx_retire(std::vector<i64> live, bool transient) {
    std::unordered_set<i64> s(live.begin(), live.end());
    std::lock_guard<std::mutex> l(nat->lin.mu);
    return nat->lin.ctx_retire_locked(s, transient);
  }
  // injury sources
  void inj_observe(i64 w, bool first, i64 e, i64 t, bool hurt) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->inj.observe(w, first, e, t, hurt); }
  void inj_record(i64 w, i64 e, i64 t, i64 snap) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->inj.record(nat->lin, w, e, t, snap); }
  i64 inj_get(i64 w, i64 e, i64 t) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    auto it = nat->inj.sources.find(abpc::Key3{w, e, t});
    if (it == nat->inj.sources.end()) throw py::key_error("injury source (" + std::to_string(w) + ", " + std::to_string(e) + ", " + std::to_string(t) + ")");
    return it->second;
  }
  i64 inj_reclaim() {   // -1: a worker was inside its pin update (nothing reclaimed)
    std::lock_guard<std::mutex> l(nat->lin.mu);
    if (!nat->tr_on) return 0;
    i64 freed = 0;
    return nat->inj.reclaim(nat->lin, nat->tr, freed) ? freed : -1;
  }
  py::tuple inj_keys() {   // (tests) the sources' keys [n, 3], each worker's hurt candidates' keys [m, 3]
    std::lock_guard<std::mutex> l(nat->lin.mu);
    std::vector<i64> a, b;
    for (auto &e : nat->inj.sources) { a.push_back(e.first.w); a.push_back(e.first.e); a.push_back(e.first.t); }
    for (auto &h : nat->inj.hurt) for (auto &k : h) { b.push_back(k.w); b.push_back(k.e); b.push_back(k.t); }
    at::Tensor ta = at::empty({(i64)a.size()/3, 3}, at::kLong), tb = at::empty({(i64)b.size()/3, 3}, at::kLong);
    if (!a.empty()) memcpy(ta.data_ptr<i64>(), a.data(), a.size()*8);
    if (!b.empty()) memcpy(tb.data_ptr<i64>(), b.data(), b.size()*8);
    return py::make_tuple(ta, tb);
  }
  // CF points
  i64 pt_capture(i64 point, i64 snap, i64 version) {   // 2 made, -1 refused, -2 - i: publish generation slot i first
    std::lock_guard<std::mutex> l(nat->lin.mu);
    i64 r = nat->pts.capture(nat->lin, point, snap, version);
    if (nat->status && (r == 2 || r == -1)) nat->status[point] = r;
    return r;
  }
  void pt_published(int slot, i64 version) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->pts.published(slot, version); }
  i64 pt_release(i64 point) { std::lock_guard<std::mutex> l(nat->lin.mu); return nat->pts.release(nat->lin, point); }
  py::object pt_get(i64 point) {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    auto it = nat->pts.table.find(point);
    if (it == nat->pts.table.end()) return py::none();
    const abpc::Point &p = it->second;
    return py::make_tuple(p.snap, p.gen, p.version, p.token);
  }
  std::vector<i64> pt_points() {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    std::vector<i64> v; for (auto &e : nat->pts.table) v.push_back(e.first);
    std::sort(v.begin(), v.end()); return v;
  }
  py::tuple pt_gens() {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    auto &g = nat->pts.gens;
    return py::make_tuple(g.version, g.refs);
  }
  // the GC over the native lineage (abpc::collect); extra_*: rows the Python side keeps (its own snapshots / logs)
  py::tuple lin_collect(i64 limit, py::object extra_kv, py::object extra_b, py::object extra_raw) {
    at::Tensor ek, eb, er;
    const uint8_t *pk = nullptr, *pb = nullptr; const i64 *pr = nullptr; i64 nr = 0;
    if (!extra_kv.is_none()) { ek = extra_kv.cast<at::Tensor>().to(at::kByte).contiguous(); TORCH_CHECK(ek.numel() == cap_d, "extra_kv: cap_d"); pk = ek.data_ptr<uint8_t>(); }
    if (!extra_b.is_none()) { eb = extra_b.cast<at::Tensor>().to(at::kByte).contiguous(); TORCH_CHECK(eb.numel() == cap_b, "extra_b: cap_b"); pb = eb.data_ptr<uint8_t>(); }
    if (!extra_raw.is_none()) { er = extra_raw.cast<at::Tensor>().to(at::kLong).contiguous(); pr = er.data_ptr<i64>(); nr = er.numel(); }
    abpc::Freed f;
    {
      py::gil_scoped_release nogil;
      f = abpc::collect(*core, nat->lin, limit, pk, pb, pr, nr);
    }
    return py::make_tuple(f.kv, f.blocks, f.raw);
  }
  py::dict lin_stats() {
    std::lock_guard<std::mutex> l(nat->lin.mu);
    py::dict d;
    abpc::Lineage &L = nat->lin;
    i64 heads = 0; for (i64 h : L.head) heads += h >= 0;
    d["logs"] = L.n_logs; d["snaps"] = L.n_snaps; d["heads"] = heads; d["made_logs"] = L.made_logs; d["made_snaps"] = L.made_snaps;
    d["contexts"] = (i64)L.ctx.size(); d["completed"] = (i64)L.completed.size(); d["next_ident"] = L.next_ident;
    d["injury"] = (i64)nat->inj.sources.size(); d["points"] = (i64)nat->pts.table.size(); d["gc_runs"] = L.gc_runs;
    i64 side = 0; for (auto &sc : L.side) for (auto &w : sc.starts) side += (i64)w.second.size();
    d["side_starts"] = side;
    return d;
  }
};

// (B58c) a Python-held snapshot: one reference, released when the Python object goes
struct SnapPin {
  std::shared_ptr<abpc::Native> nat; i64 snap;
  SnapPin(std::shared_ptr<PageStore> s, i64 id, bool retain) : nat(s->nat), snap(id) {
    if (retain) { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.retain_locked(snap); }
  }
  ~SnapPin() {
    try { std::lock_guard<std::mutex> l(nat->lin.mu); nat->lin.release_locked(snap); } catch (...) {}
  }
};

// ------------------------------------------------------------------------------------------------- graph registry

struct Layout {   // byte offsets into the slab (ActorStepGraph._layout)
  i64 room, map, body, body_mask, local, local_mask, entity, entity_mask, inventory, inventory_mask, slots, slots_mask;
  i64 pending, ids, u, replay, complete, version;
};

struct GraphEntry {
  i64 cap, wb, we, wn, ws;
  at::Tensor host_flat, dev_flat;                 // pinned slab, device slab
  std::shared_ptr<at::cuda::CUDAGraph> graph;
  at::Tensor out, cur, z, written;                // device outputs
  at::Tensor bo, bc, bz, bw;                      // pinned D2H buffers
  Layout L;
  i64 scratch = -1;                               // (B58d) the arena row its padded rows address (-1: the driver's)
  i64 filled = 0;                                 // rows of the slab holding a call's rows (zeroed past the next call's n)
  i64 heads, nh;
};

struct Record {
  int kind;                                  // 0 step, 1 capture needed, 2 python path needed (invalid frame / too wide),
                                             // 3 error, 4 lane rows stepped (B54), 5 a lane job Python must step (B54)
  i64 job = -1, owner = 0;                   // (kinds 1, 4, 5) the lane job, the graph owner (1: main or lane)
  uint8_t merged = 0;                        // (kind 4) the rows went with a main batch
  std::vector<i64> workers, slots;
  std::vector<uint8_t> go;
  at::Tensor rows;                           // uint8 [n_all * stride] (header + used frame bytes copied)
  at::Tensor priv;                           // uint8 [n_all * priv_itemsize] or empty
  std::vector<i64> cf_snap;                  // [n_all] the worker's request (read and zeroed by the driver)
  // go rows only, in batch order
  at::Tensor out, actions, pending_before, step_before, first, complete, dec, kvs, bid;
  i64 version = 0;
  i64 cap = 0, wb = 0, we = 0, wn = 0, ws = 0;   // (kind 1)
  double t_poll = 0, t_sync = 0, t_reply = 0, t_pages = 0, t_done = 0;   // driver clock (s) of the step's phases
  double t_main_reply = 0;                   // (kind 4 of a merged batch) when the batch's main rows were replied
  at::Tensor kk, snap;                       // (B58a+b, native consumer) go rows' rollout index (-1: dropped, full);
                                             // per row [n_all, N_SNAP] accumulators before an episode end reset them
  // (B58c, kind 6) a CF capture whose generation Python must publish first: point, generation slot, the row's snapshot
  // (one reference the record holds), worker
  i64 point = -1, pub_slot = -1, pub_snap = -1, pub_worker = -1;
  int lane_src = 0;   // (B58d) 1: a record of the lane thread (kind 1: a lane graph to capture; 5: a job Python steps; 3)
  at::Tensor lane_ids, lane_seq, lane_replay;   // (B58d, kind 5) the job's lanes, CRN keys, replay flags
};

// tok_policy.crn_uniforms, bit for bit: element (key, t, c) = splitmix64 output t * 256 + c + 1 of the stream seeded with
// mix(key * golden + 0x632BE59BD9B4E019); its top 23 bits k give (k + 0.5) / 2^23 as float32
static inline uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27))*0x94D049BB133111EBULL; return z ^ (z >> 31);
}
static void crn_row(i64 key, i64 t, i64 columns, float *out) {
  const uint64_t G = 0x9E3779B97F4A7C15ULL;
  uint64_t seed = mix64((uint64_t)key*G+0x632BE59BD9B4E019ULL);
  for (i64 c = 0; c < columns; c++) {
    uint64_t ctr = ((uint64_t)t << 8)+(uint64_t)c+1;
    uint64_t z = mix64(seed+ctr*G);
    out[c] = (float)(((double)(z >> 41)+0.5)/8388608.0);
  }
}
static at::Tensor crn_uniforms(at::Tensor keys, at::Tensor t, i64 columns) {
  keys = keys.to(at::kLong).contiguous(); t = t.to(at::kLong).contiguous();
  i64 n = keys.numel();
  TORCH_CHECK(t.numel() == n, "crn_uniforms: keys and t of one length");
  at::Tensor out = at::empty({n, columns}, at::kFloat);
  for (i64 i = 0; i < n; i++) crn_row(keys.data_ptr<i64>()[i], t.data_ptr<i64>()[i], columns, out.data_ptr<float>()+i*columns);
  return out;
}

struct LaneJob {   // (B54) one teacher generation's rows of a lane round, queued by the consumer
  i64 job = 0, owner = 0, version = 0, n = 0;
  std::vector<uint8_t> rows;                 // n x row_stride (ROW records: header + frame)
  std::vector<i64> pend, ids, seq, lane;     // pending [n x nh], arena rows, CRN keys (request sequence, lane)
  std::vector<uint8_t> replay;               // evaluate rows (uniform 0.5)
};

// (B58a+b / B58c) the native consumer's configuration, rollout slots, frame arenas and arithmetic: abpc (abp_core.h)
using abpc::ConsumerCfg;
using abpc::RollSlot;
using abpc::N_SNAP;

struct Xoshiro {   // xoshiro256**, seeded through splitmix64
  uint64_t s[4];
  explicit Xoshiro(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
      seed += 0x9E3779B97F4A7C15ULL; uint64_t z = seed;
      z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27))*0x94D049BB133111EBULL; s[i] = z ^ (z >> 31);
    }
  }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64-k)); }
  uint64_t next() {
    uint64_t r = rotl(s[1]*5, 7)*9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  float uniform() { return (float)(next() >> 40)*(1.0f/16777216.0f); }   // [0, 1) in 2^-24 steps (torch.rand's resolution)
};

struct ArenaReset {   // BlockSlotState.reset: kv.index_fill_(2, ids, 0) and the rows tensors' index_fill_(0, ids, fill)
  std::vector<at::Tensor> tensors; std::vector<i64> dims; std::vector<double> fills;
};

struct ActorDriver {
  std::shared_ptr<PageStore> store;
  std::map<std::vector<i64>, GraphEntry> graphs;   // key (owner, cap, wb, we, wn, ws)
  std::mutex gmu;
  // stream
  i64 stream_id = 0, device_index = 0; bool have_stream = false;
  // sampler shared memory
  uint8_t *rows_base = nullptr; i64 row_stride = 0; i64 n_workers = 0;
  std::vector<int> fds;
  int32_t *actions = nullptr; i64 actions_stride = 0;   // int32 [W, ACTION_F]
  double *main_value = nullptr, *main_vg = nullptr, *aux = nullptr; i64 aux_stride = 0;
  i64 *cf_snap = nullptr;
  uint8_t *priv = nullptr; i64 priv_itemsize = 0;
  std::string reply;
  // ROW fields: offset, size
  i64 f_first = 0, f_done = 0, f_pforce = 0, f_schema = 0, f_bytes = 0, f_frame = 0; int s_first = 1, s_done = 1, s_pforce = 1;
  i64 done_branch = 4;
  bool pf_on = false;
  // arena
  ArenaReset reset; i64 scratch_row = 0, base = 0; i64 *step_host = nullptr; i64 heads = 0, nh = 0; i64 block_steps = 32;
  i64 owner_main = 0;
  std::vector<i64> ring_dec, ring_kv, ring_ver; std::vector<i64> ring_n;   // per arena row: the last block_steps decisions
  i64 arena_rows = 0;
  std::vector<i64> pending_cur;   // [W, nh] the action under way per worker
  // frame buckets
  std::vector<i64> bk_body, bk_entity, bk_inventory, bk_slots;
  i64 frame_fixed = 1969, frame_bytes = 131072, schema_version = 0; double magic = 61107;
  // rng
  Xoshiro rng{1};
  double fixed_u = -1;   // (tests) >= 0: every uniform of the driver's steps is this value
  // driver thread
  std::thread th; std::atomic<bool> running{false}, stop_req{false}, pause_req{false}, paused{false};
  std::mutex cmu; std::condition_variable ccv;
  std::deque<std::shared_ptr<Record>> queue; std::mutex qmu; std::condition_variable qcv; i64 queue_max = 256;
  std::atomic<i64> version{0};
  std::vector<i64> ready_w, ready_s;   // a batch the driver could not step (capture needed): retried after resume
  std::string error;
  // (B58a+b) the native consumer
  ConsumerCfg cc; RollSlot slots[2]; std::atomic<int> cur_slot{-1}; std::mutex amu;   // amu: the arenas
  // (B54) lane jobs
  std::deque<std::shared_ptr<LaneJob>> lane_q; std::mutex lmu;
  bool lane_merge = true;              // (tests / ISAAC_BLOCK_LANE_MERGE=0) merge jobs of the main version into main batches
  std::atomic<bool> main_hold{false};  // (B54) lanes only: the workers' records wait (the actor waits for the learner)
  // (B54) per worker: its last record asked for a CF capture Python has not made yet; its next record waits for it, so
  // the worker never runs ahead of the capture's status (as with the Python path, which captures before replying)
  std::unique_ptr<std::atomic<uint8_t>[]> cap_wait;
  // counters
  std::atomic<i64> n_steps{0}, n_calls{0}, n_rows{0}, n_terminal{0};
  std::atomic<double> t_sync_s{0}, t_pages_s{0}, t_host_s{0};
  std::atomic<i64> n_lane_jobs{0}, n_lane_rows{0}, n_merged_rows{0}, n_merged_calls{0}, n_lane_calls{0}, n_captures{0},
      n_lane_python{0}, n_pauses{0}, n_cap_holds{0};
  std::atomic<double> t_pause_s{0};
  std::atomic<double> t_lane_s{0};                 // (B54) seconds in lane-only calls (fill, replay, pages, record)
  std::atomic<i64> lane_hist[6] = {};              // (B54) lane-only calls by rows: 1, 2, 3-4, 5-8, 9-16, > 16
  // (B58 step 0) the driver thread's time by phase (s; one writer, the driver): poll = waiting for / reading the
  // workers' messages; prep = a main step's row copies, packing, widths and inputs (t0 .. the graph); sync = the
  // graph's fill + replay + sync (t_sync_s); reply = outputs, shared arrays, the replies; pages = the main rows' pages
  // (t_pages_s, without the consumer since step 0); cons = the native consumer; push = waiting for room in the record
  // queue (every record); mlane = the merged lane rows' records and pages; lane = lane-only calls (t_lane_s);
  // hold = lanes-only mode (main_hold) idle; loop = the loop's wall time (the rest is pause_s and bookkeeping)
  std::atomic<double> t_poll_s{0}, t_prep_s{0}, t_reply_s{0}, t_cons_s{0}, t_push_s{0}, t_mlane_s{0}, t_hold_s{0}, t_loop_s{0};
  std::atomic<i64> n_polls{0}, n_idle_polls{0};
  // (B58d plan: why lane rows do not ride the main steps) lane-only rows at the main version / an older one, merged
  // rows put back (the batch too wide), lane jobs queued while the driver was inside a main step / outside it
  std::atomic<i64> n_lo_cur_rows{0}, n_lo_old_rows{0}, n_putback_rows{0}, n_enq_in_step{0}, n_enq_outside{0};
  std::atomic<bool> in_step{false};
  // (B58c) the main rows' lineage in the driver (store->nat): observe_rows, logs, snapshots, CF captures, injury
  // sources, log pushes, the rollout sidecar; Python enters only for a generation to publish (kind 6)
  bool lin_on = false; i64 lin_tag = 0;
  i64 f_episode = 0, f_t = 0, f_hurt = 0; int s_episode = 8, s_t = 8;
  std::atomic<i64> n_cap_native{0}, n_cap_refused{0}, n_cap_publish{0};
  std::atomic<double> t_lin_s{0};

  explicit ActorDriver(std::shared_ptr<PageStore> s) : store(std::move(s)) {}
  ~ActorDriver() { lanes_stop(); stop(); }

  static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

  // ---- configuration (Python, before start)
  void set_stream(i64 sid, i64 dev) { stream_id = sid; device_index = dev; have_stream = true; }
  void set_seed(i64 seed) { rng = Xoshiro((uint64_t)seed); }
  void set_fixed_uniform(double v) { fixed_u = v; }
  void set_version(i64 v) { version = v; }
  void set_buckets(std::vector<i64> body, std::vector<i64> entity, std::vector<i64> inventory, std::vector<i64> slots, i64 fixed,
                   i64 fbytes, i64 schema) {
    bk_body = body; bk_entity = entity; bk_inventory = inventory; bk_slots = slots; frame_fixed = fixed; frame_bytes = fbytes; schema_version = schema;
  }
  void set_sampler(i64 rows_ptr, i64 stride, i64 workers, std::vector<int> fds_, i64 actions_ptr, i64 actions_stride_, i64 main_value_ptr,
                   i64 main_vg_ptr, i64 aux_ptr, i64 aux_stride_, i64 cf_snap_ptr, i64 priv_ptr, i64 priv_itemsize_, py::dict fields,
                   py::bytes reply_, i64 done_branch_, bool pf_on_) {
    rows_base = (uint8_t *)rows_ptr; row_stride = stride; n_workers = workers; fds = fds_;
    actions = (int32_t *)actions_ptr; actions_stride = actions_stride_;
    main_value = (double *)main_value_ptr; main_vg = (double *)main_vg_ptr; aux = (double *)aux_ptr; aux_stride = aux_stride_;
    cf_snap = (i64 *)cf_snap_ptr; priv = (uint8_t *)priv_ptr; priv_itemsize = priv_itemsize_;
    auto f = [&](const char *name, int &size) { py::tuple t = fields[name].cast<py::tuple>(); size = t[1].cast<int>(); return t[0].cast<i64>(); };
    int dummy;
    f_first = f("first", s_first); f_done = f("done", s_done); f_pforce = f("pforce", s_pforce);
    f_schema = f("visible_schema", dummy); f_bytes = f("visible_bytes", dummy); f_frame = f("visible_frame", dummy);
    reply = reply_; done_branch = done_branch_; pf_on = pf_on_;
    pending_cur.assign(workers*nh, 0);
    cap_wait.reset(new std::atomic<uint8_t>[workers]);
    for (i64 w = 0; w < workers; w++) cap_wait[w] = 0;
  }
  // (B54) Python made (or refused) the CF captures the workers' last records asked for: their next records may go
  void capture_done(std::vector<i64> workers) { for (i64 w : workers) if (w >= 0 && w < n_workers) cap_wait[w] = 0; }
  void set_main_hold(bool on) { main_hold = on; }
  // (B58c) the main rows' lineage in the driver (the store's Native, configured by PageStore.lineage_configure): tag =
  // the logs' episode key's first part (Python: os.getpid()); fields: ROW 'episode' / 't' (int) offsets and sizes,
  // 'hurt' (float32) offset
  void set_lineage(bool on, i64 tag, py::dict fields) {
    TORCH_CHECK(!on || store->lineage_on, "set_lineage: configure the store's lineage first");
    auto f = [&](const char *name, int &size) { py::tuple t = fields[name].cast<py::tuple>(); size = t[1].cast<int>(); return t[0].cast<i64>(); };
    int dummy;
    if (on) { f_episode = f("episode", s_episode); f_t = f("t", s_t); f_hurt = f("hurt", dummy); }
    lin_tag = tag; lin_on = on;
  }
  void set_arena(std::vector<at::Tensor> tensors, std::vector<i64> dims, std::vector<double> fills, i64 scratch, i64 base_, i64 step_host_ptr,
                 i64 heads_, i64 nh_, i64 block_steps_, i64 arena_rows_, i64 owner) {
    reset.tensors = tensors; reset.dims = dims; reset.fills = fills; scratch_row = scratch; base = base_; step_host = (i64 *)step_host_ptr;
    heads = heads_; nh = nh_; block_steps = block_steps_; arena_rows = arena_rows_; owner_main = owner;
    ring_dec.assign(arena_rows*block_steps, -1); ring_kv.assign(arena_rows*block_steps, -1); ring_ver.assign(arena_rows*block_steps, -1);
    ring_n.assign(arena_rows, 0);
    if (n_workers) pending_cur.assign(n_workers*nh, 0);
  }
  void set_pending(i64 worker, at::Tensor p) {   // (the consumer mirrors forced / branch resets if it ever needs to)
    p = p.to(at::kLong).contiguous();
    memcpy(&pending_cur[worker*nh], p.data_ptr<i64>(), nh*sizeof(i64));
  }
  at::Tensor get_pending() {
    at::Tensor t = at::empty({n_workers, nh}, at::kLong);
    memcpy(t.data_ptr<i64>(), pending_cur.data(), pending_cur.size()*sizeof(i64));
    return t;
  }
  // ---- (B58a+b) native consumer configuration (Python, with the driver paused or not started)
  void set_consumer(py::dict d) {
    auto I = [&](const char *k) { return d[k].cast<i64>(); };
    auto F = [&](const char *k) { return d[k].cast<double>(); };
    auto B = [&](const char *k) { return d[k].cast<bool>(); };
    auto P = [&](const char *k) { return d.contains(k) && !d[k].is_none() ? d[k].cast<i64>() : (i64)0; };
    ConsumerCfg c;
    c.o_first = I("o_first"); c.o_done = I("o_done"); c.o_damage = I("o_damage"); c.o_hurt = I("o_hurt"); c.o_events = I("o_events");
    c.o_stage = I("o_stage"); c.o_player = I("o_player"); c.o_pforce = I("o_pforce"); c.row_frame_at = I("row_frame_at");
    c.w_damage = F("w_damage"); c.w_hurt = F("w_hurt"); c.w_time = F("w_time"); c.w_death = F("w_death"); c.w_timeout = F("w_timeout");
    c.w_room = F("w_room"); c.w_explore = F("w_explore"); c.w_boss = F("w_boss"); c.w_exit = F("w_exit"); c.w_heal = F("w_heal");
    c.w_resource = F("w_resource"); c.end_bonus = F("end_bonus"); c.gamma = F("gamma");
    c.floor = B("floor"); c.run = B("run"); c.items = B("items"); c.gains_on = B("gains_on"); c.plr_on = B("plr_on"); c.pick_on = B("pick_on");
    c.ev_exit = I("ev_exit"); c.ev_item = I("ev_item"); c.ev_active = I("ev_active"); c.ev_pill = I("ev_pill"); c.done_branch = I("done_branch");
    c.ret = (double *)P("ret"); c.ep_hurt = (double *)P("hurt"); c.ep_damage = (double *)P("damage"); c.gains = (double *)P("gains");
    c.healed = (double *)P("healed"); c.gathered = (double *)P("gathered"); c.v0 = (double *)P("v0"); c.gret = (double *)P("gret");
    c.rooms = (i64 *)P("rooms"); c.boss = (i64 *)P("boss"); c.extra = (i64 *)P("extra"); c.steps = (i64 *)P("steps");
    c.pick_steps = (i64 *)P("pick_steps");
    c.on = true;
    cc = c;
  }
  void set_rollout(int slot, py::dict d, i64 arena_bytes) {
    TORCH_CHECK(slot == 0 || slot == 1, "set_rollout: slot 0 or 1");
    auto P = [&](const char *k) { return d.contains(k) && !d[k].is_none() ? d[k].cast<i64>() : (i64)0; };
    RollSlot &r = slots[slot];
    r.cap = P("cap"); r.rowc_size = P("rowc_size"); r.off_visible_off = P("off_visible_off"); r.priv_size = P("priv_size");
    r.rows = (uint8_t *)P("rows"); r.terminal = (uint8_t *)P("terminal"); r.close_done = (uint8_t *)P("close_done");
    r.close_ev = (uint8_t *)P("close_ev"); r.priv = (uint8_t *)P("priv"); r.boot = (uint8_t *)P("boot");
    r.reward = (float *)P("reward"); r.close_hurt = (float *)P("close_hurt"); r.logp = (float *)P("logp"); r.value = (float *)P("value");
    r.value_g = (float *)P("value_g"); r.open = (i64 *)P("open"); r.length = (i64 *)P("length"); r.pending = (i64 *)P("pending");
    r.actions = (i64 *)P("actions"); r.version = (i64 *)P("version");
    std::lock_guard<std::mutex> g(amu);
    r.arena.ensure(std::max<i64>(arena_bytes, 1024));
    r.arena.reset();
    r.set = true;
  }
  void use_rollout(int slot) { TORCH_CHECK(slot >= -1 && slot <= 1, "use_rollout: slot"); cur_slot = slot; }
  // the frame arena of a slot (B45 FrameArena's buf / used, appends of the hand-off's carry)
  // (a view of the arena's current buffer; the view holds the buffer, so it stays valid after growth and reset; what
  // it shows after the slot's reset is the old rollout's frames)
  at::Tensor arena_buf(int slot) {
    std::lock_guard<std::mutex> g(amu);
    std::shared_ptr<uint8_t[]> keep = slots[slot].arena.cur;
    return at::from_blob(keep.get(), {slots[slot].arena.cap}, [keep](void *) {}, at::TensorOptions().dtype(at::kByte));
  }
  i64 arena_used(int slot) { std::lock_guard<std::mutex> g(amu); return slots[slot].arena.used; }
  void arena_reset(int slot) { std::lock_guard<std::mutex> g(amu); slots[slot].arena.reset(); }
  i64 arena_grown(int slot) { std::lock_guard<std::mutex> g(amu); return slots[slot].arena.grown; }
  at::Tensor arena_append_from(int slot, at::Tensor buf, at::Tensor offs, at::Tensor sizes) {
    buf = buf.contiguous(); offs = offs.to(at::kLong).contiguous(); sizes = sizes.to(at::kLong).contiguous();
    i64 n = offs.numel();
    at::Tensor out = at::empty({n}, at::kLong);
    std::lock_guard<std::mutex> g(amu);
    for (i64 i = 0; i < n; i++)
      out.data_ptr<i64>()[i] = slots[slot].arena.append(buf.data_ptr<uint8_t>()+offs.data_ptr<i64>()[i], sizes.data_ptr<i64>()[i]);
    return out;
  }
  // (tests) the consumer on given records, as the driver runs it for a step: rows uint8 [n_all * row_stride] (ROW),
  // workers, go mask, outputs [n_go, nh+3] float32, actions / pending [n_go, nh] int64, version, priv [n_all * priv size]
  py::tuple consume_rows(at::Tensor rows, std::vector<i64> ws, at::Tensor go, at::Tensor out, at::Tensor actions_, at::Tensor pend,
                         i64 ver, at::Tensor priv_) {
    rows = rows.contiguous(); go = go.to(at::kByte).contiguous(); out = out.to(at::kFloat).contiguous();
    actions_ = actions_.to(at::kLong).contiguous(); pend = pend.to(at::kLong).contiguous();
    i64 n_all = (i64)ws.size();
    std::vector<uint8_t> g(go.data_ptr<uint8_t>(), go.data_ptr<uint8_t>()+n_all);
    std::vector<i64> go_idx; for (i64 i = 0; i < n_all; i++) if (g[i]) go_idx.push_back(i);
    Record r;
    const uint8_t *pp = priv_.defined() && priv_.numel() ? priv_.contiguous().data_ptr<uint8_t>() : nullptr;
    consume(r, rows.data_ptr<uint8_t>(), n_all, ws, g, go_idx, out.data_ptr<float>(), actions_.data_ptr<i64>(), pend.data_ptr<i64>(), ver, pp);
    return py::make_tuple(r.kk, r.snap);
  }
  // (B54) a lane job: rows uint8 [n * row_stride], pending int64 [n * nh], ids / seq / lane int64 [n], replay uint8 [n]
  void submit_lanes(i64 job, i64 owner, i64 ver, at::Tensor rows, i64 n, at::Tensor pending, at::Tensor ids, at::Tensor seq,
                    at::Tensor lane, at::Tensor replay) {
    auto jb = std::make_shared<LaneJob>();
    jb->job = job; jb->owner = owner; jb->version = ver; jb->n = n;
    rows = rows.contiguous(); pending = pending.to(at::kLong).contiguous(); ids = ids.to(at::kLong).contiguous();
    seq = seq.to(at::kLong).contiguous(); lane = lane.to(at::kLong).contiguous(); replay = replay.to(at::kByte).contiguous();
    TORCH_CHECK(rows.scalar_type() == at::kByte && rows.numel() == n*row_stride, "submit_lanes: rows uint8 [n * row_stride]");
    TORCH_CHECK(pending.numel() == n*nh && ids.numel() == n && seq.numel() == n && lane.numel() == n && replay.numel() == n,
                "submit_lanes: shapes");
    jb->rows.assign(rows.data_ptr<uint8_t>(), rows.data_ptr<uint8_t>()+n*row_stride);
    jb->pend.assign(pending.data_ptr<i64>(), pending.data_ptr<i64>()+n*nh);
    jb->ids.assign(ids.data_ptr<i64>(), ids.data_ptr<i64>()+n); jb->seq.assign(seq.data_ptr<i64>(), seq.data_ptr<i64>()+n);
    jb->lane.assign(lane.data_ptr<i64>(), lane.data_ptr<i64>()+n); jb->replay.assign(replay.data_ptr<uint8_t>(), replay.data_ptr<uint8_t>()+n);
    (in_step.load() ? n_enq_in_step : n_enq_outside)++;
    std::lock_guard<std::mutex> g(lmu);
    lane_q.push_back(jb);
  }
  i64 lane_queue_size() { std::lock_guard<std::mutex> g(lmu); return (i64)lane_q.size(); }
  bool drop_lane_job(i64 job) {   // (after a kind 5 record: Python stepped the job itself)
    std::lock_guard<std::mutex> g(lmu);
    for (auto it = lane_q.begin(); it != lane_q.end(); ++it) if ((*it)->job == job) { lane_q.erase(it); return true; }
    return false;
  }
  void set_lane_merge(bool on) { lane_merge = on; }
  // (B54) a restored lane row's lineage ring: the decisions of its last block_steps steps (step s at position s % bs;
  // -1 where the branch has none). Only for rows with no lane job in flight (the consumer restores idle lanes).
  void set_ring(i64 row, at::Tensor dec, at::Tensor kv, i64 n) {
    dec = dec.to(at::kLong).contiguous(); kv = kv.to(at::kLong).contiguous();
    TORCH_CHECK(row >= 0 && row < arena_rows && dec.numel() == block_steps && kv.numel() == block_steps, "set_ring: row / shapes");
    memcpy(&ring_dec[row*block_steps], dec.data_ptr<i64>(), block_steps*8);
    memcpy(&ring_kv[row*block_steps], kv.data_ptr<i64>(), block_steps*8);
    ring_n[row] = n;
  }
  at::Tensor get_ring(i64 row) {
    at::Tensor t = at::empty({2, block_steps}, at::kLong);
    memcpy(t.data_ptr<i64>(), &ring_dec[row*block_steps], block_steps*8); memcpy(t.data_ptr<i64>()+block_steps, &ring_kv[row*block_steps], block_steps*8);
    return t;
  }
  void register_graph(std::vector<i64> key, at::Tensor host_flat, at::Tensor dev_flat, std::shared_ptr<at::cuda::CUDAGraph> graph,
                      at::Tensor out, at::Tensor cur, at::Tensor z, at::Tensor written, at::Tensor bo, at::Tensor bc, at::Tensor bz,
                      at::Tensor bw, py::dict layout, i64 heads_, i64 nh_, i64 scratch_) {
    TORCH_CHECK(key.size() == 6, "graph key: (owner, cap, wb, we, wn, ws)");
    GraphEntry e;
    e.cap = key[1]; e.wb = key[2]; e.we = key[3]; e.wn = key[4]; e.ws = key[5];
    e.host_flat = host_flat; e.dev_flat = dev_flat; e.graph = std::move(graph);
    e.out = out; e.cur = cur; e.z = z; e.written = written; e.bo = bo; e.bc = bc; e.bz = bz; e.bw = bw; e.heads = heads_; e.nh = nh_;
    e.scratch = scratch_;
    auto o = [&](const char *k) { return layout[k].cast<py::tuple>()[0].cast<i64>(); };
    e.L.room = o("v_room"); e.L.map = o("v_map"); e.L.body = o("v_body"); e.L.body_mask = o("v_body_mask"); e.L.local = o("v_local");
    e.L.local_mask = o("v_local_mask"); e.L.entity = o("v_entity"); e.L.entity_mask = o("v_entity_mask"); e.L.inventory = o("v_inventory");
    e.L.inventory_mask = o("v_inventory_mask"); e.L.slots = o("v_slots"); e.L.slots_mask = o("v_slots_mask"); e.L.pending = o("pending");
    e.L.ids = o("ids"); e.L.u = o("u"); e.L.replay = o("replay"); e.L.complete = o("complete"); e.L.version = o("version");
    std::lock_guard<std::mutex> g(gmu);
    graphs[key] = std::move(e);
  }
  bool has_graph(std::vector<i64> key) { std::lock_guard<std::mutex> g(gmu); return graphs.count(key) > 0; }
  i64 n_graphs() { std::lock_guard<std::mutex> g(gmu); return (i64)graphs.size(); }

  // ---- the step's middle: frames -> slab -> H2D -> replay -> D2H -> sync (no GIL needed; the caller holds gmu or is the driver)
  void fill_and_run(GraphEntry &e, i64 n, const uint8_t *rows, i64 stride, const i64 *pending, const i64 *ids, const float *u,
                    const uint8_t *replay, const uint8_t *complete, i64 ver, bool any_complete) {
    uint8_t *h = e.host_flat.data_ptr<uint8_t>();
    int rc = abp_visible_unpack(rows+f_frame, stride, (int)n, (int)e.wb, (int)e.we, (int)e.wn, (int)e.ws, (float *)(h+e.L.room),
                                (float *)(h+e.L.map), (float *)(h+e.L.body), h+e.L.body_mask, (float *)(h+e.L.local), h+e.L.local_mask,
                                (float *)(h+e.L.entity), h+e.L.entity_mask, (float *)(h+e.L.inventory), h+e.L.inventory_mask,
                                (float *)(h+e.L.slots), h+e.L.slots_mask);
    TORCH_CHECK(rc == 0, "abp_visible_unpack: a count exceeds its bucket width");
    i64 nh_ = e.nh, hd = e.heads;
    memcpy(h+e.L.pending, pending, n*nh_*sizeof(i64));
    i64 m = e.filled;
    if (m > n) {   // rows past n held another call's rows: zero them (every per-row field, as the Python path does)
      auto zero = [&](i64 off, i64 row_bytes) { memset(h+off+n*row_bytes, 0, (m-n)*row_bytes); };
      zero(e.L.room, 1792*4); zero(e.L.map, 169*4); zero(e.L.body, e.wb*80*4); zero(e.L.body_mask, e.wb); zero(e.L.local, e.wb*324*4);
      zero(e.L.local_mask, e.wb); zero(e.L.entity, e.we*48*4); zero(e.L.entity_mask, e.we); zero(e.L.inventory, e.wn*4*4);
      zero(e.L.inventory_mask, e.wn); zero(e.L.slots, e.ws*13*4); zero(e.L.slots_mask, e.ws); zero(e.L.pending, nh_*8);
    }
    e.filled = n;
    i64 *hid = (i64 *)(h+e.L.ids);
    for (i64 i = 0; i < e.cap; i++) hid[i] = i < n ? ids[i] : (e.scratch >= 0 ? e.scratch : scratch_row);
    float *hu = (float *)(h+e.L.u);
    for (i64 i = 0; i < e.cap*hd; i++) hu[i] = .5f;
    if (u) memcpy(hu, u, n*hd*sizeof(float));
    memset(h+e.L.replay, 0, e.cap); memcpy(h+e.L.replay, replay, n);
    memset(h+e.L.complete, 0, e.cap); memcpy(h+e.L.complete, complete, n);
    *(i64 *)(h+e.L.version) = ver;
    e.dev_flat.copy_(e.host_flat, true);
    e.graph->replay();
    e.bo.copy_(e.out, true); e.bc.copy_(e.cur, true); e.bz.copy_(e.z, true);
    if (any_complete) e.bw.copy_(e.written, true);
    c10::cuda::getCurrentCUDAStream().synchronize();
  }
  // Python entry (ActorStepGraph.__call__): numpy-backed tensors; the caller's current stream is used
  void run_graph(std::vector<i64> key, at::Tensor rows, i64 stride, i64 n, at::Tensor pending, at::Tensor ids, py::object u,
                 at::Tensor replay, at::Tensor complete, i64 ver) {
    GraphEntry *e;
    { std::lock_guard<std::mutex> g(gmu); auto it = graphs.find(key); TORCH_CHECK(it != graphs.end(), "run_graph: graph not registered"); e = &it->second; }
    TORCH_CHECK(rows.scalar_type() == at::kByte && pending.scalar_type() == at::kLong && ids.scalar_type() == at::kLong, "run_graph: dtypes");
    const float *up = nullptr; at::Tensor ut;
    if (!u.is_none()) { ut = u.cast<at::Tensor>().to(at::kFloat).contiguous(); up = ut.data_ptr<float>(); }
    at::Tensor rp = replay.to(at::kByte).contiguous(), cp = complete.to(at::kByte).contiguous();
    bool any = false; for (i64 i = 0; i < n; i++) any |= cp.data_ptr<uint8_t>()[i] != 0;
    py::gil_scoped_release nogil;
    fill_and_run(*e, n, rows.data_ptr<uint8_t>(), stride, pending.contiguous().data_ptr<i64>(), ids.contiguous().data_ptr<i64>(), up,
                 rp.data_ptr<uint8_t>(), cp.data_ptr<uint8_t>(), ver, any);
  }

  // ================================================================ (B58d) the native lane loop: its own thread
  // abpc::LaneCore (abp_lanes.h) runs the lane service; its GPU work (a branch's restore, the jobs' lane-only graph
  // replays) goes on the lane stream (the lane graphs captured in a pool of their own: they replay while the main
  // graphs do); the lane rows are the arena rows base(gen) + lane, which only this thread touches. A job the lane
  // thread cannot replay (a graph to capture: kind 1, a frame the graphs cannot take: kind 5) is a record for Python,
  // the lane thread waits for its answer (lane_job_resume / lane_python_done).
  abpc::LaneCore lc;
  bool lanes_native = false;
  std::thread lth;
  std::atomic<bool> lrunning{false}, lstop{false}, lpause_req{false}, lpaused{false};
  std::mutex lcmu; std::condition_variable lccv;
  i64 lane_stream_id = 0;
  std::vector<int> lane_fds;
  std::string lerror;
  std::vector<i64> gen_owner, gen_base;   // per generation (N: the point table's)
  std::atomic<int> l_wait{0};          // 1: a lane job waits for Python
  abpc::LaneJobOut l_py_out; bool l_py_done = false;
  std::atomic<i64> n_l_jobs{0}, n_l_rows{0}, n_l_restores{0}, n_l_python{0}, n_l_captures{0};
  std::atomic<double> t_l_restore{0}, t_l_job{0}, t_l_pause{0};

  // ---- (d2 M1) a lane job of the main version rides the driver's next main batch (one replay) instead of a lane-only
  // call; an older generation's job, a job whose version changed while it waited (the main weights were taken), one
  // the merged batch cannot take (cap, widths, no graph) and every job while the driver has no main rows go lane-only
  // at once, in the same pass. Only the driver thread touches merge_q (and LaneCore): no lock, the thread asserted.
  // With M1 the driver thread does all the lane GPU work (restores, lane-only calls) on the main stream: the merged
  // step needs no cross-stream order. ISAAC_BLOCK_LANE_MERGE=0 (set_m1_merge(false)): the d1 behaviour.
  // A waiting job pins its teacher generation (refs + 1): its slot cannot be re-published under it.
  struct MergeJob { std::shared_ptr<LaneJob> jb; i64 g; double t_enq; };
  std::deque<MergeJob> merge_q;
  bool m1_merge = true;
  i64 m1_cap = 32;                      // (the largest merged batch: PRECAPTURE_MERGED_CAPS)
  int restore_test = 0;                 // (tests) 1: the restore on the lane stream behind a long kernel, no wait (must fail)
  bool m1_hold = false;                 // (tests) the waiting jobs wait for a main batch even while the driver idles
  bool m1_test_capture = false;         // (tests) a merged batch without its graph: captured by Python and retried (not refused)
  std::thread::id drv_tid;
  // jobs merged / rows merged; lane-only jobs by reason: an older version, the version changed while waiting, refused by
  // the merged batch, the driver without main rows, other (merging off / a job Python steps)
  std::atomic<i64> n_m1_jobs{0}, n_m1_rows{0}, n_lo_jobs[5] = {{0}, {0}, {0}, {0}, {0}};
  std::vector<double> m1_wait_s;        // (per merged job: enqueue -> the main step that took it)
  bool m1_on() const { return m1_merge && lanes_native && !lrunning.load() && running.load(); }
  void m1_thread_check(const char *where) {
    TORCH_CHECK(std::this_thread::get_id() == drv_tid, where, ": the M1 merge queue belongs to the driver thread");
  }
  void gen_pin(i64 g, i64 d) {
    abpc::Native &N = *store->nat;
    std::lock_guard<std::mutex> l(N.lin.mu);
    N.pts.gens.refs.at(g) += d; N.pts.gens.track(abpc::mono_now());
  }
  // the waiting jobs, lane-only now (reason: 3 the driver has no main rows; a job whose version changed: 1)
  void m1_flush(int reason) {
    if (m1_hold && reason == 3 && !main_hold) return;
    while (!merge_q.empty()) {
      m1_thread_check("m1_flush");
      MergeJob m = merge_q.front(); merge_q.pop_front();
      lane_run(*m.jb, m.g, m.jb->version != version.load() ? 1 : reason);
      gen_pin(m.g, -1);
      if (lstop || stop_req) return;
    }
  }
  void m1_done(const MergeJob &m, GraphEntry &e, const uint8_t *rows_ptr, i64 o, std::vector<i64> &ids, std::vector<i64> &steps_before,
               std::vector<uint8_t> &first, std::vector<uint8_t> &complete, i64 ver, double t_step) {
    LaneJob &jb = *m.jb;
    i64 k = jb.n;
    Record lr;
    lane_record(lr, jb, e, o, steps_before, first, complete, ver);
    write_part(e, rows_ptr, o, k, ids, steps_before, complete, first, jb.pend.data(), ver, lr);
    abpc::LaneJobOut out;
    out.out.assign(lr.out.data_ptr<float>(), lr.out.data_ptr<float>()+k*(nh+3));
    out.step_before.assign(steps_before.begin()+o, steps_before.begin()+o+k);
    out.first.assign(first.begin()+o, first.begin()+o+k); out.complete.assign(complete.begin()+o, complete.begin()+o+k);
    out.version = ver;
    out.dec.assign(lr.dec.data_ptr<i64>(), lr.dec.data_ptr<i64>()+k); out.kvs.assign(lr.kvs.data_ptr<i64>(), lr.kvs.data_ptr<i64>()+k);
    out.bid.assign(lr.bid.data_ptr<i64>(), lr.bid.data_ptr<i64>()+k);
    lc.job_done(jb.job, out);
    gen_pin(m.g, -1);
    m1_wait_s.push_back(t_step-m.t_enq);
    n_m1_jobs++; n_m1_rows = n_m1_rows+k;
    n_l_jobs++; n_l_rows = n_l_rows+k; n_lane_jobs++; n_lane_rows = n_lane_rows+k; n_merged_rows = n_merged_rows+k;
  }

  void lanes_configure(py::dict c, py::dict s, std::vector<int> fds, i64 stream_id, std::vector<i64> owners, std::vector<i64> bases) {
    TORCH_CHECK(store->lineage_on, "lanes_configure: the store's lineage first");
    TORCH_CHECK(owners.size() == bases.size() && (i64)owners.size() == store->nat->pts.gens.n(), "lanes_configure: a graph owner and a base per generation");
    auto I = [&](py::dict d, const char *k) { return d.contains(k) && !d[k].is_none() ? d[k].cast<i64>() : (i64)0; };
    auto F = [&](py::dict d, const char *k) { return d[k].cast<double>(); };
    abpc::LaneCfg cfg; cfg.lanes = I(c, "lanes"); cfg.nh = nh; cfg.tw = I(c, "tw"); cfg.wait = F(c, "wait"); cfg.quiet = F(c, "quiet");
    cfg.active_s = F(c, "active_s"); cfg.defer_s = F(c, "defer_s"); cfg.reclaim_s = F(c, "reclaim_s"); cfg.pid = I(c, "pid");
    abpc::LaneShared sh;
    sh.rows = (const uint8_t *)I(s, "rows"); sh.row_size = row_stride; sh.o_episode = I(s, "o_episode"); sh.s_episode = (int)I(s, "s_episode");
    sh.io = (int32_t *)I(s, "io"); sh.io_cols = I(s, "io_cols"); sh.out = (double *)I(s, "out"); sh.out_cols = I(s, "out_cols");
    sh.t = (float *)I(s, "t"); sh.v2_t = (const float *)I(s, "v2_t"); sh.v2_ring = I(s, "v2_ring");
    sh.mode = (const i64 *)I(s, "mode"); sh.status = (i64 *)I(s, "status"); sh.pts_per = I(s, "pts_per");
    sh.kept = (const i64 *)I(s, "kept"); sh.kept_cols = I(s, "kept_cols"); sh.cf_snap = (i64 *)I(s, "cf_snap");
    lane_fds = fds; lane_stream_id = stream_id;
    gen_owner = owners; gen_base = bases;
    lc.configure(cfg, sh, store->nat.get());
    lc.ops.restore = [this](i64 g, i64 lane, i64 snap) { lane_restore(g, lane, snap); };
    lc.ops.submit = [this](i64 job, i64 g, std::vector<uint8_t> rows, std::vector<i64> pend, std::vector<i64> lanes, std::vector<i64> seq,
                           std::vector<uint8_t> replay) { lane_job(job, g, std::move(rows), std::move(pend), std::move(lanes), std::move(seq), std::move(replay)); };
    lc.ops.reply = [this](const std::vector<i64> &lanes) {
      static const char R[5] = {0, 0, 0, 1, 'a'};
      for (i64 l : lanes) if (l >= 0 && l < (i64)lane_fds.size() && lane_fds[l] >= 0) { ssize_t k = ::write(lane_fds[l], R, 5); (void)k; }
    };
    lc.ops.gen_version = [this](i64 g) { std::lock_guard<std::mutex> l(store->nat->lin.mu); return store->nat->pts.gens.version[g]; };
    lc.ops.main_version = [this]() { return version.load(); };
    lc.ops.base = [this](i64 g) { return gen_base[g]; };
    lanes_native = true;
    lane_merge = false;   // (lane rows never go into main batches: the main graphs and the lane graphs replay concurrently)
  }
  // state.load (BlockSlotState) of arena row base(g) + lane from a snapshot's parts, on the lane stream; the row's head
  // the snapshot's branch, its lineage ring from that log
  void lane_restore(i64 g, i64 lane, i64 snap) {
    double t0 = now();
    i64 row = gen_base[g]+lane;
    abpc::Native &N = *store->nat;
    abpc::Parts p;
    std::vector<i64> rdec(block_steps, -1), rkv(block_steps, -1); i64 rn = 0;
    {
      std::lock_guard<std::mutex> l(N.lin.mu);
      p = N.lin.parts_locked(snap);
      N.lin.set_head_locked(row, N.lin.branch_locked(snap));
      const abpc::LogData &hg = N.lin.logs[N.lin.head[row]];
      i64 step = hg.end(), lo = std::max(hg.offset, step-block_steps);
      for (i64 s_ = lo; s_ < step; s_++) { rdec[s_ % block_steps] = hg.dec[s_-hg.offset]; rkv[s_ % block_steps] = hg.kvs[s_-hg.offset]; }
      rn = step-lo;
    }
    abpc::PageCore &pc = *store->core;
    at::Tensor kv_t = reset.tensors[0];
    at::TensorOptions H = at::TensorOptions().pinned_memory(true);
    i64 n = (i64)p.kvs.size(), np_ = (i64)p.pdec.size(), m = (i64)p.bid.size();
    i64 kvb = pc.kv_row_bytes, ze = pc.z_elems, be = pc.b_elems;
    at::Tensor kvh = at::empty({std::max<i64>(n*kvb, 1)}, H.dtype(at::kByte)), zh = at::empty({std::max<i64>(np_*ze, 1)}, H.dtype(at::kFloat)),
               bh = at::empty({std::max<i64>(m*be, 1)}, H.dtype(at::kFloat));
    at::Tensor kver = at::empty({n}, at::kLong), pver = at::empty({np_}, at::kLong), bver = at::empty({m}, at::kLong);
    at::Tensor bsrc = at::empty({m, block_steps}, at::kLong), bsrcv = at::empty({m, block_steps}, at::kLong);
    {
      std::lock_guard<std::mutex> l(pc.mu);
      for (i64 i = 0; i < n; i++) { memcpy(kvh.data_ptr<uint8_t>()+i*kvb, pc.kv_row(p.kvs[i]), kvb); kver.data_ptr<i64>()[i] = pc.d_meta[2*p.kvs[i]+1]; }
      for (i64 i = 0; i < np_; i++) { memcpy(zh.data_ptr<float>()+i*ze, pc.z_row(p.pkvs[i]), ze*4); pver.data_ptr<i64>()[i] = pc.d_meta[2*p.pkvs[i]+1]; }
      for (i64 i = 0; i < m; i++) {
        i64 b = p.bid[i];
        memcpy(bh.data_ptr<float>()+i*be, pc.b_row(b), be*4); bver.data_ptr<i64>()[i] = pc.b_meta[2*b+1];
        memcpy(bsrc.data_ptr<i64>()+i*block_steps, pc.b_src.get()+b*block_steps, block_steps*8);
        memcpy(bsrcv.data_ptr<i64>()+i*block_steps, pc.b_srcver.get()+b*block_steps, block_steps*8);
      }
    }
    std::unique_ptr<c10::cuda::CUDAStreamGuard> test_guard;
    if (restore_test == 1) {   // (the control: the restore's copies on the lane stream behind a long kernel; nothing waits)
      auto ls = c10::cuda::CUDAStream::unpack3(lane_stream_id, (c10::DeviceIndex)device_index, c10::DeviceType::CUDA);
      test_guard.reset(new c10::cuda::CUDAStreamGuard(ls));
      at::cuda::sleep(400000000LL);
    }
    reset_rows({row});
    auto dev = kv_t.device();
    auto D = [&](at::Tensor x) { return x.to(dev, /*non_blocking=*/true); };
    // reset.tensors: kv, then BlockSlotState._rows: bank, partial, steps, kv_steps, kv_versions, bank_ends, bank_versions,
    // partial_sources, partial_versions, bank_sources, bank_source_versions
    at::Tensor bank = reset.tensors[1], partial = reset.tensors[2], steps_t = reset.tensors[3], kv_steps = reset.tensors[4],
               kv_versions = reset.tensors[5], bank_ends = reset.tensors[6], bank_versions = reset.tensors[7], partial_sources = reset.tensors[8],
               partial_versions = reset.tensors[9], bank_sources = reset.tensors[10], bank_source_versions = reset.tensors[11];
    i64 lat = partial.size(2), wl = kv_steps.size(1), mb = bank.size(1);
    if (n) {
      at::Tensor st = at::from_blob(p.kv_steps.data(), {n}, at::kLong).clone();
      at::Tensor cols = (st.unsqueeze(1)*lat+at::arange(lat, at::kLong).unsqueeze(0)).remainder(wl).reshape({-1});
      at::Tensor colsd = D(cols.pin_memory());
      at::Tensor vals = D(kvh.narrow(0, 0, n*kvb).view(kv_t.scalar_type()).view({n, kv_t.size(0), kv_t.size(1), kv_t.size(3), lat, kv_t.size(5)}));
      vals = vals.permute({1, 2, 3, 0, 4, 5}).reshape({kv_t.size(0), kv_t.size(1), kv_t.size(3), n*lat, kv_t.size(5)});
      kv_t.select(2, row).index_copy_(3, colsd, vals);
      kv_steps.select(0, row).index_copy_(0, colsd, D(st.repeat_interleave(lat).pin_memory()));
      kv_versions.select(0, row).index_copy_(0, colsd, D(kver.repeat_interleave(lat).pin_memory()));
    }
    if (np_) {
      partial.select(0, row).narrow(0, 0, np_).copy_(D(zh.narrow(0, 0, np_*ze).view({np_, lat, partial.size(3)})).to(partial.scalar_type()));
      partial_versions.select(0, row).narrow(0, 0, np_).copy_(D(pver.pin_memory()));
      partial_sources.select(0, row).narrow(0, 0, np_).copy_(D(at::from_blob(p.pdec.data(), {np_}, at::kLong).clone().pin_memory()));
    }
    if (m) {
      at::Tensor ends = at::from_blob(p.ends.data(), {m}, at::kLong).clone();
      at::Tensor col = D(at::floor_divide(ends, block_steps).remainder(mb).pin_memory());
      bank.select(0, row).index_copy_(0, col, D(bh.narrow(0, 0, m*be).view({m, bank.size(2), bank.size(3)})).to(bank.scalar_type()));
      bank_ends.select(0, row).index_copy_(0, col, D(ends.pin_memory()));
      bank_versions.select(0, row).index_copy_(0, col, D(bver.pin_memory()));
      bank_sources.select(0, row).index_copy_(0, col, D(bsrc.pin_memory()));
      bank_source_versions.select(0, row).index_copy_(0, col, D(bsrcv.pin_memory()));
    }
    step_host[row] = p.step;
    steps_t.select(0, row).fill_(p.step);
    memcpy(&ring_dec[row*block_steps], rdec.data(), block_steps*8); memcpy(&ring_kv[row*block_steps], rkv.data(), block_steps*8);
    ring_n[row] = rn;
    n_l_restores++; t_l_restore = t_l_restore+(now()-t0);
  }
  // wait for Python's answer to a lane record (kind 1: captured; 5: stepped)
  void lane_wait_python(std::shared_ptr<Record> r) {
    l_wait = 1;
    push_record(r);
    std::unique_lock<std::mutex> lk(lcmu);
    while (l_wait.load() && !lstop) lccv.wait_for(lk, std::chrono::milliseconds(2));
  }
  void lane_job_resume() { std::lock_guard<std::mutex> g(lcmu); l_wait = 0; lccv.notify_all(); }
  // (kind 5) Python stepped the job on its lane actor: the rows' outputs and pages
  void lane_python_done(at::Tensor out, at::Tensor step_before, at::Tensor first, at::Tensor complete, at::Tensor dec, at::Tensor kvs,
                        at::Tensor bid, i64 ver) {
    out = out.to(at::kFloat).contiguous(); step_before = step_before.to(at::kLong).contiguous(); first = first.to(at::kByte).contiguous();
    complete = complete.to(at::kByte).contiguous(); dec = dec.to(at::kLong).contiguous(); kvs = kvs.to(at::kLong).contiguous();
    bid = bid.to(at::kLong).contiguous();
    std::lock_guard<std::mutex> g(lcmu);
    l_py_out.out.assign(out.data_ptr<float>(), out.data_ptr<float>()+out.numel());
    l_py_out.step_before.assign(step_before.data_ptr<i64>(), step_before.data_ptr<i64>()+step_before.numel());
    l_py_out.first.assign(first.data_ptr<uint8_t>(), first.data_ptr<uint8_t>()+first.numel());
    l_py_out.complete.assign(complete.data_ptr<uint8_t>(), complete.data_ptr<uint8_t>()+complete.numel());
    l_py_out.dec.assign(dec.data_ptr<i64>(), dec.data_ptr<i64>()+dec.numel());
    l_py_out.kvs.assign(kvs.data_ptr<i64>(), kvs.data_ptr<i64>()+kvs.numel());
    l_py_out.bid.assign(bid.data_ptr<i64>(), bid.data_ptr<i64>()+bid.numel());
    l_py_out.version = ver; l_py_done = true; l_wait = 0; lccv.notify_all();
  }
  void lane_job(i64 job, i64 g, std::vector<uint8_t> rows, std::vector<i64> pend, std::vector<i64> lanes, std::vector<i64> seq,
                std::vector<uint8_t> replay) {
    auto jb = std::make_shared<LaneJob>();
    jb->job = job; jb->owner = gen_owner[g]; jb->version = lc.ops.gen_version(g); jb->n = (i64)lanes.size();
    jb->rows = std::move(rows); jb->pend = std::move(pend); jb->seq = std::move(seq); jb->lane = std::move(lanes); jb->replay = std::move(replay);
    for (i64 l : jb->lane) jb->ids.push_back(gen_base[g]+l);
    bool cur = jb->version == version.load();
    if (cur && m1_on() && lane_force_python != -2 && lane_force_python != g) {   // (d2 M1: the next main batch takes it)
      m1_thread_check("lane_job");
      gen_pin(g, 1);
      merge_q.push_back(MergeJob{jb, g, now()});
      return;
    }
    lane_run(*jb, g, cur ? 4 : 0);
  }
  // a lane-only call of one job through its generation's graph (d1), or its kind 5 / kind 1 hand-over to Python
  void lane_run(LaneJob &jb, i64 g, int reason) {
    double t0 = now();
    i64 job = jb.job;
    const std::vector<i64> &lanes = jb.lane, &seq = jb.seq;
    ((jb.version == version.load()) ? n_lo_cur_rows : n_lo_old_rows) += jb.n;
    n_lo_jobs[reason]++;
    i64 k = jb.n;
    while (true) {
      i64 wb, we, wn, wsl;
      bool ok = batch_widths(jb.rows.data(), k, wb, we, wn, wsl) && lane_force_python != -2 && lane_force_python != g;
      i64 cap = 1; while (cap < k) cap <<= 1;
      auto rec = [&](int kind) {
        auto r = std::make_shared<Record>(); finish_empty(*r);   // (first: finish_empty sets the main version)
        r->kind = kind; r->job = job; r->owner = jb.owner; r->version = jb.version; r->lane_src = 1;   // (the generation's)
        r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl; r->point = g;
        return r;
      };
      if (!ok) {   // the frames the graphs cannot take: Python steps the job on its lane actor
        auto r = rec(5);
        r->rows = at::empty({k*row_stride}, at::kByte); memcpy(r->rows.data_ptr<uint8_t>(), jb.rows.data(), k*row_stride);
        r->pending_before = at::empty({k, nh}, at::kLong); memcpy(r->pending_before.data_ptr<i64>(), jb.pend.data(), k*nh*8);
        auto T = [](const std::vector<i64> &v) { at::Tensor t = at::empty({(i64)v.size()}, at::kLong); memcpy(t.data_ptr<i64>(), v.data(), v.size()*8); return t; };
        r->lane_ids = T(lanes); r->lane_seq = T(seq);
        r->lane_replay = at::empty({k}, at::kByte); memcpy(r->lane_replay.data_ptr<uint8_t>(), jb.replay.data(), k);
        { std::lock_guard<std::mutex> lg(lcmu); l_py_done = false; }
        n_l_python++;
        lane_wait_python(r);
        if (lstop) return;
        abpc::LaneJobOut o;
        { std::lock_guard<std::mutex> lg(lcmu); o = l_py_out; }
        lc.job_done(job, o);
        break;
      }
      std::vector<i64> key{jb.owner, cap, wb, we, wn, wsl};
      GraphEntry *e = find_graph(key);
      if (!e) { n_l_captures++; lane_wait_python(rec(1)); if (lstop) return; continue; }
      std::vector<i64> steps_before(k); std::vector<uint8_t> complete(k), first(k);
      std::vector<i64> to_reset;
      for (i64 j = 0; j < k; j++) {
        first[j] = read_int(jb.rows.data()+j*row_stride+f_first, s_first) != 0;
        if (first[j]) to_reset.push_back(jb.ids[j]);
      }
      if (!to_reset.empty()) reset_rows(to_reset);
      bool any_complete = false;
      for (i64 j = 0; j < k; j++) { steps_before[j] = step_host[jb.ids[j]]; complete[j] = (steps_before[j]+1) % block_steps == 0; any_complete |= complete[j]; }
      std::vector<float> u(k*heads);
      lane_uniforms(jb, u.data());
      fill_and_run(*e, k, jb.rows.data(), row_stride, jb.pend.data(), jb.ids.data(), u.data(), jb.replay.data(), complete.data(), jb.version, any_complete);
      Record r;
      lane_record(r, jb, *e, 0, steps_before, first, complete, jb.version);
      write_part(*e, jb.rows.data(), 0, k, jb.ids, steps_before, complete, first, jb.pend.data(), jb.version, r);
      abpc::LaneJobOut o;
      o.out.assign(r.out.data_ptr<float>(), r.out.data_ptr<float>()+k*(nh+3));
      o.step_before = steps_before; o.first = first; o.complete = complete; o.version = jb.version;
      o.dec.assign(r.dec.data_ptr<i64>(), r.dec.data_ptr<i64>()+k); o.kvs.assign(r.kvs.data_ptr<i64>(), r.kvs.data_ptr<i64>()+k);
      o.bid.assign(r.bid.data_ptr<i64>(), r.bid.data_ptr<i64>()+k);
      lc.job_done(job, o);
      lane_hist[k <= 1 ? 0 : k <= 2 ? 1 : k <= 4 ? 2 : k <= 8 ? 3 : k <= 16 ? 4 : 5]++;
      break;
    }
    n_l_jobs++; n_l_rows = n_l_rows+k; n_lane_jobs++; n_lane_calls++; n_lane_rows = n_lane_rows+k;
    t_l_job = t_l_job+(now()-t0); t_lane_s = t_lane_s+(now()-t0);
  }
  // one pass of the lane service (B54's tick): control, a finished round answered, the lane pipes read (up to timeout
  // s), a due round submitted; its GPU work on the lane stream. The lane thread (d2) or the driver thread (d1) runs it.
  bool lane_pass(double timeout) {
    i64 sid = (m1_merge && !lrunning.load() && std::this_thread::get_id() == drv_tid) ? stream_id : lane_stream_id;
    auto stream = c10::cuda::CUDAStream::unpack3(sid, (c10::DeviceIndex)device_index, c10::DeviceType::CUDA);
    c10::cuda::CUDAStreamGuard guard(stream);
    double t = now();
    lc.control(t);
    bool did = false;
    if (lc.fly && lc.fly->ready) { lc.finish(t); did = true; }
    double wl = lc.wait_left(t);
    double to = wl < 0 ? timeout : std::min(wl, timeout);
    if (lpfds.size() != lane_fds.size()) lpfds.resize(lane_fds.size());
    for (size_t i = 0; i < lane_fds.size(); i++) { lpfds[i].fd = lane_fds[i]; lpfds[i].events = lane_fds[i] >= 0 ? POLLIN : 0; lpfds[i].revents = 0; }
    struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = (long)(std::max(0.0, to)*1e9);
    int k = ::ppoll(lpfds.data(), (nfds_t)lpfds.size(), &ts, nullptr);
    for (size_t i = 0; k > 0 && i < lane_fds.size(); i++) {
      if (!(lpfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      unsigned char msg[5]; ssize_t got = ::read(lane_fds[i], msg, 5);
      if (got == 5 && msg[0] == 0 && msg[1] == 0 && msg[2] == 0 && msg[3] == 1 && (msg[4] == '0' || msg[4] == '1')) lc.intake((i64)i, now());
      else lane_fds[i] = -1;   // (the lane closed)
    }
    t = now();
    if (!lc.ready.empty() && !lc.fly && lc.due(t)) {
      lc.submit(t);
      if (lc.fly && lc.fly->ready) lc.finish(now());
      did = true;
    }
    return did;
  }
  std::vector<struct pollfd> lpfds;
  void lane_loop() {
    try {
      c10::InferenceMode inference;
      while (!lstop) {
        if (lpause_req) {
          double t0 = now();
          std::unique_lock<std::mutex> lk(lcmu);
          lpaused = true; lccv.notify_all();
          while (lpause_req && !lstop) lccv.wait_for(lk, std::chrono::milliseconds(2));
          lpaused = false; t_l_pause = t_l_pause+(now()-t0);
          continue;
        }
        lane_pass(1e-3);
      }
    } catch (const std::exception &ex) {
      lerror = ex.what();
      auto r = std::make_shared<Record>(); r->kind = 3; r->lane_src = 1; finish_empty(*r); push_record(r);
    }
    lrunning = false;
  }
  void lanes_start() {
    TORCH_CHECK(lanes_native && !lrunning, "lanes_start: configure first / already running");
    lstop = false; lpause_req = false; lpaused = false; lerror.clear(); lrunning = true;
    lth = std::thread([this] { lane_loop(); });
  }
  void lanes_stop() {
    if (!lth.joinable()) return;
    lstop = true;
    { std::lock_guard<std::mutex> g(lcmu); lccv.notify_all(); }
    lth.join();
    lrunning = false;
  }
  // pause: the lane thread finishes its pass and waits (a job waiting for Python counts as paused: nothing runs)
  void lanes_pause() {
    if (!lrunning) return;
    lpause_req = true;
    std::unique_lock<std::mutex> lk(lcmu);
    while (!lpaused && lrunning && !l_wait.load()) lccv.wait_for(lk, std::chrono::milliseconds(2));
  }
  void lanes_resume() { lpause_req = false; std::lock_guard<std::mutex> g(lcmu); lccv.notify_all(); }
  // (tests) the lane service driven by the caller with an injected clock: a request read at `now`; one pass
  // (control, a finished round, a due round) at `now`, without the pipes
  i64 lane_force_python = -1;   // (tests) -1 none, -2 every job, g: generation g's jobs go to Python (kind 5)
  void lane_test_intake(i64 lane, double now) { lc.intake(lane, now); }
  bool lane_test_tick(double now) {
    c10::InferenceMode inference;
    auto stream = c10::cuda::CUDAStream::unpack3(lane_stream_id, (c10::DeviceIndex)device_index, c10::DeviceType::CUDA);
    c10::cuda::CUDAStreamGuard guard(stream);
    lc.control(now);
    bool did = false;
    if (lc.fly && lc.fly->ready) { lc.finish(now); did = true; }
    if (!lc.ready.empty() && !lc.fly && lc.due(now)) {
      lc.submit(now);
      if (lc.fly && lc.fly->ready) lc.finish(now);
      did = true;
    }
    return did;
  }
  py::dict lane_state() {   // (tests) the service's bookkeeping
    py::dict d;
    py::dict canc, lp, ls;
    for (auto &e : lc.cancelled_req) canc[py::int_(e.first)] = e.second;
    for (auto &e : lc.lane_point) lp[py::int_(e.first)] = e.second;
    for (auto &e : lc.lane_steps) ls[py::int_(e.first)] = e.second;
    d["cancelled"] = canc; d["lane_point"] = lp; d["lane_steps"] = ls; d["ready"] = lc.ready; d["fly"] = (bool)lc.fly;
    return d;
  }
  // (D1 / B58d) this collect's lane request and point timings (p50 / p90; the lists restart) and the seconds every
  // generation was pinned
  py::dict lane_timing() {
    // (d2: the lane thread writes these lists: it stands still while they are read)
    bool held = lrunning.load() && !lpause_req.load();
    if (held) { py::gil_scoped_release nogil; lanes_pause(); }
    struct Resume { ActorDriver *d; bool on; ~Resume() { if (on) d->lanes_resume(); } } resume_{this, held};
    auto q = [](std::vector<double> &v, double p) -> py::object {
      if (v.empty()) return py::str("");
      std::vector<double> s = v; std::sort(s.begin(), s.end());
      double x = p*(s.size()-1); size_t i = (size_t)x; double f = x-i;
      return py::float_(i+1 < s.size() ? s[i]*(1-f)+s[i+1]*f : s[i]);
    };
    py::dict d;
    auto put = [&](const char *name, std::vector<double> &v) {
      d[(std::string(name)+"_n").c_str()] = (i64)v.size();
      d[(std::string(name)+"_p50").c_str()] = q(v, .5); d[(std::string(name)+"_p90").c_str()] = q(v, .9); v.clear();
    };
    put("m1_wait_s", m1_wait_s);
    put("lt_window_s", lc.win_s); put("lt_worker_s", lc.wk_s); put("lt_driver_s", lc.drv_s); put("lt_gpu_s", lc.gpu_s); put("lt_rt_s", lc.rt_s);
    abpc::Native &N = *store->nat;
    std::lock_guard<std::mutex> l(N.lin.mu);
    put("pt_first_s", N.pts.first_s); put("pt_run_s", N.pts.run_s); put("pt_life_s", N.pts.life_s); put("pt_life_v", N.pts.life_v);
    d["pt_refused"] = N.pts.refused; N.pts.refused = 0;
    d["pt_all_pinned_s"] = N.pts.gens.take_all_s(abpc::mono_now());
    { auto lv = N.pts.take_live(abpc::mono_now()); d["pt_live_mean"] = lv.first; d["pt_live_max"] = lv.second; }
    return d;
  }
  py::dict lane_counters() {
    py::dict d;
    d["rounds"] = lc.n_rounds; d["rows"] = lc.n_rows; d["jobs"] = n_l_jobs.load(); d["job_rows"] = n_l_rows.load(); d["restores"] = n_l_restores.load();
    d["python_jobs"] = n_l_python.load(); d["captures"] = n_l_captures.load(); d["deferred"] = lc.n_deferred; d["orphan"] = lc.n_orphan;
    d["retired"] = lc.n_retired; d["held_s"] = lc.held_s; d["fly_s"] = lc.fly_s; d["restore_s"] = t_l_restore.load(); d["job_s"] = t_l_job.load();
    d["pause_s"] = t_l_pause.load(); d["control_scans"] = lc.n_cancel_scans; d["begins"] = lc.n_begins;
    return d;
  }

  // ================================================================ (B58e) the main records in batches
  // With the native consumer and lineage, Python needs nothing per step but the episode events (the rows with a first or
  // an end) and the counters progress.csv shows (infer_ms, rows_per_call, polls, ...). In batch mode the driver keeps
  // those counters itself, collects the event rows (header + frame, worker, the consumer's per-row snapshot) and hands
  // them over as one kind 8 record when ev_max rows are waiting; Python takes the rest at the collect's end and before a
  // hand-off (take_events). A collect target (set_collect_target: usable decisions of the current rollout slot) makes the
  // driver push one kind 7 record when it is reached, so the main thread blocks on next_record instead of polling.
  bool batch_on = false;
  i64 ev_max = 256;
  std::mutex evmu;
  std::vector<uint8_t> ev_rows; std::vector<i64> ev_ws; std::vector<double> ev_snap;
  std::atomic<i64> collect_target{-1};
  std::atomic<bool> busy_flag{false};   // (Python: an update runs on the learner: the calls_busy / act_busy counters)
  std::atomic<i64> b_polls{0}, b_calls{0}, b_rows{0}, b_calls_busy{0}, b_batches{0}, b_ev_rows{0};
  std::atomic<double> b_latency{0}, b_act{0}, b_act_busy{0};
  void set_batch(bool on, i64 max_rows) { batch_on = on; ev_max = std::max<i64>(1, max_rows); }
  void set_collect_target(i64 n) { collect_target = n; }
  void set_busy(bool on) { busy_flag = on; }
  std::atomic<bool> test_fail{false};   // (B58e test: the driver thread throws at its next iteration: a driver error)
  // (Python, any thread) one kind 9 record: the main thread's blocking wait returns (new weights to take)
  void wake() { auto w = std::make_shared<Record>(); finish_empty(*w); w->kind = 9; push_record(w); }
  // the usable decisions of the current rollout slot (Rollout.usable: length.sum() - (open >= 0).sum())
  i64 usable_now() {
    int sl = cur_slot.load();
    if (sl < 0 || !slots[sl].set) return 0;
    const RollSlot &s = slots[sl];
    i64 u = 0;
    for (i64 w = 0; w < n_workers; w++) { u += abpc::load_i64(&s.length[w]); u -= abpc::load_i64(&s.open[w]) >= 0; }
    return u;
  }
  // instead of push_record for a main record (kind 0) in batch mode
  void deliver(std::shared_ptr<Record> r, const uint8_t *rc, i64 n_all, const std::vector<i64> &ws, i64 n_go) {
    if (!batch_on) { push_record(r); return; }
    b_polls++;
    if (n_go > 0) {
      double lat = r->t_reply-r->t_poll;
      b_calls++; b_rows = b_rows+n_go; b_latency = b_latency+lat; b_act = b_act+(r->t_done-r->t_poll);
      if (busy_flag.load()) { b_calls_busy++; b_act_busy = b_act_busy+lat; }
    }
    std::shared_ptr<Record> batch;
    {
      std::lock_guard<std::mutex> g(evmu);
      const double *sn = r->snap.defined() && r->snap.numel() == n_all*N_SNAP ? r->snap.data_ptr<double>() : nullptr;
      for (i64 i = 0; i < n_all; i++) {
        const uint8_t *src = rc+i*row_stride;
        if (read_int(src+f_first, s_first) == 0 && read_int(src+f_done, s_done) == 0) continue;
        ev_rows.insert(ev_rows.end(), src, src+row_stride);
        ev_ws.push_back(ws[i]);
        for (i64 q = 0; q < N_SNAP; q++) ev_snap.push_back(sn ? sn[i*N_SNAP+q] : 0.0);
      }
      if ((i64)ev_ws.size() >= ev_max) batch = take_events_locked();
    }
    if (batch) push_record(batch);
    i64 tgt = collect_target.load();
    if (tgt >= 0 && usable_now() >= tgt) {
      collect_target = -1;
      auto k7 = std::make_shared<Record>(); finish_empty(*k7); k7->kind = 7; push_record(k7);
    }
  }
  std::shared_ptr<Record> take_events_locked() {
    auto b = std::make_shared<Record>(); finish_empty(*b); b->kind = 8; b->kk = at::empty({0}, at::kLong);
    i64 m = (i64)ev_ws.size();
    b->rows = at::empty({(i64)ev_rows.size()}, at::kByte); if (m) memcpy(b->rows.data_ptr<uint8_t>(), ev_rows.data(), ev_rows.size());
    b->workers = ev_ws;
    b->snap = at::empty({m, N_SNAP}, at::kDouble); if (m) memcpy(b->snap.data_ptr<double>(), ev_snap.data(), ev_snap.size()*8);
    ev_rows.clear(); ev_ws.clear(); ev_snap.clear();
    b_batches++; b_ev_rows = b_ev_rows+m;
    return b;
  }
  // (Python, the driver paused) the events still waiting, as a kind 8 record's fields (rows, workers, snap)
  py::tuple take_events() {
    std::shared_ptr<Record> b;
    { std::lock_guard<std::mutex> g(evmu); b = take_events_locked(); }
    at::Tensor w = at::empty({(i64)b->workers.size()}, at::kLong);
    if (!b->workers.empty()) memcpy(w.data_ptr<i64>(), b->workers.data(), b->workers.size()*8);
    return py::make_tuple(b->rows, w, b->snap);
  }
  py::dict batch_counters() {
    py::dict d;
    d["polls"] = b_polls.load(); d["calls"] = b_calls.load(); d["rows"] = b_rows.load(); d["calls_busy"] = b_calls_busy.load();
    d["latency"] = b_latency.load(); d["act"] = b_act.load(); d["act_busy"] = b_act_busy.load();
    d["batches"] = b_batches.load(); d["ev_rows"] = b_ev_rows.load();
    return d;
  }

  // ---- the driver thread
  void start() {
    TORCH_CHECK(!running, "driver already running");
    TORCH_CHECK(have_stream && rows_base && step_host && n_workers > 0, "driver: set_stream / set_sampler / set_arena first");
    stop_req = false; pause_req = false; paused = false; error.clear();
    running = true;
    th = std::thread([this] { loop(); });
  }
  void stop() {   // (B58e: a thread that ended on an error is joined too: a joinable std::thread destroyed aborts)
    if (!running && !th.joinable()) return;
    stop_req = true;
    { std::lock_guard<std::mutex> g(cmu); ccv.notify_all(); }
    if (th.joinable()) th.join();
    running = false;
  }
  // pause: the driver finishes its current step and waits; resume: continues (a pending capture-needed batch is retried)
  void pause() {
    if (!running) return;
    pause_req = true;
    std::unique_lock<std::mutex> lk(cmu);
    while (!paused && running && error.empty()) ccv.wait_for(lk, std::chrono::milliseconds(5));
  }
  void resume() {
    pause_req = false;
    std::lock_guard<std::mutex> g(cmu); ccv.notify_all();
  }
  bool is_paused() { return paused; }
  std::string last_error() { return error; }
  py::object next_record(double timeout_s) {
    std::shared_ptr<Record> r;
    {
      py::gil_scoped_release nogil;
      std::unique_lock<std::mutex> lk(qmu);
      if (queue.empty()) qcv.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] { return !queue.empty(); });
      if (!queue.empty()) { r = queue.front(); queue.pop_front(); qcv.notify_all(); }
    }
    if (!r) return py::none();
    auto v = [](const std::vector<i64> &x) { at::Tensor t = at::empty({(i64)x.size()}, at::kLong); memcpy(t.data_ptr<i64>(), x.data(), x.size()*8); return t; };
    py::dict d;
    d["kind"] = r->kind; d["workers"] = v(r->workers); d["slots"] = v(r->slots);
    at::Tensor go = at::empty({(i64)r->go.size()}, at::kByte); memcpy(go.data_ptr<uint8_t>(), r->go.data(), r->go.size()); d["go"] = go;
    d["rows"] = r->rows; d["priv"] = r->priv; d["cf_snap"] = v(r->cf_snap);
    d["out"] = r->out; d["actions"] = r->actions; d["pending_before"] = r->pending_before; d["step_before"] = r->step_before;
    d["first"] = r->first; d["complete"] = r->complete; d["dec"] = r->dec; d["kvs"] = r->kvs; d["bid"] = r->bid; d["version"] = r->version;
    d["cap"] = r->cap; d["widths"] = py::make_tuple(r->wb, r->we, r->wn, r->ws);
    d["t_poll"] = r->t_poll; d["t_sync"] = r->t_sync; d["t_reply"] = r->t_reply; d["t_pages"] = r->t_pages; d["t_done"] = r->t_done;
    d["job"] = r->job; d["owner"] = r->owner; d["merged"] = (int)r->merged; d["t_main_reply"] = r->t_main_reply;
    if (r->kk.defined()) { d["kk"] = r->kk; d["snap"] = r->snap; } else { d["kk"] = py::none(); d["snap"] = py::none(); }
    d["point"] = r->point; d["pub_slot"] = r->pub_slot; d["pub_snap"] = r->pub_snap; d["pub_worker"] = r->pub_worker;
    d["lane_src"] = r->lane_src;
    if (r->lane_ids.defined()) { d["lane_ids"] = r->lane_ids; d["lane_seq"] = r->lane_seq; d["lane_replay"] = r->lane_replay; }
    return d;
  }
  i64 queue_size() { std::lock_guard<std::mutex> g(qmu); return (i64)queue.size(); }
  py::dict counters() {
    py::dict d;
    d["steps"] = n_steps.load(); d["calls"] = n_calls.load(); d["rows"] = n_rows.load(); d["terminal"] = n_terminal.load();
    d["sync_s"] = t_sync_s.load(); d["pages_s"] = t_pages_s.load(); d["host_s"] = t_host_s.load();
    d["lane_jobs"] = n_lane_jobs.load(); d["lane_rows"] = n_lane_rows.load(); d["merged_rows"] = n_merged_rows.load();
    d["merged_calls"] = n_merged_calls.load(); d["lane_calls"] = n_lane_calls.load(); d["captures"] = n_captures.load();
    d["lane_python"] = n_lane_python.load(); d["pauses"] = n_pauses.load(); d["pause_s"] = t_pause_s.load();
    d["lane_s"] = t_lane_s.load(); d["cap_holds"] = n_cap_holds.load();
    d["poll_s"] = t_poll_s.load(); d["prep_s"] = t_prep_s.load(); d["reply_s"] = t_reply_s.load(); d["cons_s"] = t_cons_s.load();
    d["push_s"] = t_push_s.load(); d["mlane_s"] = t_mlane_s.load(); d["hold_s"] = t_hold_s.load(); d["loop_s"] = t_loop_s.load();
    d["polls"] = n_polls.load(); d["idle_polls"] = n_idle_polls.load();
    d["cap_native"] = n_cap_native.load(); d["cap_refused"] = n_cap_refused.load(); d["cap_publish"] = n_cap_publish.load();
    d["lin_s"] = t_lin_s.load();
    d["m1_jobs"] = n_m1_jobs.load(); d["m1_rows"] = n_m1_rows.load();
    d["lo_old_jobs"] = n_lo_jobs[0].load(); d["lo_switch_jobs"] = n_lo_jobs[1].load(); d["lo_refused_jobs"] = n_lo_jobs[2].load();
    d["lo_idle_jobs"] = n_lo_jobs[3].load(); d["lo_other_jobs"] = n_lo_jobs[4].load(); d["m1_waiting"] = (i64)merge_q.size();
    d["lo_cur_rows"] = n_lo_cur_rows.load(); d["lo_old_rows"] = n_lo_old_rows.load(); d["putback_rows"] = n_putback_rows.load();
    d["enq_in_step"] = n_enq_in_step.load(); d["enq_outside"] = n_enq_outside.load();
    py::list h; for (auto &x : lane_hist) h.append(x.load()); d["lane_hist"] = h;
    return d;
  }

 private:
  static i64 bucket(const std::vector<i64> &b, i64 m) { for (i64 w : b) if (m <= w) return w; return -1; }

  void push_record(std::shared_ptr<Record> r) {
    double t0 = now();
    std::unique_lock<std::mutex> lk(qmu);
    qcv.wait(lk, [&] { return (i64)queue.size() < queue_max || stop_req; });
    queue.push_back(std::move(r));
    qcv.notify_all();
    lk.unlock();
    t_push_s = t_push_s+(now()-t0);
  }
  void wait_paused() {
    double t0 = now();
    std::unique_lock<std::mutex> lk(cmu);
    paused = true; ccv.notify_all();
    while (pause_req && !stop_req) ccv.wait_for(lk, std::chrono::milliseconds(2));
    paused = false;
    t_pause_s = t_pause_s+(now()-t0); n_pauses++;
  }
  void reset_rows(const std::vector<i64> &rows) {
    at::Tensor ids = at::from_blob((void *)rows.data(), {(i64)rows.size()}, at::kLong).to(reset.tensors[0].device());
    for (size_t k = 0; k < reset.tensors.size(); k++) reset.tensors[k].index_fill_(reset.dims[k], ids, reset.fills[k]);
    for (i64 r : rows) { step_host[r] = 0; ring_n[r] = 0; }
  }

  void loop() {
    try {
      c10::InferenceMode inference;   // (the graphs' pinned buffers were made under inference mode: writes to them need it)
      auto stream = c10::cuda::CUDAStream::unpack3(stream_id, (c10::DeviceIndex)device_index, c10::DeviceType::CUDA);
      c10::cuda::CUDAStreamGuard guard(stream);
      std::vector<struct pollfd> pfds(fds.size());
      std::vector<i64> ws, ss;   // the batch: workers and row slots
      std::vector<i64> held_w, held_s;   // (B54) messages waiting for their worker's CF capture (cap_wait)
      drv_tid = std::this_thread::get_id();   // (d2 M1: merge_q / LaneCore belong to this thread)
      double t_it = now();
      while (!stop_req) {
        { double t = now(); t_loop_s = t_loop_s+(t-t_it); t_it = t; }
        if (pause_req) { wait_paused(); continue; }
        if (test_fail) throw std::runtime_error("set_test_fail: a driver error (B58e test)");
        bool lanes_waiting = lane_queue_size() > 0;
        if (main_hold) {   // (B54) lanes only
          if (lanes_native && !lrunning) { lane_pass(3e-4); m1_flush(3); continue; }   // (B58d d1: the native lane service here)
          if (lanes_waiting) run_lanes();
          else { double t = now(); std::this_thread::sleep_for(std::chrono::microseconds(300)); t_hold_s = t_hold_s+(now()-t); }
          continue;
        }
        if (ready_w.empty()) {
          double t_p = now();
          ws.clear(); ss.clear();
          for (size_t q = 0; q < held_w.size();) {   // held messages whose capture is done join the batch
            if (!cap_wait[held_w[q]]) { ws.push_back(held_w[q]); ss.push_back(held_s[q]); held_w.erase(held_w.begin()+q); held_s.erase(held_s.begin()+q); }
            else q++;
          }
          for (size_t i = 0; i < fds.size(); i++) { pfds[i].fd = fds[i]; pfds[i].events = fds[i] >= 0 ? POLLIN : 0; pfds[i].revents = 0; }
          int k = ::poll(pfds.data(), (nfds_t)pfds.size(), (lanes_waiting || !ws.empty() || !merge_q.empty()) ? 0 : 1);
          for (size_t i = 0; k > 0 && i < fds.size(); i++) {
            if (!(pfds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            unsigned char msg[5]; ssize_t got = ::read(fds[i], msg, 5);
            if (got == 5 && msg[0] == 0 && msg[1] == 0 && msg[2] == 0 && msg[3] == 1 && (msg[4] == '0' || msg[4] == '1')) {
              i64 slot = 2*(i64)i+(msg[4] == '1');
              if (cap_wait[i]) { held_w.push_back((i64)i); held_s.push_back(slot); n_cap_holds++; }
              else { ws.push_back((i64)i); ss.push_back(slot); }
            } else {
              fds[i] = -1;   // the worker ended (or something unexpected): its pipe leaves the poll; Python notices through its own bookkeeping
            }
          }
          t_poll_s = t_poll_s+(now()-t_p); n_polls++;
          if (ws.empty()) {
            n_idle_polls++;
            if (lanes_waiting) run_lanes();
            if (lanes_native && !lrunning) { m1_flush(3); lane_pass(0); m1_flush(3); }   // (B58d d1; M1: no main rows now)
            continue;
          }
        } else { ws = ready_w; ss = ready_s; ready_w.clear(); ready_s.clear(); }
        in_step = true; step_batch(ws, ss); in_step = false;
        if (!pause_req) run_lanes();   // (the other generations' jobs after the main reply, as the Python path)
        if (lanes_native && !lrunning && !pause_req) lane_pass(0);   // (B58d d1: a lane pass after every main step)
      }
    } catch (const std::exception &ex) {
      error = ex.what();
      std::lock_guard<std::mutex> g(cmu); paused = true; ccv.notify_all();
      auto r = std::make_shared<Record>(); r->kind = 3; finish_empty(*r); push_record(r);
    }
    running = false;
  }

  // (B58a+b) Episodes.arrive over the record's rows, then store over its go rows, into the current rollout slot (abpc)
  void consume(Record &r, const uint8_t *rc, i64 n_all, const std::vector<i64> &ws, const std::vector<uint8_t> &go,
               const std::vector<i64> &go_idx, const float *out, const i64 *act, const i64 *pend, i64 ver, const uint8_t *privp) {
    (void)go;
    int sl = cur_slot.load();
    i64 n = (i64)go_idx.size();
    r.kk = at::full({n}, -1, at::kLong);
    r.snap = at::zeros({n_all, N_SNAP}, at::kDouble);
    if (!cc.on || sl < 0 || !slots[sl].set) return;
    abpc::RowLayout L{row_stride, f_bytes, f_frame, frame_bytes, nh, priv_itemsize};
    abpc::consume(cc, slots[sl], amu, L, rc, n_all, ws.data(), go_idx.data(), n, out, act, pend, ver, privp, r.kk.data_ptr<i64>(),
                  r.snap.data_ptr<double>());
  }
  // a record's frame header: (body, entity, inventory, slot) counts; false when the frame is not a valid visible-v2 frame
  bool frame_counts(const uint8_t *src, i64 &p, i64 &e, i64 &nn, i64 &s) {
    const float *f = (const float *)(src+f_frame);
    i64 schema = read_int(src+f_schema, 4), size = read_int(src+f_bytes, 4);
    p = (i64)f[2]; e = (i64)f[3]; nn = (i64)f[4]; s = (i64)f[5];
    i64 expected = frame_fixed+p*(80+324)+e*48+nn*4+s*13;
    return !(schema != schema_version || f[0] != (float)magic || f[1] != (float)expected || size != expected*4 || p <= 0 || e > 128 ||
             nn < 0 || s < 0 || size > frame_bytes);
  }
  // widths of rows [0, n) of a contiguous batch; false when a frame is invalid or the batch too wide for a graph
  bool batch_widths(const uint8_t *rows, i64 n, i64 &wb, i64 &we, i64 &wn, i64 &wsl) {
    i64 pm = 0, em = 0, nm = 0, sm = 0; bool valid = true;
    for (i64 j = 0; j < n; j++) {
      i64 p, e, nn, s;
      valid &= frame_counts(rows+j*row_stride, p, e, nn, s);
      pm = std::max(pm, p); em = std::max(em, e); nm = std::max(nm, nn); sm = std::max(sm, s);
    }
    wb = bucket(bk_body, pm); we = bucket(bk_entity, em); wn = bucket(bk_inventory, nm); wsl = bucket(bk_slots, sm);
    bool too_wide = wb < 0 || we < 0 || wn < 0 || wsl < 0 || 30+2*wb+we+wn+wsl > 1024;
    return valid && !too_wide;
  }
  GraphEntry *find_graph(const std::vector<i64> &key) {
    std::lock_guard<std::mutex> g(gmu); auto it = graphs.find(key); return it == graphs.end() ? nullptr : &it->second;
  }
  // the pages of batch rows [o, o + k): decisions (K/V, z, raw frame), lineage ring, completed blocks, step counters;
  // the record's dec / kvs / bid
  void write_part(GraphEntry &e, const uint8_t *rows, i64 o, i64 k, const std::vector<i64> &ids, const std::vector<i64> &steps_before,
                  const std::vector<uint8_t> &complete, const std::vector<uint8_t> &first, const i64 *pend, i64 ver, Record &r) {
    r.dec = at::empty({k}, at::kLong); r.kvs = at::empty({k}, at::kLong); r.bid = at::full({k}, -1, at::kLong);
    if (k == 0) return;
    i64 kv_bytes = store->kv_elems*store->kv_itemsize;
    {
      std::vector<const uint8_t *> fr(k); std::vector<i64> nb(k), st(k); std::vector<uint8_t> fi(k); std::vector<int32_t> sc(k);
      for (i64 j = 0; j < k; j++) {
        const uint8_t *src = rows+(o+j)*row_stride;
        fr[j] = src+f_frame; nb[j] = read_int(src+f_bytes, 4); st[j] = steps_before[o+j]; fi[j] = first[o+j]; sc[j] = (int32_t)read_int(src+f_schema, 4);
      }
      store->write_decisions_raw(k, (const uint8_t *)e.bc.data_ptr()+o*kv_bytes, kv_bytes, e.bz.data_ptr<float>()+o*store->z_elems, store->z_elems,
                                 st.data(), ver, fi.data(), sc.data(), fr.data(), nb.data(), pend, r.dec.data_ptr<i64>(), r.kvs.data_ptr<i64>());
    }
    i64 *dec = r.dec.data_ptr<i64>(), *kvs = r.kvs.data_ptr<i64>();
    const i64 *dmeta = store->d_meta.data_ptr<i64>();
    std::vector<i64> ends, src, srcver, done_rows;
    for (i64 j = 0; j < k; j++) {
      i64 row = ids[o+j];
      i64 *rd = &ring_dec[row*block_steps], *rk = &ring_kv[row*block_steps];
      i64 pos = steps_before[o+j] % block_steps;
      rd[pos] = dec[j]; rk[pos] = kvs[j]; ring_n[row] = std::min<i64>(ring_n[row]+1, block_steps);
      if (complete[o+j]) {   // the block's block_steps decisions end at this step: positions 0..bs-1 in step order
        done_rows.push_back(j); ends.push_back(steps_before[o+j]);
        for (i64 q = 0; q < block_steps; q++) { src.push_back(rd[q]); srcver.push_back(rd[q] >= 0 ? dmeta[2*rk[q]+1] : -1); }
      }
      step_host[row] += 1;
    }
    if (!done_rows.empty()) {
      i64 m = (i64)done_rows.size();
      std::vector<float> vals(m*store->b_elems);
      const float *bw = e.bw.data_ptr<float>();   // (every row of the graph; the complete rows in row order)
      for (i64 q = 0; q < m; q++) memcpy(&vals[q*store->b_elems], bw+(o+done_rows[q])*store->b_elems, store->b_elems*sizeof(float));
      std::vector<i64> bid(m);
      store->write_blocks_raw(m, vals.data(), store->b_elems, ends.data(), ver, src.data(), srcver.data(), bid.data());
      for (i64 q = 0; q < m; q++) r.bid.data_ptr<i64>()[done_rows[q]] = bid[q];
    }
  }
  // the lane rows' uniforms: 0.5 on replayed rows, else the CRN of (request sequence, lane)
  void lane_uniforms(const LaneJob &jb, float *u) {
    for (i64 j = 0; j < jb.n; j++) {
      if (jb.replay[j]) { for (i64 c = 0; c < heads; c++) u[j*heads+c] = .5f; }
      else crn_row(jb.seq[j], jb.lane[j], heads, u+j*heads);
    }
  }
  // a lane record's per-row fields from batch rows [o, o + k)
  void lane_record(Record &r, const LaneJob &jb, GraphEntry &e, i64 o, const std::vector<i64> &steps_before, const std::vector<uint8_t> &first,
                   const std::vector<uint8_t> &complete, i64 ver) {
    i64 k = jb.n, ow = nh+3;
    r.kind = 4; r.job = jb.job; r.owner = jb.owner; r.version = ver;
    r.out = at::empty({k, ow}, at::kFloat); memcpy(r.out.data_ptr<float>(), e.bo.data_ptr<float>()+o*ow, k*ow*sizeof(float));
    r.actions = at::empty({k, nh}, at::kLong);
    for (i64 j = 0; j < k; j++) for (i64 h = 0; h < nh; h++) r.actions.data_ptr<i64>()[j*nh+h] = (i64)r.out.data_ptr<float>()[j*ow+h];
    r.pending_before = at::empty({k, nh}, at::kLong); memcpy(r.pending_before.data_ptr<i64>(), jb.pend.data(), k*nh*8);
    r.step_before = at::empty({k}, at::kLong); memcpy(r.step_before.data_ptr<i64>(), &steps_before[o], k*8);
    r.first = at::empty({k}, at::kByte); memcpy(r.first.data_ptr<uint8_t>(), &first[o], k);
    r.complete = at::empty({k}, at::kByte); memcpy(r.complete.data_ptr<uint8_t>(), &complete[o], k);
    r.rows = at::empty({0}, at::kByte); r.priv = at::empty({0}, at::kByte);
  }

  // (B58c) observe_rows over a record's rows (every row, terminal ones too)
  void lineage_observe(const uint8_t *rc, i64 n_all, const std::vector<i64> &ws) {
    abpc::Native &N = *store->nat;
    std::lock_guard<std::mutex> l(N.lin.mu);
    for (i64 i = 0; i < n_all; i++) {
      const uint8_t *src = rc+i*row_stride;
      float hurt; memcpy(&hurt, src+f_hurt, 4);
      N.inj.observe(ws[i], read_int(src+f_first, s_first) != 0, read_int(src+f_episode, s_episode), read_int(src+f_t, s_t), hurt > 0);
    }
  }
  // (B58c) the go rows' lineage, as consume_record did it: a new log at an episode's first row, the snapshot before
  // the step (PagedSnapshot(store, log, step_before, pending_before, version)), the CF captures it makes, the injury
  // source, the step's decision / K/V row / block into the log, the sidecar's chunk start / open decision (the native
  // consumer's rollout index); captures whose generation Python must publish first become kind 6 records (pubs)
  void lineage_main(Record &r, const uint8_t *rc, const std::vector<i64> &ws, const std::vector<i64> &go_idx, const std::vector<i64> &ids,
                    const std::vector<uint8_t> &first, const std::vector<i64> &steps_before, const i64 *pend, i64 ver,
                    const std::vector<uint8_t> &complete, std::vector<std::shared_ptr<Record>> &pubs) {
    abpc::Native &N = *store->nat;
    abpc::Lineage &L = N.lin;
    i64 n = (i64)go_idx.size();
    const i64 *dec = r.dec.data_ptr<i64>(), *kvs = r.kvs.data_ptr<i64>(), *bid = r.bid.data_ptr<i64>();
    const i64 *kk = r.kk.defined() && r.kk.numel() == n ? r.kk.data_ptr<i64>() : nullptr;
    int sl = cur_slot.load();
    std::lock_guard<std::mutex> l(L.mu);
    std::vector<i64> sn(n), ep(n), tt(n);
    for (i64 j = 0; j < n; j++) {
      i64 row = ids[j], w = ws[go_idx[j]];
      const uint8_t *src = rc+go_idx[j]*row_stride;
      ep[j] = read_int(src+f_episode, s_episode); tt[j] = read_int(src+f_t, s_t);
      if (first[j]) L.set_head_locked(row, L.new_log_locked(lin_tag, w, ep[j], 0));
      i64 g = L.head.at(row);
      if (g < 0) throw std::runtime_error("lineage: a worker's row without a log (its episode began before the native lineage)");
      if (L.logs[g].end() != steps_before[j]) throw std::runtime_error("lineage: a row's log end differs from its step");
      sn[j] = L.snap_row_locked(row, pend+j*nh, ver);
    }
    for (i64 j = 0; j < n; j++) {   // capture_deferred
      i64 req = r.cf_snap[go_idx[j]];
      if (!req) continue;
      i64 point = req-1;
      i64 c = N.pts.capture(L, point, sn[j], ver);
      if (c == 2 || c == -1) {
        if (N.status) N.status[point] = c;
        (c == 2 ? n_cap_native : n_cap_refused)++;
      } else {   // the generation's weights first (Python): the worker's next record waits for it
        auto pr = std::make_shared<Record>(); pr->kind = 6; pr->point = point; pr->pub_slot = -2-c; pr->pub_snap = sn[j];
        pr->pub_worker = ws[go_idx[j]]; pr->version = ver; finish_empty(*pr);
        L.retain_locked(sn[j]);
        cap_wait[ws[go_idx[j]]] = 1; n_cap_publish++;
        pubs.push_back(pr);
      }
    }
    for (i64 j = 0; j < n; j++) N.inj.record(L, ws[go_idx[j]], ep[j], tt[j], sn[j]);   // record_main
    for (i64 j = 0; j < n; j++) {
      L.push_locked(ids[j], dec[j], kvs[j]);
      if (complete[j]) L.push_block_locked(ids[j], bid[j], steps_before[j]);
    }
    if (kk && sl >= 0)
      for (i64 j = 0; j < n; j++) if (kk[j] >= 0) L.side_record_locked(sl, ws[go_idx[j]], kk[j], first[j] != 0, sn[j]);
    for (i64 j = 0; j < n; j++) L.release_locked(sn[j]);
    L.published = record_seq.load();   // (B59: a GC while the driver runs keeps rows stamped after this)
  }
  std::atomic<i64> record_seq{0};   // (B58c) every step's rows carry its sequence (PageCore stamp; B59's GC rule)
  void step_batch(const std::vector<i64> &ws, const std::vector<i64> &ss) {
    double t0 = now();
    store->set_stamp(++record_seq);
    i64 n_all = (i64)ws.size();
    auto r = std::make_shared<Record>();
    r->kind = 0; r->owner = owner_main; r->workers = ws; r->slots = ss; r->go.assign(n_all, 0); r->cf_snap.assign(n_all, 0); r->t_poll = t0;
    // the records: header + used frame bytes copied (the worker may rewrite this slot after two more replies)
    r->rows = at::empty({n_all*row_stride}, at::kByte);
    uint8_t *rc = r->rows.data_ptr<uint8_t>();
    if (priv) r->priv = at::empty({n_all*priv_itemsize}, at::kByte); else r->priv = at::empty({0}, at::kByte);
    std::vector<i64> go_idx; go_idx.reserve(n_all);
    for (i64 i = 0; i < n_all; i++) {
      const uint8_t *src = rows_base+ss[i]*row_stride;
      i64 nbytes = read_int(src+f_bytes, 4);
      i64 used = f_frame+std::max<i64>(0, std::min<i64>(nbytes, frame_bytes));
      memcpy(rc+i*row_stride, src, (size_t)std::min<i64>(used, row_stride));
      if (priv) memcpy(r->priv.data_ptr<uint8_t>()+i*priv_itemsize, priv+ss[i]*priv_itemsize, priv_itemsize);
      i64 done = read_int(src+f_done, s_done), first = read_int(src+f_first, s_first);
      bool ended = done != 0, boot = done == done_branch;
      bool go = !ended || boot;
      r->go[i] = go;
      if (cf_snap) { r->cf_snap[i] = cf_snap[ws[i]]; if (r->cf_snap[i] && !lin_on) cap_wait[ws[i]] = 1; }   // (before the reply;
      // B58c: the driver captures natively below, a worker waits only for a generation Python must publish)
      if (ended && !boot) memset(&pending_cur[ws[i]*nh], 0, nh*8);   // Episodes.arrive: pending[idx[ended & ~boot]] = 0
      if (first && go) memset(&pending_cur[ws[i]*nh], 0, nh*8);      //                  pending[idx[first & go]] = 0
      if (go) go_idx.push_back(i); else n_terminal++;
    }
    i64 n = (i64)go_idx.size();
    if (n == 0) {   // terminal records only: reply, hand the bookkeeping over
      for (i64 i = 0; i < n_all; i++) if (cf_snap && r->cf_snap[i]) cf_snap[ws[i]] = 0;
      double tq = now(); t_prep_s = t_prep_s+(tq-t0);
      reply_rows(ws);
      r->t_sync = r->t_reply = r->t_pages = r->t_done = now();
      t_reply_s = t_reply_s+(r->t_reply-tq);
      finish_empty(*r);
      if (cc.on) consume(*r, rc, n_all, ws, r->go, go_idx, nullptr, nullptr, nullptr, version.load(),
                         priv ? r->priv.data_ptr<uint8_t>() : nullptr);   // (B58a+b: the episode ends)
      t_cons_s = t_cons_s+(now()-r->t_reply);
      if (lin_on) lineage_observe(rc, n_all, ws);
      deliver(r, rc, n_all, ws, 0); n_calls++;   // (B58e: batched in batch mode)
      return;
    }
    // the batch's rows, contiguous: the go rows of the copied records, then (B54) the merged lane jobs' rows
    std::vector<std::shared_ptr<LaneJob>> merged;
    if (lane_merge) {
      std::lock_guard<std::mutex> g(lmu);
      for (auto it = lane_q.begin(); it != lane_q.end();) {
        if ((*it)->version == version.load()) { merged.push_back(*it); it = lane_q.erase(it); } else ++it;
      }
    }
    // (d2 M1) the native jobs waiting for this batch: the version checked again (a job of a version taken since goes
    // lane-only), at most m1_cap rows in all; the others run lane-only after this step's reply (later)
    std::vector<MergeJob> nat_m;
    std::vector<std::pair<MergeJob, int>> later;
    if (!merge_q.empty()) {
      m1_thread_check("step_batch");
      i64 v = version.load(), tot = n;
      while (!merge_q.empty()) {
        MergeJob m = merge_q.front(); merge_q.pop_front();
        if (m.jb->version != v) later.emplace_back(m, 1);
        else if (tot+m.jb->n > m1_cap || (!merged.empty() && nat_m.empty())) later.emplace_back(m, 2);
        else { tot += m.jb->n; nat_m.push_back(m); }
      }
      for (auto &m : nat_m) merged.push_back(m.jb);
    }
    auto run_later = [&]() {   // (the same pass: after the main reply, or before handing the batch to Python)
      for (auto &x : later) { lane_run(*x.first.jb, x.first.g, x.second); gen_pin(x.first.g, -1); }
      later.clear();
    };
    i64 n_lane = 0; for (auto &jb : merged) n_lane += jb->n;
    std::vector<uint8_t> packed;
    const uint8_t *rows_ptr;
    auto pack = [&](bool with_lanes) {
      i64 tot = n+(with_lanes ? n_lane : 0);
      if (tot == n_all && n == n_all) { rows_ptr = rc; return; }
      packed.resize(tot*row_stride);
      for (i64 j = 0; j < n; j++) memcpy(&packed[j*row_stride], rc+go_idx[j]*row_stride, row_stride);
      if (with_lanes) { i64 o = n; for (auto &jb : merged) { memcpy(&packed[o*row_stride], jb->rows.data(), jb->n*row_stride); o += jb->n; } }
      rows_ptr = packed.data();
    };
    auto put_back = [&]() {   // (the merged jobs return to the front of the queue, in order; M1's: lane-only, refused)
      if (!nat_m.empty()) {
        for (auto &m : nat_m) later.emplace_back(m, 2);
        nat_m.clear(); merged.clear(); n_lane = 0;
        return;
      }
      std::lock_guard<std::mutex> g(lmu);
      for (auto it = merged.rbegin(); it != merged.rend(); ++it) lane_q.push_front(*it);
      merged.clear(); n_lane = 0;
    };
    pack(!merged.empty());
    i64 wb, we, wn, wsl;
    bool ok = batch_widths(rows_ptr, n+n_lane, wb, we, wn, wsl);
    if (!ok && !merged.empty()) {   // the lanes would make the batch invalid / too wide: main rows alone
      n_putback_rows = n_putback_rows+n_lane;
      put_back(); pack(false); ok = batch_widths(rows_ptr, n, wb, we, wn, wsl);
    }
    i64 n_tot = n+n_lane;
    i64 cap = 1; while (cap < n_tot) cap <<= 1;
    if (!ok) {   // Python handles this batch (it raises the same error, or takes the eager path)
      t_prep_s = t_prep_s+(now()-t0);
      run_later();
      r->kind = 2; r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl;
      hand_to_python(r, ws, ss);
      return;
    }
    std::vector<i64> key{owner_main, cap, wb, we, wn, wsl};
    GraphEntry *e = find_graph(key);
    if (!e && !nat_m.empty() && m1_test_capture) {   // (tests: the merged graph captured, the batch and the jobs retried)
      t_prep_s = t_prep_s+(now()-t0);
      for (auto it = nat_m.rbegin(); it != nat_m.rend(); ++it) merge_q.push_front(*it);
      nat_m.clear(); merged.clear(); n_lane = 0;
      run_later();
      r->kind = 1; r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl;
      n_captures++;
      hand_to_python(r, ws, ss);
      return;
    }
    if (!e && !nat_m.empty()) {   // (M1: no merged graph of these widths: the main rows alone, the jobs lane-only)
      put_back(); pack(false);
      ok = batch_widths(rows_ptr, n, wb, we, wn, wsl);
      n_tot = n; cap = 1; while (cap < n_tot) cap <<= 1;
      if (!ok) {
        t_prep_s = t_prep_s+(now()-t0);
        run_later();
        r->kind = 2; r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl;
        hand_to_python(r, ws, ss);
        return;
      }
      key = {owner_main, cap, wb, we, wn, wsl};
      e = find_graph(key);
    }
    if (!e) {   // not captured yet: Python captures (needs the GIL and the capture admission); the batch is retried
      t_prep_s = t_prep_s+(now()-t0);
      put_back();
      run_later();
      r->kind = 1; r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl;
      n_captures++;
      hand_to_python(r, ws, ss);
      return;
    }
    // the step's inputs: main rows [0, n), lane rows [n, n_tot)
    std::vector<i64> ids(n_tot), pend(n_tot*nh), steps_before(n_tot); std::vector<uint8_t> replay(n_tot, 0), complete(n_tot), first(n_tot);
    std::vector<float> u(n_tot*heads);
    std::vector<i64> to_reset;
    for (i64 j = 0; j < n; j++) {
      i64 w = ws[go_idx[j]];
      ids[j] = base+w;
      memcpy(&pend[j*nh], &pending_cur[w*nh], nh*8);
    }
    { i64 o = n; for (auto &jb : merged) { for (i64 q = 0; q < jb->n; q++) { ids[o+q] = jb->ids[q]; replay[o+q] = jb->replay[q]; }
        memcpy(&pend[o*nh], jb->pend.data(), jb->n*nh*8); o += jb->n; } }
    for (i64 j = 0; j < n_tot; j++) {
      first[j] = read_int(rows_ptr+j*row_stride+f_first, s_first) != 0;
      if (first[j]) to_reset.push_back(ids[j]);
    }
    if (!to_reset.empty()) reset_rows(to_reset);   // (device ops on the actor stream; before the graph)
    bool any_complete = false;
    for (i64 j = 0; j < n_tot; j++) {
      steps_before[j] = step_host[ids[j]];
      complete[j] = (steps_before[j]+1) % block_steps == 0; any_complete |= complete[j];
    }
    for (i64 x = 0; x < n*heads; x++) u[x] = fixed_u >= 0 ? (float)fixed_u : rng.uniform();
    { i64 o = n; for (auto &jb : merged) { lane_uniforms(*jb, &u[o*heads]); o += jb->n; } }
    double t_host0 = now(); t_prep_s = t_prep_s+(t_host0-t0);
    i64 ver = version.load();
    fill_and_run(*e, n_tot, rows_ptr, row_stride, pend.data(), ids.data(), u.data(), replay.data(), complete.data(), ver, any_complete);
    double t1 = now(); r->t_sync = t1; t_sync_s = t_sync_s+(t1-t_host0);
    // main outputs -> actions (pforce), shared arrays, reply
    const float *bo = e->bo.data_ptr<float>(); i64 ow = nh+3;   // (the step's outputs: actions [nh], logp, value, vg)
    r->out = at::empty({n, ow}, at::kFloat); memcpy(r->out.data_ptr<float>(), bo, n*ow*sizeof(float));
    r->actions = at::empty({n, nh}, at::kLong); i64 *act = r->actions.data_ptr<i64>();
    r->pending_before = at::empty({n, nh}, at::kLong); memcpy(r->pending_before.data_ptr<i64>(), pend.data(), n*nh*8);
    r->step_before = at::empty({n}, at::kLong); memcpy(r->step_before.data_ptr<i64>(), steps_before.data(), n*8);
    r->first = at::empty({n}, at::kByte); memcpy(r->first.data_ptr<uint8_t>(), first.data(), n);
    r->complete = at::empty({n}, at::kByte); memcpy(r->complete.data_ptr<uint8_t>(), complete.data(), n);
    r->version = ver;
    for (i64 j = 0; j < n; j++) {
      i64 i = go_idx[j], w = ws[i];
      const uint8_t *src = rows_base+ss[i]*row_stride;
      for (i64 h = 0; h < nh; h++) act[j*nh+h] = (i64)bo[j*ow+h];
      if (pf_on) { i64 fm = read_int(src+f_pforce, s_pforce); if (fm) { act[j*nh] = fm; if (nh > 1) act[j*nh+1] = 0; } }
      double value = bo[j*ow+nh+1], logp = bo[j*ow+nh], vg = bo[j*ow+nh+2];
      for (i64 h = 0; h < nh; h++) actions[w*actions_stride+h] = (int32_t)act[j*nh+h];
      if (main_value) main_value[w] = value;
      if (main_vg) main_vg[w] = vg;
      if (aux) { double *a = aux+w*aux_stride; a[0] = value; a[1] = logp; a[2] = 0; a[3] = 0; a[5] = (double)r->version; }
      memcpy(&pending_cur[w*nh], &act[j*nh], nh*8);
    }
    for (i64 i = 0; i < n_all; i++) if (cf_snap && r->cf_snap[i]) cf_snap[ws[i]] = 0;
    reply_rows(ws);
    double t2 = now(); r->t_reply = t2; t_reply_s = t_reply_s+(t2-t1);
    // the main rows' pages, then their record (the lanes' pages and records only after it: B54 condition a)
    write_part(*e, rows_ptr, 0, n, ids, steps_before, complete, first, pend.data(), ver, *r);
    double t2c = now(); t_pages_s = t_pages_s+(t2c-t2);
    if (cc.on) consume(*r, rc, n_all, ws, r->go, go_idx, r->out.data_ptr<float>(), act, pend.data(), ver,
                       priv ? r->priv.data_ptr<uint8_t>() : nullptr);   // (B58a+b: arrive + store, before the record)
    double t3 = now(); r->t_pages = t3; r->t_done = t3; t_cons_s = t_cons_s+(t3-t2c); t_host_s = t_host_s+(t3-t0)-(t1-t_host0);
    n_steps++; n_calls++; n_rows = n_rows+n;
    std::vector<std::shared_ptr<Record>> pubs;
    if (lin_on) {   // (B58c: before the record: Python's consume_record did this between arrive and the store)
      lineage_observe(rc, n_all, ws);
      lineage_main(*r, rc, ws, go_idx, ids, first, steps_before, pend.data(), ver, complete, pubs);
      t_lin_s = t_lin_s+(now()-t3);
    }
    deliver(r, rc, n_all, ws, n);   // (B58e: batched in batch mode)
    for (auto &pr : pubs) push_record(pr);
    double t4 = now();
    if (!nat_m.empty()) {   // (d2 M1: the lane rows of this replay: their pages, their lineage, their answers)
      n_merged_calls++;
      i64 o = n;
      for (auto &m : nat_m) { m1_done(m, *e, rows_ptr, o, ids, steps_before, first, complete, ver, t0); o += m.jb->n; }
      t_mlane_s = t_mlane_s+(now()-t4);
      merged.clear();
    }
    run_later();
    if (!merged.empty()) {
      n_merged_calls++;
      i64 o = n;
      for (auto &jb : merged) {
        auto lr = std::make_shared<Record>();
        lr->t_poll = t0; lr->t_sync = t1; lr->t_main_reply = t2; lr->merged = 1;
        lane_record(*lr, *jb, *e, o, steps_before, first, complete, ver);
        lr->t_reply = now();
        write_part(*e, rows_ptr, o, jb->n, ids, steps_before, complete, first, jb->pend.data(), ver, *lr);
        lr->t_pages = lr->t_done = now();
        n_lane_jobs++; n_lane_rows = n_lane_rows+jb->n; n_merged_rows = n_merged_rows+jb->n;
        push_record(lr);
        o += jb->n;
      }
      t_mlane_s = t_mlane_s+(now()-t4);
    }
  }
  // (B54) the queued lane jobs, each a lane-only call through its generation's graph, until the queue is empty or the
  // driver paused itself (a capture / a job Python must step)
  void run_lanes() {
    while (!pause_req && !stop_req) {
      std::shared_ptr<LaneJob> jb;
      { std::lock_guard<std::mutex> g(lmu); if (lane_q.empty()) return; jb = lane_q.front(); }
      if (!run_lane_job(*jb)) return;   // (the job stays at the front: retried after resume, or dropped by Python)
      std::lock_guard<std::mutex> g(lmu);
      if (!lane_q.empty() && lane_q.front() == jb) lane_q.pop_front();
    }
  }
  bool run_lane_job(LaneJob &jb) {
    double t0 = now();
    i64 k = jb.n;
    ((jb.version == version.load()) ? n_lo_cur_rows : n_lo_old_rows) += k;
    const uint8_t *rows_ptr = jb.rows.data();
    i64 wb, we, wn, wsl;
    bool ok = batch_widths(rows_ptr, k, wb, we, wn, wsl);
    i64 cap = 1; while (cap < k) cap <<= 1;
    auto pause_with = [&](int kind) {
      auto r = std::make_shared<Record>(); r->kind = kind; r->job = jb.job; r->owner = jb.owner; r->version = jb.version;
      r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl; finish_empty(*r);
      pause_req = true; push_record(r);
    };
    if (!ok) { n_lane_python++; pause_with(5); return false; }
    std::vector<i64> key{jb.owner, cap, wb, we, wn, wsl};
    GraphEntry *e = find_graph(key);
    if (!e) { n_captures++; pause_with(1); return false; }
    std::vector<i64> steps_before(k); std::vector<uint8_t> complete(k), first(k);
    std::vector<i64> to_reset;
    for (i64 j = 0; j < k; j++) {
      first[j] = read_int(rows_ptr+j*row_stride+f_first, s_first) != 0;
      if (first[j]) to_reset.push_back(jb.ids[j]);
    }
    if (!to_reset.empty()) reset_rows(to_reset);
    bool any_complete = false;
    for (i64 j = 0; j < k; j++) {
      steps_before[j] = step_host[jb.ids[j]];
      complete[j] = (steps_before[j]+1) % block_steps == 0; any_complete |= complete[j];
    }
    std::vector<float> u(k*heads);
    lane_uniforms(jb, u.data());
    fill_and_run(*e, k, rows_ptr, row_stride, jb.pend.data(), jb.ids.data(), u.data(), jb.replay.data(), complete.data(), jb.version, any_complete);
    auto r = std::make_shared<Record>();
    r->t_poll = t0; r->t_sync = now();
    lane_record(*r, jb, *e, 0, steps_before, first, complete, jb.version);
    r->t_reply = now();
    write_part(*e, rows_ptr, 0, k, jb.ids, steps_before, complete, first, jb.pend.data(), jb.version, *r);
    r->t_pages = r->t_done = now();
    n_lane_jobs++; n_lane_calls++; n_lane_rows = n_lane_rows+k;
    push_record(r);
    lane_hist[k <= 1 ? 0 : k <= 2 ? 1 : k <= 4 ? 2 : k <= 8 ? 3 : k <= 16 ? 4 : 5]++;
    t_lane_s = t_lane_s+(now()-t0);
    return true;
  }
  void finish_empty(Record &r) {
    if (!r.rows.defined()) r.rows = at::empty({0}, at::kByte);
    if (!r.priv.defined()) r.priv = at::empty({0}, at::kByte);
    r.out = at::empty({0, nh+3}, at::kFloat); r.actions = at::empty({0, nh}, at::kLong); r.pending_before = at::empty({0, nh}, at::kLong);
    r.step_before = at::empty({0}, at::kLong); r.first = at::empty({0}, at::kByte); r.complete = at::empty({0}, at::kByte);
    r.dec = at::empty({0}, at::kLong); r.kvs = at::empty({0}, at::kLong); r.bid = at::empty({0}, at::kLong); r.version = version.load();
  }
  void hand_to_python(std::shared_ptr<Record> &r, const std::vector<i64> &ws, const std::vector<i64> &ss) {
    // the batch stays pending (ready_w / ready_s) and the driver pauses itself; Python resolves (captures / steps) and resumes
    finish_empty(*r);
    ready_w = ws; ready_s = ss;
    pause_req = true;
    push_record(r);
  }
  // after a kind-2 record Python stepped the batch itself: drop it from the retry list
  public:
  void drop_pending_batch() { ready_w.clear(); ready_s.clear(); }
  void reply_rows(const std::vector<i64> &ws) {
    for (i64 w : ws) {
      int fd = fds[w];
      if (fd < 0) continue;
      ssize_t k = ::write(fd, reply.data(), reply.size());
      (void)k;
    }
  }
  // the consumer-side step counter / ring maintenance after Python stepped a kind-2 batch itself is not supported:
  // such batches raise in Python (invalid frames) or take the eager path once; the driver's ring misses them only for
  // frames the Python path rejects as well.
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("visible_version", [] { return abp_visible_version(); });
  m.def("crn_uniforms", &crn_uniforms, py::arg("keys"), py::arg("t"), py::arg("columns"));
  py::class_<PageStore, std::shared_ptr<PageStore>>(m, "PageStore")
      .def(py::init<i64, i64, bool, i64, i64, i64, i64, i64, i64>(), py::arg("page"), py::arg("kv_elems"), py::arg("kv_bf16"), py::arg("z_elems"),
           py::arg("b_elems"), py::arg("block_steps"), py::arg("nh"), py::arg("max_pages_d"), py::arg("max_pages_b"))
      .def("kv_page", &PageStore::kv_page).def("z_page", &PageStore::z_page).def("b_page", &PageStore::b_page)
      .def("n_pages_d", &PageStore::n_pages_d).def("n_pages_b", &PageStore::n_pages_b)
      .def("n_free_d", &PageStore::n_free_d).def("n_free_b", &PageStore::n_free_b)
      .def("grow_ahead", &PageStore::grow_ahead)
      .def_property_readonly("d_meta", [](PageStore &s) { return s.d_meta; }).def_property_readonly("d_used", [](PageStore &s) { return s.d_used; })
      .def_property_readonly("b_meta", [](PageStore &s) { return s.b_meta; }).def_property_readonly("b_src", [](PageStore &s) { return s.b_src; })
      .def_property_readonly("b_srcver", [](PageStore &s) { return s.b_srcver; }).def_property_readonly("b_used", [](PageStore &s) { return s.b_used; })
      .def_property_readonly("cap_d", [](PageStore &s) { return s.cap_d; }).def_property_readonly("cap_b", [](PageStore &s) { return s.cap_b; })
      .def_property("next_decision", [](PageStore &s) { std::lock_guard<std::mutex> g(s.core->mu); return s.core->next_decision; },
                    [](PageStore &s, i64 v) { std::lock_guard<std::mutex> g(s.core->mu); s.core->next_decision = v; })
      .def("write_decisions", &PageStore::write_decisions).def("write_blocks", &PageStore::write_blocks)
      .def("raw_entry", &PageStore::raw_entry).def("raw_has", &PageStore::raw_has).def("raw_count", &PageStore::raw_count)
      .def("raw_keys", &PageStore::raw_keys).def("raw_mb", &PageStore::raw_mb).def("source_arena", &PageStore::source_arena)
      .def("free_rows", &PageStore::free_rows, py::arg("dead_d"), py::arg("dead_b"), py::arg("dead_raw"), py::arg("limit") = 0)
      .def("set_stamp", &PageStore::set_stamp).def("stats", &PageStore::stats)
      // (B58c) the native lineage
      .def("lineage_configure", &PageStore::lineage_configure, py::arg("window"), py::arg("block_steps"), py::arg("memory_blocks"),
           py::arg("nh"), py::arg("rows"), py::arg("workers"), py::arg("depths"), py::arg("chunk_steps"), py::arg("generations") = 2).def("lineage_status", &PageStore::lineage_status)
      .def("lineage_teach", &PageStore::lineage_teach)
      .def_property_readonly("lineage_on", [](PageStore &s) { return s.lineage_on; })
      .def("lin_new_head", &PageStore::lin_new_head).def("lin_head_end", &PageStore::lin_head_end).def("lin_snap_row", &PageStore::lin_snap_row)
      .def("lin_push", &PageStore::lin_push).def("lin_push_block", &PageStore::lin_push_block)
      .def("lin_retain", &PageStore::lin_retain).def("lin_release", &PageStore::lin_release).def("lin_restore", &PageStore::lin_restore)
      .def("lin_materialize", &PageStore::lin_materialize).def("lin_collect", &PageStore::lin_collect).def("lin_stats", &PageStore::lin_stats)
      .def("side_record", &PageStore::side_record).def("side_reset", &PageStore::side_reset).def("side_carry", &PageStore::side_carry)
      .def("side_payload", &PageStore::side_payload)
      .def("ctx_put", &PageStore::ctx_put).def("ctx_get", &PageStore::ctx_get).def("ctx_complete", &PageStore::ctx_complete)
      .def("ctx_retire", &PageStore::ctx_retire)
      .def("inj_observe", &PageStore::inj_observe).def("inj_record", &PageStore::inj_record).def("inj_get", &PageStore::inj_get)
      .def("inj_reclaim", &PageStore::inj_reclaim).def("inj_keys", &PageStore::inj_keys)
      .def("pt_capture", &PageStore::pt_capture).def("pt_published", &PageStore::pt_published).def("pt_release", &PageStore::pt_release)
      .def("pt_get", &PageStore::pt_get).def("pt_points", &PageStore::pt_points).def("pt_gens", &PageStore::pt_gens);
  py::class_<abpc::LaneCore>(m, "LaneWindowProbe")   // (tests) tok_lanes.window_due / LaneService.due over a LaneCore
      .def(py::init([](i64 lanes, double wait, double quiet, double active_s) {
        auto *c = new abpc::LaneCore(); abpc::LaneCfg cfg; cfg.lanes = lanes; cfg.wait = wait; cfg.quiet = quiet; cfg.active_s = active_s;
        c->cfg = cfg; c->seen.assign(lanes, -1e9); return c; }))
      .def("intake", &abpc::LaneCore::intake).def("due", &abpc::LaneCore::due).def("wait_left", &abpc::LaneCore::wait_left)
      .def("take", [](abpc::LaneCore &c, double now) {   // (LaneService._submit's take: distinct lanes, seen = now)
        std::set<i64> s(c.ready.begin(), c.ready.end()); c.ready.clear(); for (i64 l : s) c.seen[l] = now;
        return std::vector<i64>(s.begin(), s.end()); });
  py::class_<SnapPin, std::shared_ptr<SnapPin>>(m, "SnapPin")
      .def(py::init<std::shared_ptr<PageStore>, i64, bool>(), py::arg("store"), py::arg("snap"), py::arg("retain") = true)
      .def_readonly("snap", &SnapPin::snap);
  py::class_<ActorDriver, std::shared_ptr<ActorDriver>>(m, "ActorDriver")
      .def(py::init<std::shared_ptr<PageStore>>())
      .def("set_stream", &ActorDriver::set_stream).def("set_seed", &ActorDriver::set_seed).def("set_version", &ActorDriver::set_version)
      .def("set_fixed_uniform", &ActorDriver::set_fixed_uniform)
      .def("set_buckets", &ActorDriver::set_buckets).def("set_sampler", &ActorDriver::set_sampler).def("set_arena", &ActorDriver::set_arena)
      .def("set_pending", &ActorDriver::set_pending).def("get_pending", &ActorDriver::get_pending)
      .def("register_graph", &ActorDriver::register_graph).def("has_graph", &ActorDriver::has_graph).def("n_graphs", &ActorDriver::n_graphs)
      .def("run_graph", &ActorDriver::run_graph)
      .def("start", &ActorDriver::start).def("stop", &ActorDriver::stop).def("pause", &ActorDriver::pause, py::call_guard<py::gil_scoped_release>())
      .def("resume", &ActorDriver::resume).def("is_paused", &ActorDriver::is_paused).def("last_error", &ActorDriver::last_error)
      .def("next_record", &ActorDriver::next_record).def("queue_size", &ActorDriver::queue_size).def("counters", &ActorDriver::counters)
      .def("drop_pending_batch", &ActorDriver::drop_pending_batch)
      .def("submit_lanes", &ActorDriver::submit_lanes).def("lane_queue_size", &ActorDriver::lane_queue_size)
      .def("drop_lane_job", &ActorDriver::drop_lane_job).def("set_lane_merge", &ActorDriver::set_lane_merge)
      .def("set_ring", &ActorDriver::set_ring).def("get_ring", &ActorDriver::get_ring)
      .def("capture_done", &ActorDriver::capture_done).def("set_main_hold", &ActorDriver::set_main_hold)
      .def("set_lineage", &ActorDriver::set_lineage)
      .def("set_batch", &ActorDriver::set_batch).def("set_collect_target", &ActorDriver::set_collect_target)
      .def("set_busy", &ActorDriver::set_busy).def("wake", &ActorDriver::wake, py::call_guard<py::gil_scoped_release>())
      .def("set_test_fail", [](ActorDriver &d, bool on) { d.test_fail = on; })
      .def("set_m1_merge", [](ActorDriver &d, bool on) { d.m1_merge = on; }).def("set_m1_cap", [](ActorDriver &d, i64 c) { d.m1_cap = c; })
      .def("set_restore_test", [](ActorDriver &d, int m) { d.restore_test = m; })
      .def("set_m1_hold", [](ActorDriver &d, bool on) { d.m1_hold = on; }).def("set_m1_test_capture", [](ActorDriver &d, bool on) { d.m1_test_capture = on; }).def("take_events", &ActorDriver::take_events).def("batch_counters", &ActorDriver::batch_counters)
      .def("usable_now", &ActorDriver::usable_now)
      .def("lanes_configure", &ActorDriver::lanes_configure).def("lanes_start", &ActorDriver::lanes_start)
      .def("lanes_stop", &ActorDriver::lanes_stop, py::call_guard<py::gil_scoped_release>())
      .def("lanes_pause", &ActorDriver::lanes_pause, py::call_guard<py::gil_scoped_release>()).def("lanes_resume", &ActorDriver::lanes_resume)
      .def("lane_job_resume", &ActorDriver::lane_job_resume).def("lane_python_done", &ActorDriver::lane_python_done)
      .def("lane_test_intake", &ActorDriver::lane_test_intake)
      .def("lane_test_tick", &ActorDriver::lane_test_tick, py::call_guard<py::gil_scoped_release>())
      .def("lane_state", &ActorDriver::lane_state)
      .def("set_lane_force_python", [](ActorDriver &d, i64 g) { d.lane_force_python = g; })
      .def_property_readonly("version", [](ActorDriver &d) { return d.version.load(); })
      .def("set_lane_defer", [](ActorDriver &d, double s) { d.lc.cfg.defer_s = s; })
      .def("lane_timing", &ActorDriver::lane_timing).def("lane_counters", &ActorDriver::lane_counters).def("lane_error", [](ActorDriver &d) { return d.lerror; })
      .def_property_readonly("lanes_running", [](ActorDriver &d) { return d.lrunning.load(); })
      .def("set_consumer", &ActorDriver::set_consumer).def("set_rollout", &ActorDriver::set_rollout).def("use_rollout", &ActorDriver::use_rollout)
      .def("arena_buf", &ActorDriver::arena_buf).def("arena_used", &ActorDriver::arena_used).def("arena_reset", &ActorDriver::arena_reset)
      .def("arena_grown", &ActorDriver::arena_grown).def("arena_append_from", &ActorDriver::arena_append_from)
      .def("consume_rows", &ActorDriver::consume_rows)
      .def_property_readonly("consumer_on", [](ActorDriver &d) { return d.cc.on; })
      .def_property_readonly("running", [](ActorDriver &d) { return d.running.load(); });
}
