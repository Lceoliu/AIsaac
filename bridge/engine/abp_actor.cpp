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

struct RawEntry {
  uint8_t first;
  int32_t schema;
  std::string frame;
  std::vector<i64> pending;
};

struct PageStore {
  i64 page, kv_elems, z_elems, b_elems, bs, nh;
  int kv_itemsize;
  at::ScalarType kv_dtype;
  i64 cap_d, cap_b;
  std::vector<at::Tensor> kv_pages, z_pages, b_pages;
  at::Tensor d_meta, d_used, b_meta, b_src, b_srcver, b_used;
  std::vector<i64> d_free, b_free;
  i64 next_decision = 0;
  std::unordered_map<i64, RawEntry> raw;
  i64 freed_kv = 0, freed_blocks = 0, freed_raw = 0;
  std::mutex mu;   // allocation, free lists, page vectors, raw

  PageStore(i64 page_, i64 kv_elems_, bool kv_bf16, i64 z_elems_, i64 b_elems_, i64 block_steps, i64 nh_, i64 max_pages_d,
            i64 max_pages_b)
      : page(page_), kv_elems(kv_elems_), z_elems(z_elems_), b_elems(b_elems_), bs(block_steps), nh(nh_) {
    kv_dtype = kv_bf16 ? at::kBFloat16 : at::kFloat;
    kv_itemsize = kv_bf16 ? 2 : 4;
    cap_d = max_pages_d*page; cap_b = max_pages_b*page;
    auto l = at::TensorOptions().dtype(at::kLong);
    d_meta = at::full({cap_d, 2}, -1, l); d_used = at::zeros({cap_d}, at::kBool);
    b_meta = at::full({cap_b, 2}, -1, l); b_src = at::full({cap_b, bs}, -1, l); b_srcver = at::full({cap_b, bs}, -1, l);
    b_used = at::zeros({cap_b}, at::kBool);
    // (B58a) the page vectors never reallocate: a page tensor's pointer (and the vector element) stays where it is, so
    // other threads may read kv_pages[p] once p is published; reallocs counts any capacity change (must stay 0)
    kv_pages.reserve(max_pages_d); z_pages.reserve(max_pages_d); b_pages.reserve(max_pages_b);
  }
  i64 reallocs = 0;
  // (pages: kept as flat [page, elems] tensors; Python views them with the row shape)
  void grow_d_locked() {
    i64 p = (i64)kv_pages.size();
    TORCH_CHECK(p < cap_d/page, "PageStore: decision pages exhausted (raise max_pages_d)");
    size_t c0 = kv_pages.capacity(), c1 = z_pages.capacity();
    kv_pages.push_back(at::zeros({page, kv_elems}, at::TensorOptions().dtype(kv_dtype)));
    z_pages.push_back(at::zeros({page, z_elems}, at::kFloat));
    reallocs += (kv_pages.capacity() != c0)+(z_pages.capacity() != c1);
    i64 base = p*page;
    // Python: d_free = np.concatenate((np.arange(base+P-1, base-1, -1), d_free)): the new page's ids, descending,
    // in front of the old free ids; _alloc takes the array's last n ids reversed (our pops from the back)
    for (i64 i = page-1; i >= 0; i--) d_free.push_back(base+i);
    std::rotate(d_free.begin(), d_free.end()-page, d_free.end());
  }
  void grow_b_locked() {
    i64 p = (i64)b_pages.size();
    TORCH_CHECK(p < cap_b/page, "PageStore: block pages exhausted (raise max_pages_b)");
    size_t c2 = b_pages.capacity();
    b_pages.push_back(at::zeros({page, b_elems}, at::kFloat));
    reallocs += (b_pages.capacity() != c2);
    i64 base = p*page;
    for (i64 i = page-1; i >= 0; i--) b_free.push_back(base+i);
    std::rotate(b_free.begin(), b_free.end()-page, b_free.end());
  }
  // Python's _alloc: ids = pool[len-n:][::-1] (the n ids at the end of the free array, reversed), pool = pool[:-n].
  // Our vector mirrors the array (same order), so popping n from the back and reversing reproduces it.
  void alloc_locked(std::vector<i64> &free, bool decisions, i64 n, i64 *out) {
    while ((i64)free.size() < n) (decisions ? grow_d_locked() : grow_b_locked());
    bool *used = decisions ? d_used.data_ptr<bool>() : b_used.data_ptr<bool>();
    for (i64 i = 0; i < n; i++) {
      out[i] = free.back(); free.pop_back();
      used[out[i]] = true;
    }
  }
  void grow_ahead(i64 min_free_d, i64 min_free_b) {
    std::lock_guard<std::mutex> g(mu);
    while ((i64)d_free.size() < min_free_d) grow_d_locked();
    while ((i64)b_free.size() < min_free_b) grow_b_locked();
  }
  at::Tensor kv_page(i64 p) { std::lock_guard<std::mutex> g(mu); return kv_pages.at(p); }
  at::Tensor z_page(i64 p) { std::lock_guard<std::mutex> g(mu); return z_pages.at(p); }
  at::Tensor b_page(i64 p) { std::lock_guard<std::mutex> g(mu); return b_pages.at(p); }
  i64 n_pages_d() { std::lock_guard<std::mutex> g(mu); return (i64)kv_pages.size(); }
  i64 n_pages_b() { std::lock_guard<std::mutex> g(mu); return (i64)b_pages.size(); }
  i64 n_free_d() { std::lock_guard<std::mutex> g(mu); return (i64)d_free.size(); }
  i64 n_free_b() { std::lock_guard<std::mutex> g(mu); return (i64)b_free.size(); }

  // raw page pointers (the vectors only grow; a page tensor never moves)
  uint8_t *kv_row(i64 id) { return (uint8_t *)kv_pages[id/page].data_ptr()+(id % page)*kv_elems*kv_itemsize; }
  float *z_row(i64 id) { return z_pages[id/page].data_ptr<float>()+(id % page)*z_elems; }
  float *b_row(i64 id) { return b_pages[id/page].data_ptr<float>()+(id % page)*b_elems; }

  // the decisions of one step: kv rows (n x kv_elems of kv_dtype, contiguous), z rows, steps, frames (pointers), pending
  void write_decisions_raw(i64 n, const uint8_t *kv, i64 kv_row_bytes, const float *z, i64 z_row_floats, const i64 *steps,
                           i64 version, const uint8_t *first, const int32_t *schema, const uint8_t *const *frames,
                           const i64 *nbytes, const i64 *pending, i64 *dec_out, i64 *kvs_out) {
    std::lock_guard<std::mutex> g(mu);
    alloc_locked(d_free, true, n, kvs_out);
    for (i64 i = 0; i < n; i++) dec_out[i] = next_decision+i;
    next_decision += n;
    i64 *meta = d_meta.data_ptr<i64>();
    for (i64 i = 0; i < n; i++) {
      i64 id = kvs_out[i];
      memcpy(kv_row(id), kv+i*kv_row_bytes, kv_elems*kv_itemsize);
      memcpy(z_row(id), z+i*z_row_floats, z_elems*sizeof(float));
      meta[2*id] = steps[i]; meta[2*id+1] = version;
      RawEntry &e = raw[dec_out[i]];
      e.first = first[i]; e.schema = schema[i];
      e.frame.assign((const char *)frames[i], (size_t)nbytes[i]);
      e.pending.assign(pending+i*nh, pending+(i+1)*nh);
    }
  }
  void write_blocks_raw(i64 m, const float *values, i64 row_floats, const i64 *ends, i64 version, const i64 *src,
                        const i64 *srcver, i64 *bid_out) {
    std::lock_guard<std::mutex> g(mu);
    alloc_locked(b_free, false, m, bid_out);
    i64 *meta = b_meta.data_ptr<i64>(), *bsrc = b_src.data_ptr<i64>(), *bver = b_srcver.data_ptr<i64>();
    for (i64 i = 0; i < m; i++) {
      i64 id = bid_out[i];
      memcpy(b_row(id), values+i*row_floats, b_elems*sizeof(float));
      meta[2*id] = ends[i]; meta[2*id+1] = version;
      memcpy(bsrc+id*bs, src+i*bs, bs*sizeof(i64)); memcpy(bver+id*bs, srcver+i*bs, bs*sizeof(i64));
    }
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
    write_decisions_raw(n, (const uint8_t *)kv.data_ptr(), kv_elems*kv_itemsize, z.data_ptr<float>(), z_elems, steps.data_ptr<i64>(),
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
    write_blocks_raw(m, values.data_ptr<float>(), b_elems, ends.data_ptr<i64>(), version, src.data_ptr<i64>(), srcver.data_ptr<i64>(),
                     bid.data_ptr<i64>());
    return bid;
  }
  // raw frames
  py::object raw_entry(i64 dec) {
    std::lock_guard<std::mutex> g(mu);
    auto it = raw.find(dec);
    TORCH_CHECK(it != raw.end(), "PageStore: no raw frame for decision ", dec);
    const RawEntry &e = it->second;
    at::Tensor pend = at::empty({nh}, at::kLong);
    memcpy(pend.data_ptr<i64>(), e.pending.data(), nh*sizeof(i64));
    return py::make_tuple((int)e.first, (int)e.schema, py::bytes(e.frame), pend);
  }
  bool raw_has(i64 dec) { std::lock_guard<std::mutex> g(mu); return raw.count(dec) > 0; }
  i64 raw_count() { std::lock_guard<std::mutex> g(mu); return (i64)raw.size(); }
  at::Tensor raw_keys() {
    std::lock_guard<std::mutex> g(mu);
    at::Tensor k = at::empty({(i64)raw.size()}, at::kLong);
    i64 *p = k.data_ptr<i64>(); i64 i = 0;
    for (auto &kv : raw) p[i++] = kv.first;
    return k;
  }
  double raw_mb() {
    std::lock_guard<std::mutex> g(mu);
    double b = 0; for (auto &kv : raw) b += kv.second.frame.size();
    return b/1048576.0;
  }
  // (B45 source_arena) first [n] u8, schema [n] i32, sizes [n] i64, offsets [n] i64, joined frames u8, pending [n, nh]
  py::tuple source_arena(at::Tensor decisions) {
    decisions = decisions.to(at::kLong).contiguous();
    i64 n = decisions.numel();
    const i64 *d = decisions.data_ptr<i64>();
    std::lock_guard<std::mutex> g(mu);
    at::Tensor first = at::empty({n}, at::kByte), schema = at::empty({n}, at::kInt), sizes = at::empty({n}, at::kLong),
               off = at::empty({n}, at::kLong), pend = at::empty({n, nh}, at::kLong);
    i64 total = 0;
    std::vector<const RawEntry *> es(n);
    for (i64 i = 0; i < n; i++) {
      auto it = raw.find(d[i]);
      TORCH_CHECK(it != raw.end(), "PageStore.source_arena: no raw frame for decision ", d[i]);
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
  void free_rows(at::Tensor dead_d, at::Tensor dead_b, at::Tensor dead_raw) {
    dead_d = dead_d.to(at::kLong).contiguous(); dead_b = dead_b.to(at::kLong).contiguous(); dead_raw = dead_raw.to(at::kLong).contiguous();
    std::lock_guard<std::mutex> g(mu);
    bool *du = d_used.data_ptr<bool>(), *bu = b_used.data_ptr<bool>();
    i64 *dm = d_meta.data_ptr<i64>(), *bm = b_meta.data_ptr<i64>();
    // Python: d_free = concatenate((dead[::-1], d_free)): dead ids go to the front of the array = the bottom of our
    // stack (popped last); dead is ascending, reversed -> the array holds dead descending at the front
    const i64 *dd = dead_d.data_ptr<i64>(); i64 nd = dead_d.numel();
    std::vector<i64> nf; nf.reserve(nd+d_free.size());
    for (i64 i = nd-1; i >= 0; i--) nf.push_back(dd[i]);
    nf.insert(nf.end(), d_free.begin(), d_free.end()); d_free.swap(nf);
    for (i64 i = 0; i < nd; i++) { du[dd[i]] = false; dm[2*dd[i]] = -1; dm[2*dd[i]+1] = -1; }
    const i64 *db = dead_b.data_ptr<i64>(); i64 nb = dead_b.numel();
    std::vector<i64> bf; bf.reserve(nb+b_free.size());
    for (i64 i = nb-1; i >= 0; i--) bf.push_back(db[i]);
    bf.insert(bf.end(), b_free.begin(), b_free.end()); b_free.swap(bf);
    for (i64 i = 0; i < nb; i++) { bu[db[i]] = false; bm[2*db[i]] = -1; bm[2*db[i]+1] = -1; }
    const i64 *dr = dead_raw.data_ptr<i64>(); i64 nr = dead_raw.numel();
    for (i64 i = 0; i < nr; i++) raw.erase(dr[i]);
    freed_kv += nd; freed_blocks += nb; freed_raw += nr;
  }
  py::dict stats() {
    std::lock_guard<std::mutex> g(mu);
    py::dict d;
    d["kv_pages"] = (i64)kv_pages.size(); d["b_pages"] = (i64)b_pages.size(); d["d_free"] = (i64)d_free.size(); d["b_free"] = (i64)b_free.size();
    d["raw"] = (i64)raw.size(); d["next_decision"] = next_decision; d["reallocs"] = reallocs;
    d["cap_pages_d"] = (i64)kv_pages.capacity(); d["cap_pages_b"] = (i64)b_pages.capacity(); d["freed_kv"] = freed_kv; d["freed_blocks"] = freed_blocks; d["freed_raw"] = freed_raw;
    return d;
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

// (B58a+b) the native consumer: train_tok's Episodes.arrive (per-record arithmetic) and store (rollout writes) of the
// main rows, on the driver thread, in the Python path's order (arrive over all the record's rows, then store over the
// go rows); the episode-level Python work (first-record snapshots, finished-episode summaries) stays in Python, fed by
// each record's per-row snapshot of the accumulators taken before an episode end resets them.
struct ConsumerCfg {
  bool on = false;
  i64 o_first = 0, o_done = 0, o_damage = 0, o_hurt = 0, o_events = 0, o_stage = 0, o_player = 0, o_pforce = 0;
  i64 row_frame_at = 0;          // ROW's visible_frame offset (ROWC copies the bytes before it)
  double w_damage = 0, w_hurt = 0, w_time = 0, w_death = 0, w_timeout = 0, w_room = 0, w_explore = 0, w_boss = 0, w_exit = 0,
         w_heal = 0, w_resource = 0, end_bonus = 0, gamma = 0.99;
  bool floor = false, run = false, items = false, gains_on = false, plr_on = false, pick_on = false;
  i64 ev_exit = 8, ev_item = 16, ev_active = 32, ev_pill = 64, done_branch = 4;
  // Episodes arrays (numpy, owned by Python)
  double *ret = nullptr, *ep_hurt = nullptr, *ep_damage = nullptr, *gains = nullptr, *healed = nullptr, *gathered = nullptr,
         *v0 = nullptr, *gret = nullptr;
  i64 *rooms = nullptr, *boss = nullptr, *extra = nullptr, *steps = nullptr, *pick_steps = nullptr;
};
struct RollSlot {                 // one Rollout's arrays (numpy, owned by Python) and its frame arena (owned here)
  bool set = false;
  i64 cap = 0, rowc_size = 0, off_visible_off = 0, priv_size = 0;
  uint8_t *rows = nullptr, *terminal = nullptr, *close_done = nullptr, *close_ev = nullptr, *priv = nullptr, *boot = nullptr;
  float *reward = nullptr, *close_hurt = nullptr, *logp = nullptr, *value = nullptr, *value_g = nullptr;
  i64 *open = nullptr, *length = nullptr, *pending = nullptr, *actions = nullptr, *version = nullptr;
  at::Tensor arena; i64 used = 0; i64 grown = 0;
};
static const i64 N_SNAP = 17;    // ret, hurt, damage, rooms, boss, extra[6], healed, gathered, pick steps, gret, v0, steps

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

  explicit ActorDriver(std::shared_ptr<PageStore> s) : store(std::move(s)) {}
  ~ActorDriver() { stop(); }

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
    if (!r.arena.defined() || r.arena.numel() < arena_bytes) r.arena = at::zeros({std::max<i64>(arena_bytes, 1024)}, at::kByte);
    r.used = 0;
    r.set = true;
  }
  void use_rollout(int slot) { TORCH_CHECK(slot >= -1 && slot <= 1, "use_rollout: slot"); cur_slot = slot; }
  // the frame arena of a slot (B45 FrameArena's buf / used, appends of the hand-off's carry)
  at::Tensor arena_buf(int slot) { std::lock_guard<std::mutex> g(amu); return slots[slot].arena; }
  i64 arena_used(int slot) { std::lock_guard<std::mutex> g(amu); return slots[slot].used; }
  void arena_reset(int slot) { std::lock_guard<std::mutex> g(amu); slots[slot].used = 0; }
  i64 arena_grown(int slot) { std::lock_guard<std::mutex> g(amu); return slots[slot].grown; }
  at::Tensor arena_append_from(int slot, at::Tensor buf, at::Tensor offs, at::Tensor sizes) {
    buf = buf.contiguous(); offs = offs.to(at::kLong).contiguous(); sizes = sizes.to(at::kLong).contiguous();
    i64 n = offs.numel();
    at::Tensor out = at::empty({n}, at::kLong);
    std::lock_guard<std::mutex> g(amu);
    for (i64 i = 0; i < n; i++)
      out.data_ptr<i64>()[i] = arena_append_locked(slots[slot], buf.data_ptr<uint8_t>()+offs.data_ptr<i64>()[i], sizes.data_ptr<i64>()[i]);
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
                      at::Tensor bw, py::dict layout, i64 heads_, i64 nh_) {
    TORCH_CHECK(key.size() == 6, "graph key: (owner, cap, wb, we, wn, ws)");
    GraphEntry e;
    e.cap = key[1]; e.wb = key[2]; e.we = key[3]; e.wn = key[4]; e.ws = key[5];
    e.host_flat = host_flat; e.dev_flat = dev_flat; e.graph = std::move(graph);
    e.out = out; e.cur = cur; e.z = z; e.written = written; e.bo = bo; e.bc = bc; e.bz = bz; e.bw = bw; e.heads = heads_; e.nh = nh_;
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
    for (i64 i = 0; i < e.cap; i++) hid[i] = i < n ? ids[i] : scratch_row;
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

  // ---- the driver thread
  void start() {
    TORCH_CHECK(!running, "driver already running");
    TORCH_CHECK(have_stream && rows_base && step_host && n_workers > 0, "driver: set_stream / set_sampler / set_arena first");
    stop_req = false; pause_req = false; paused = false; error.clear();
    running = true;
    th = std::thread([this] { loop(); });
  }
  void stop() {
    if (!running) return;
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
    py::list h; for (auto &x : lane_hist) h.append(x.load()); d["lane_hist"] = h;
    return d;
  }

 private:
  static i64 bucket(const std::vector<i64> &b, i64 m) { for (i64 w : b) if (m <= w) return w; return -1; }

  void push_record(std::shared_ptr<Record> r) {
    std::unique_lock<std::mutex> lk(qmu);
    qcv.wait(lk, [&] { return (i64)queue.size() < queue_max || stop_req; });
    queue.push_back(std::move(r));
    qcv.notify_all();
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
      while (!stop_req) {
        if (pause_req) { wait_paused(); continue; }
        bool lanes_waiting = lane_queue_size() > 0;
        if (main_hold) {   // (B54) lanes only
          if (lanes_waiting) run_lanes(); else std::this_thread::sleep_for(std::chrono::microseconds(300));
          continue;
        }
        if (ready_w.empty()) {
          ws.clear(); ss.clear();
          for (size_t q = 0; q < held_w.size();) {   // held messages whose capture is done join the batch
            if (!cap_wait[held_w[q]]) { ws.push_back(held_w[q]); ss.push_back(held_s[q]); held_w.erase(held_w.begin()+q); held_s.erase(held_s.begin()+q); }
            else q++;
          }
          for (size_t i = 0; i < fds.size(); i++) { pfds[i].fd = fds[i]; pfds[i].events = fds[i] >= 0 ? POLLIN : 0; pfds[i].revents = 0; }
          int k = ::poll(pfds.data(), (nfds_t)pfds.size(), (lanes_waiting || !ws.empty()) ? 0 : 1);
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
          if (ws.empty()) { if (lanes_waiting) run_lanes(); continue; }
        } else { ws = ready_w; ss = ready_s; ready_w.clear(); ready_s.clear(); }
        step_batch(ws, ss);
        if (!pause_req) run_lanes();   // (the other generations' jobs after the main reply, as the Python path)
      }
    } catch (const std::exception &ex) {
      error = ex.what();
      std::lock_guard<std::mutex> g(cmu); paused = true; ccv.notify_all();
      auto r = std::make_shared<Record>(); r->kind = 3; finish_empty(*r); push_record(r);
    }
    running = false;
  }

  i64 arena_append_locked(RollSlot &r, const uint8_t *src, i64 n) {
    i64 padded = (n+7)/8*8;
    i64 at_ = r.used;
    if (at_+padded > r.arena.numel()) {   // grow by 1.5 (FrameArena's rule), the used bytes copied
      at::Tensor g = at::zeros({std::max<i64>(at_+padded, (i64)(r.arena.numel()*1.5))}, at::kByte);
      memcpy(g.data_ptr<uint8_t>(), r.arena.data_ptr<uint8_t>(), at_);
      r.arena = g; r.grown++;
    }
    if (n) memcpy(r.arena.data_ptr<uint8_t>()+at_, src, n);
    r.used = at_+padded;
    return at_;
  }
  // (B58a+b) Episodes.arrive over all the record's rows, then store over its go rows (main rows only), in the Python
  // path's arithmetic (float64 rewards, float32 gains as numpy computes them), into the current rollout slot
  void consume(Record &r, const uint8_t *rc, i64 n_all, const std::vector<i64> &ws, const std::vector<uint8_t> &go,
               const std::vector<i64> &go_idx, const float *out, const i64 *act, const i64 *pend, i64 ver, const uint8_t *privp) {
    int sl = cur_slot.load();
    i64 n = (i64)go_idx.size();
    r.kk = at::full({n}, -1, at::kLong);
    r.snap = at::zeros({n_all, N_SNAP}, at::kDouble);
    if (!cc.on || sl < 0 || !slots[sl].set) return;
    RollSlot &R = slots[sl];
    const ConsumerCfg &c = cc;
    double *snap = r.snap.data_ptr<double>();
    auto u8 = [&](const uint8_t *src, i64 o) { return (i64)src[o]; };
    auto f32 = [&](const uint8_t *src, i64 o) { float v; memcpy(&v, src+o, 4); return v; };
    auto cur_gains = [&](const uint8_t *src, double *g) {
      float p7 = f32(src, c.o_player+7*4), p8 = f32(src, c.o_player+8*4), p9 = f32(src, c.o_player+9*4), p10 = f32(src, c.o_player+10*4);
      float a = std::nearbyint(p7*12.0f);
      float b = std::nearbyint(p8*10.0f)+std::nearbyint(p9*10.0f);
      b = b+std::nearbyint(p10*50.0f);
      g[0] = (double)a; g[1] = (double)b;
    };
    // ---- arrive
    for (i64 i = 0; i < n_all; i++) {
      i64 w = ws[i];
      const uint8_t *src = rc+i*row_stride;
      bool first = u8(src, c.o_first) != 0;
      i64 d = u8(src, c.o_done);
      i64 k = R.open[w];
      i64 ix;
      if (first && k >= 0) {   // the episode before broke off
        ix = w*R.cap+k;
        R.reward[ix] = 0.0f; R.terminal[ix] = 1;
        if (R.close_hurt) { R.close_hurt[ix] = 0.0f; R.close_ev[ix] = 0; R.close_done[ix] = 255; }
        R.open[w] = -1; k = -1;
      }
      if (k >= 0) {
        ix = w*R.cap+k;
        double dmg = (double)f32(src, c.o_damage), hu = (double)f32(src, c.o_hurt);
        i64 ev = u8(src, c.o_events);
        if (R.close_hurt) { R.close_hurt[ix] = (float)hu; R.close_done[ix] = (uint8_t)d; R.close_ev[ix] = (uint8_t)ev; }
        double endt = d == 1 ? c.end_bonus : d == 2 ? -c.w_death : d == 3 ? -c.w_timeout : 0.0;
        double reward = c.w_damage*dmg-c.w_hurt*hu-c.w_time+endt;
        if (c.floor) {
          reward = reward+(c.w_room*(double)((ev & 1) > 0)+c.w_explore*(double)((ev & 2) > 0)+c.w_boss*(double)((ev & 4) > 0));
          if (c.run) reward = reward+c.w_exit*(double)((ev & c.ev_exit) > 0);
          c.rooms[w] += (ev & 1) > 0; c.boss[w] += (ev & 4) > 0;
          if (c.gains_on) {
            double g[2]; cur_gains(src, g);
            double dl0 = g[0]-c.gains[w*2], dl1 = g[1]-c.gains[w*2+1];
            reward = reward+c.w_heal*std::max(dl0, 0.0)+c.w_resource*std::max(dl1, 0.0);
            c.healed[w] += std::max(dl0, 0.0); c.gathered[w] += std::max(dl1, 0.0);
          }
        }
        if (c.run || c.items) {
          i64 *x = c.extra+w*6;
          x[0] += (ev & c.ev_exit) > 0; x[1] += (ev & c.ev_item) > 0; x[2] += (ev & c.ev_active) > 0; x[3] += (ev & c.ev_pill) > 0;
          x[5] = std::max<i64>(x[5], u8(src, c.o_stage));
        }
        R.reward[ix] = (float)reward; R.terminal[ix] = d != 0 && d != c.done_branch; R.open[w] = -1;
        c.ret[w] += reward;
        if (c.plr_on) {
          if (c.steps[w] == 0) c.v0[w] = (double)R.value[ix];
          c.gret[w] += std::pow(c.gamma, (double)c.steps[w])*reward;
          c.steps[w] += 1;
        }
        c.ep_hurt[w] += hu; c.ep_damage[w] += dmg;
      }
      if (c.gains_on && c.gains) cur_gains(src, c.gains+w*2);
      if ((c.run || c.items) && first) { c.extra[w*6+4] = u8(src, c.o_stage); c.extra[w*6+5] = u8(src, c.o_stage); }
      if (c.pick_on) { if (first) c.pick_steps[w] = 0; c.pick_steps[w] += u8(src, c.o_pforce) > 0; }
      if (d != 0) {   // an episode end: the accumulators for Python's summary, then reset
        double *sv = snap+i*N_SNAP;
        sv[0] = c.ret[w]; sv[1] = c.ep_hurt[w]; sv[2] = c.ep_damage[w]; sv[3] = (double)c.rooms[w]; sv[4] = (double)c.boss[w];
        for (int q = 0; q < 6; q++) sv[5+q] = c.extra ? (double)c.extra[w*6+q] : 0.0;
        sv[11] = c.healed[w]; sv[12] = c.gathered[w]; sv[13] = c.pick_steps ? (double)c.pick_steps[w] : 0.0;
        sv[14] = c.plr_on ? c.gret[w] : 0.0; sv[15] = c.plr_on ? c.v0[w] : 0.0; sv[16] = c.plr_on ? (double)c.steps[w] : 0.0;
        c.ret[w] = 0; c.ep_hurt[w] = 0; c.ep_damage[w] = 0; c.rooms[w] = 0; c.boss[w] = 0;
        if (c.extra) for (int q = 0; q < 6; q++) c.extra[w*6+q] = 0;
        c.healed[w] = 0; c.gathered[w] = 0;
        if (c.plr_on) { c.v0[w] = 0; c.gret[w] = 0; c.steps[w] = 0; }
      }
    }
    // ---- store (go rows)
    i64 ow = nh+3;
    i64 *kk = r.kk.data_ptr<i64>();
    for (i64 j = 0; j < n; j++) {
      i64 i = go_idx[j], w = ws[i];
      const uint8_t *src = rc+i*row_stride;
      i64 k = R.length[w];
      if (k >= R.cap) continue;   // (cannot happen with the trainer's capacity; dropped rather than overrun, as store)
      i64 ix = w*R.cap+k;
      uint8_t *dst = R.rows+ix*R.rowc_size;
      memset(dst, 0, R.rowc_size);
      memcpy(dst, src, c.row_frame_at);
      i64 nbytes = read_int(src+f_bytes, 4);
      i64 off;
      { std::lock_guard<std::mutex> g(amu); off = arena_append_locked(R, src+f_frame, std::max<i64>(0, std::min<i64>(nbytes, frame_bytes))); }
      memcpy(dst+R.off_visible_off, &off, 8);
      memcpy(R.pending+ix*nh, pend+j*nh, nh*8);
      memcpy(R.actions+ix*nh, act+j*nh, nh*8);
      R.logp[ix] = out[j*ow+nh]; R.value[ix] = out[j*ow+nh+1];
      if (R.value_g) R.value_g[ix] = out[j*ow+nh+2];
      R.version[ix] = ver;
      R.boot[ix] = u8(src, c.o_done) == c.done_branch;
      if (R.priv && privp) memcpy(R.priv+ix*R.priv_size, privp+i*priv_itemsize, R.priv_size);
      R.length[w] = k+1; R.open[w] = k;
      kk[j] = k;
    }
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

  void step_batch(const std::vector<i64> &ws, const std::vector<i64> &ss) {
    double t0 = now();
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
      if (cf_snap) { r->cf_snap[i] = cf_snap[ws[i]]; if (r->cf_snap[i]) cap_wait[ws[i]] = 1; }   // (before the reply)
      if (ended && !boot) memset(&pending_cur[ws[i]*nh], 0, nh*8);   // Episodes.arrive: pending[idx[ended & ~boot]] = 0
      if (first && go) memset(&pending_cur[ws[i]*nh], 0, nh*8);      //                  pending[idx[first & go]] = 0
      if (go) go_idx.push_back(i); else n_terminal++;
    }
    i64 n = (i64)go_idx.size();
    if (n == 0) {   // terminal records only: reply, hand the bookkeeping over
      for (i64 i = 0; i < n_all; i++) if (cf_snap && r->cf_snap[i]) cf_snap[ws[i]] = 0;
      reply_rows(ws);
      r->t_sync = r->t_reply = r->t_pages = r->t_done = now();
      finish_empty(*r);
      if (cc.on) consume(*r, rc, n_all, ws, r->go, go_idx, nullptr, nullptr, nullptr, version.load(),
                         priv ? r->priv.data_ptr<uint8_t>() : nullptr);   // (B58a+b: the episode ends)
      push_record(r); n_calls++;
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
    auto put_back = [&]() {   // (the merged jobs return to the front of the queue, in order)
      std::lock_guard<std::mutex> g(lmu);
      for (auto it = merged.rbegin(); it != merged.rend(); ++it) lane_q.push_front(*it);
      merged.clear(); n_lane = 0;
    };
    pack(!merged.empty());
    i64 wb, we, wn, wsl;
    bool ok = batch_widths(rows_ptr, n+n_lane, wb, we, wn, wsl);
    if (!ok && !merged.empty()) {   // the lanes would make the batch invalid / too wide: main rows alone
      put_back(); pack(false); ok = batch_widths(rows_ptr, n, wb, we, wn, wsl);
    }
    i64 n_tot = n+n_lane;
    i64 cap = 1; while (cap < n_tot) cap <<= 1;
    if (!ok) {   // Python handles this batch (it raises the same error, or takes the eager path)
      r->kind = 2; r->cap = cap; r->wb = wb; r->we = we; r->wn = wn; r->ws = wsl;
      hand_to_python(r, ws, ss);
      return;
    }
    std::vector<i64> key{owner_main, cap, wb, we, wn, wsl};
    GraphEntry *e = find_graph(key);
    if (!e) {   // not captured yet: Python captures (needs the GIL and the capture admission); the batch is retried
      put_back();
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
    double t_host0 = now();
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
    double t2 = now(); r->t_reply = t2;
    // the main rows' pages, then their record (the lanes' pages and records only after it: B54 condition a)
    write_part(*e, rows_ptr, 0, n, ids, steps_before, complete, first, pend.data(), ver, *r);
    if (cc.on) consume(*r, rc, n_all, ws, r->go, go_idx, r->out.data_ptr<float>(), act, pend.data(), ver,
                       priv ? r->priv.data_ptr<uint8_t>() : nullptr);   // (B58a+b: arrive + store, before the record)
    double t3 = now(); r->t_pages = t3; r->t_done = t3; t_pages_s = t_pages_s+(t3-t2); t_host_s = t_host_s+(t3-t0)-(t1-t_host0);
    n_steps++; n_calls++; n_rows = n_rows+n;
    push_record(r);
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
      .def_property("next_decision", [](PageStore &s) { return s.next_decision; }, [](PageStore &s, i64 v) { s.next_decision = v; })
      .def("write_decisions", &PageStore::write_decisions).def("write_blocks", &PageStore::write_blocks)
      .def("raw_entry", &PageStore::raw_entry).def("raw_has", &PageStore::raw_has).def("raw_count", &PageStore::raw_count)
      .def("raw_keys", &PageStore::raw_keys).def("raw_mb", &PageStore::raw_mb).def("source_arena", &PageStore::source_arena)
      .def("free_rows", &PageStore::free_rows).def("stats", &PageStore::stats);
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
      .def("set_consumer", &ActorDriver::set_consumer).def("set_rollout", &ActorDriver::set_rollout).def("use_rollout", &ActorDriver::use_rollout)
      .def("arena_buf", &ActorDriver::arena_buf).def("arena_used", &ActorDriver::arena_used).def("arena_reset", &ActorDriver::arena_reset)
      .def("arena_grown", &ActorDriver::arena_grown).def("arena_append_from", &ActorDriver::arena_append_from)
      .def("consume_rows", &ActorDriver::consume_rows)
      .def_property_readonly("consumer_on", [](ActorDriver &d) { return d.cc.on; })
      .def_property_readonly("running", [](ActorDriver &d) { return d.running.load(); });
}
