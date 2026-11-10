// B58c: the torch-free core of the C++ actor (plain pointers and integers; abp_actor.cpp wraps it in tensors for Python):
//
//   PageCore     the paged store's memory, allocation, metadata, raw frames, and the GC stamp rule (B59): every row
//                carries the stamp of the record that allocated it (the driver's record sequence); a collection frees
//                only rows stamped before the first record its consumer has not taken yet, so a GC may scan while the
//                driver writes and records wait in the queue
//   Arena        a rollout's frame arena (8-byte aligned appends; grows by 1.5, retired buffers kept until reset, so a
//                reader's view of an older buffer stays valid while the writer appends)
//   consume()    the native consumer (B58a+b): Episodes.arrive's arithmetic over a record's rows, store's rollout writes
//                of its go rows; length / open published with atomic stores
//
// Threads: the driver writes (alloc, pages, consumer); the Python main thread collects (begin_collect / free_rows) and
// reads published rows; the learner reads rows of a handed rollout and pages of its job's snapshots. Every shared
// structure is under PageCore::mu or the arena mutex, or written once before its publication (a record / a hand-off).
// bridge/engine/core_tests/test_abp_core.cpp runs it under -fsanitize=thread with random interleavings
// (bridge/ops/run_core_tsan.sh -> reports/b42-speed/core-tsan.log).
//
// Lifetimes (B58c): the wrapper holds the PageCore in a shared_ptr and every tensor view of its memory (pages, meta,
// used, src) carries a copy of it in its deleter, so a view outlives the store (a learner snapshot holding a page
// after the store's owner went away reads the page, never freed memory). Arena buffers are shared_ptr too: a view of
// a buffer (Python's frames) keeps it alive across the arena's growth and reset. The rollout arrays a consumer slot
// points at are Python's (numpy, owned by train_tok's Rollout objects): Python keeps them alive while the slot is set
// and re-sets the slot (set_rollout) before it replaces or drops them; the driver is stopped before they go.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace abpc {
using i64 = int64_t;

inline i64 rd_int(const uint8_t *p, int size) {
  switch (size) {
    case 1: return *p;
    case 2: { int16_t v; memcpy(&v, p, 2); return v; }
    case 4: { int32_t v; memcpy(&v, p, 4); return v; }
    default: { i64 v; memcpy(&v, p, 8); return v; }
  }
}
inline void store_i64(i64 *p, i64 v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline i64 load_i64(const i64 *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

struct RawEntry {
  uint8_t first;
  int32_t schema;
  i64 gen = 0;   // (B58c) the stamp of the record that wrote it (the GC stamp rule covers raw frames too)
  std::string frame;
  std::vector<i64> pending;
};

// ------------------------------------------------------------------------------------------------------- PageCore
struct PageCore {
  i64 page, kv_row_bytes, z_elems, b_elems, bs, nh, max_pages_d, max_pages_b, cap_d, cap_b;
  std::vector<uint8_t *> kv_pages, z_pages, b_pages;   // reserved to the caps: never reallocated
  std::vector<std::unique_ptr<uint8_t[]>> owned;        // the pages' memory (zeroed), alive with the core
  std::unique_ptr<i64[]> d_meta, b_meta, b_src, b_srcver, d_gen, b_gen;
  std::unique_ptr<uint8_t[]> d_used, b_used;
  std::vector<i64> d_free, b_free;
  i64 next_decision = 0, freed_kv = 0, freed_blocks = 0, freed_raw = 0, reallocs = 0, stamp = 0, stamp_kept = 0;
  std::unordered_map<i64, RawEntry> raw;
  std::mutex mu;   // allocation, free lists, page vectors, raw, stamps

  PageCore(i64 page_, i64 kv_row_bytes_, i64 z_elems_, i64 b_elems_, i64 bs_, i64 nh_, i64 max_pages_d_, i64 max_pages_b_)
      : page(page_), kv_row_bytes(kv_row_bytes_), z_elems(z_elems_), b_elems(b_elems_), bs(bs_), nh(nh_),
        max_pages_d(max_pages_d_), max_pages_b(max_pages_b_) {
    cap_d = max_pages_d*page; cap_b = max_pages_b*page;
    d_meta.reset(new i64[cap_d*2]); b_meta.reset(new i64[cap_b*2]); b_src.reset(new i64[cap_b*bs]); b_srcver.reset(new i64[cap_b*bs]);
    d_gen.reset(new i64[cap_d]()); b_gen.reset(new i64[cap_b]());
    std::fill(d_meta.get(), d_meta.get()+cap_d*2, -1); std::fill(b_meta.get(), b_meta.get()+cap_b*2, -1);
    std::fill(b_src.get(), b_src.get()+cap_b*bs, -1); std::fill(b_srcver.get(), b_srcver.get()+cap_b*bs, -1);
    d_used.reset(new uint8_t[cap_d]()); b_used.reset(new uint8_t[cap_b]());
    kv_pages.reserve(max_pages_d); z_pages.reserve(max_pages_d); b_pages.reserve(max_pages_b);
  }
  uint8_t *new_page(i64 bytes) { owned.emplace_back(new uint8_t[bytes]()); return owned.back().get(); }
  void grow_d_locked() {
    i64 p = (i64)kv_pages.size();
    if (p >= max_pages_d) throw std::runtime_error("PageStore: decision pages exhausted (raise max_pages_d)");
    size_t c0 = kv_pages.capacity(), c1 = z_pages.capacity();
    kv_pages.push_back(new_page(page*kv_row_bytes)); z_pages.push_back(new_page(page*z_elems*4));
    reallocs += (kv_pages.capacity() != c0)+(z_pages.capacity() != c1);
    i64 base = p*page;
    // Python: d_free = np.concatenate((np.arange(base+P-1, base-1, -1), d_free)): the new page's ids, descending, in
    // front of the old free ids; _alloc takes the array's last n ids reversed (our pops from the back)
    for (i64 i = page-1; i >= 0; i--) d_free.push_back(base+i);
    std::rotate(d_free.begin(), d_free.end()-page, d_free.end());
  }
  void grow_b_locked() {
    i64 p = (i64)b_pages.size();
    if (p >= max_pages_b) throw std::runtime_error("PageStore: block pages exhausted (raise max_pages_b)");
    size_t c2 = b_pages.capacity();
    b_pages.push_back(new_page(page*b_elems*4));
    reallocs += (b_pages.capacity() != c2);
    i64 base = p*page;
    for (i64 i = page-1; i >= 0; i--) b_free.push_back(base+i);
    std::rotate(b_free.begin(), b_free.end()-page, b_free.end());
  }
  void alloc_locked(bool decisions, i64 n, i64 *out) {
    std::vector<i64> &free = decisions ? d_free : b_free;
    while ((i64)free.size() < n) (decisions ? grow_d_locked() : grow_b_locked());
    uint8_t *used = decisions ? d_used.get() : b_used.get();
    i64 *g = decisions ? d_gen.get() : b_gen.get();
    for (i64 i = 0; i < n; i++) {
      out[i] = free.back(); free.pop_back();
      used[out[i]] = 1; g[out[i]] = stamp;
    }
  }
  void grow_ahead(i64 min_free_d, i64 min_free_b) {
    std::lock_guard<std::mutex> l(mu);
    while ((i64)d_free.size() < min_free_d) grow_d_locked();
    while ((i64)b_free.size() < min_free_b) grow_b_locked();
  }
  uint8_t *kv_row(i64 id) { return kv_pages[id/page]+(id % page)*kv_row_bytes; }
  float *z_row(i64 id) { return (float *)z_pages[id/page]+(id % page)*z_elems; }
  float *b_row(i64 id) { return (float *)b_pages[id/page]+(id % page)*b_elems; }
  // page pointers for other threads (B58a: the vectors never reallocate; under the lock for the size check)
  uint8_t *kv_page(i64 p) { std::lock_guard<std::mutex> l(mu); return kv_pages.at(p); }
  uint8_t *z_page(i64 p) { std::lock_guard<std::mutex> l(mu); return z_pages.at(p); }
  uint8_t *b_page(i64 p) { std::lock_guard<std::mutex> l(mu); return b_pages.at(p); }

  void write_decisions_raw(i64 n, const uint8_t *kv, i64 kv_stride, const float *z, i64 z_stride, const i64 *steps, i64 version,
                           const uint8_t *first, const int32_t *schema, const uint8_t *const *frames, const i64 *nbytes,
                           const i64 *pending, i64 *dec_out, i64 *kvs_out) {
    std::lock_guard<std::mutex> l(mu);
    alloc_locked(true, n, kvs_out);
    for (i64 i = 0; i < n; i++) dec_out[i] = next_decision+i;
    next_decision += n;
    for (i64 i = 0; i < n; i++) {
      i64 id = kvs_out[i];
      memcpy(kv_row(id), kv+i*kv_stride, kv_row_bytes);
      memcpy(z_row(id), z+i*z_stride, z_elems*sizeof(float));
      d_meta[2*id] = steps[i]; d_meta[2*id+1] = version;
      RawEntry &e = raw[dec_out[i]];
      e.first = first[i]; e.schema = schema[i]; e.gen = stamp;
      e.frame.assign((const char *)frames[i], (size_t)nbytes[i]);
      e.pending.assign(pending+i*nh, pending+(i+1)*nh);
    }
  }
  void write_blocks_raw(i64 m, const float *values, i64 row_floats, const i64 *ends, i64 version, const i64 *src, const i64 *srcver,
                        i64 *bid_out) {
    std::lock_guard<std::mutex> l(mu);
    alloc_locked(false, m, bid_out);
    for (i64 i = 0; i < m; i++) {
      i64 id = bid_out[i];
      memcpy(b_row(id), values+i*row_floats, b_elems*sizeof(float));
      b_meta[2*id] = ends[i]; b_meta[2*id+1] = version;
      memcpy(b_src.get()+id*bs, src+i*bs, bs*sizeof(i64)); memcpy(b_srcver.get()+id*bs, srcver+i*bs, bs*sizeof(i64));
    }
  }
  // (B59) the stamp of the rows allocated from now on (the driver: its record sequence number)
  void set_stamp(i64 s) { std::lock_guard<std::mutex> l(mu); stamp = s; }
  // the used flags, copied under the lock (a GC scanning while the driver allocates)
  void used_snapshot(uint8_t *d_out, uint8_t *b_out) {
    std::lock_guard<std::mutex> l(mu);
    memcpy(d_out, d_used.get(), cap_d); memcpy(b_out, b_used.get(), cap_b);
  }
  // free dead ids (decisions, blocks, raw frames); with limit > 0 only rows stamped before it (the records the GC's
  // consumer had taken when it scanned); Python's order of the free arrays (dead[::-1] in front)
  void free_rows(const i64 *dd, i64 nd, const i64 *db, i64 nb, const i64 *dr, i64 nr, i64 limit_gen = 0) {
    std::lock_guard<std::mutex> l(mu);
    auto do_free = [&](const i64 *dead, i64 n_, std::vector<i64> &free, uint8_t *used, i64 *meta, const i64 *g, i64 &freed) {
      std::vector<i64> keep_ids; keep_ids.reserve(n_);
      for (i64 i = 0; i < n_; i++) {
        i64 id = dead[i];
        if (limit_gen > 0 && g[id] >= limit_gen) { stamp_kept++; continue; }   // its record not taken yet: not seen
        if (!used[id]) continue;
        keep_ids.push_back(id);
      }
      std::vector<i64> nf; nf.reserve(keep_ids.size()+free.size());
      for (i64 i = (i64)keep_ids.size()-1; i >= 0; i--) nf.push_back(keep_ids[i]);
      nf.insert(nf.end(), free.begin(), free.end()); free.swap(nf);
      for (i64 id : keep_ids) { used[id] = 0; meta[2*id] = -1; meta[2*id+1] = -1; }
      freed += (i64)keep_ids.size();
    };
    do_free(dd, nd, d_free, d_used.get(), d_meta.get(), d_gen.get(), freed_kv);
    do_free(db, nb, b_free, b_used.get(), b_meta.get(), b_gen.get(), freed_blocks);
    for (i64 i = 0; i < nr; i++) {
      auto it = raw.find(dr[i]);
      if (it == raw.end()) continue;
      if (limit_gen > 0 && it->second.gen >= limit_gen) { stamp_kept++; continue; }
      raw.erase(it); freed_raw++;
    }
  }
};

// ---------------------------------------------------------------------------------------------------------- Arena
struct Arena {
  std::shared_ptr<uint8_t[]> cur;                    // (shared: a reader's view keeps its buffer alive, see Lifetimes)
  std::vector<std::shared_ptr<uint8_t[]>> retired;   // buffers outgrown since the last reset (readers may hold views)
  i64 cap = 0, used = 0, grown = 0;
  void ensure(i64 bytes) {
    if (bytes <= cap) return;
    if (cur) retired.push_back(cur);
    cur.reset(new uint8_t[bytes]()); cap = bytes; used = 0;
  }
  i64 append(const uint8_t *src, i64 n) {   // FrameArena.append's rule: 8-byte aligned starts, grow by 1.5
    i64 padded = (n+7)/8*8, at = used;
    if (at+padded > cap) {
      i64 nc = std::max<i64>(at+padded, (i64)(cap*1.5));
      std::shared_ptr<uint8_t[]> g(new uint8_t[nc]());
      if (at) memcpy(g.get(), cur.get(), at);
      retired.push_back(std::move(cur)); cur = std::move(g); cap = nc; grown++;
    }
    if (n) memcpy(cur.get()+at, src, n);
    used = at+padded;
    return at;
  }
  void reset() { used = 0; retired.clear(); }
};

// --------------------------------------------------------------------------------------------- the native consumer
struct ConsumerCfg {
  bool on = false;
  i64 o_first = 0, o_done = 0, o_damage = 0, o_hurt = 0, o_events = 0, o_stage = 0, o_player = 0, o_pforce = 0;
  i64 row_frame_at = 0;
  double w_damage = 0, w_hurt = 0, w_time = 0, w_death = 0, w_timeout = 0, w_room = 0, w_explore = 0, w_boss = 0, w_exit = 0,
         w_heal = 0, w_resource = 0, end_bonus = 0, gamma = 0.99;
  bool floor = false, run = false, items = false, gains_on = false, plr_on = false, pick_on = false;
  i64 ev_exit = 8, ev_item = 16, ev_active = 32, ev_pill = 64, done_branch = 4;
  double *ret = nullptr, *ep_hurt = nullptr, *ep_damage = nullptr, *gains = nullptr, *healed = nullptr, *gathered = nullptr,
         *v0 = nullptr, *gret = nullptr;
  i64 *rooms = nullptr, *boss = nullptr, *extra = nullptr, *steps = nullptr, *pick_steps = nullptr;
};
struct RollSlot {
  bool set = false;
  i64 cap = 0, rowc_size = 0, off_visible_off = 0, priv_size = 0;
  uint8_t *rows = nullptr, *terminal = nullptr, *close_done = nullptr, *close_ev = nullptr, *priv = nullptr, *boot = nullptr;
  float *reward = nullptr, *close_hurt = nullptr, *logp = nullptr, *value = nullptr, *value_g = nullptr;
  i64 *open = nullptr, *length = nullptr, *pending = nullptr, *actions = nullptr, *version = nullptr;
  Arena arena;
};
struct RowLayout { i64 row_stride, f_bytes, f_frame, frame_bytes, nh, priv_itemsize; };
static const i64 N_SNAP = 17;   // ret, hurt, damage, rooms, boss, extra[6], healed, gathered, pick steps, gret, v0, steps

// Episodes.arrive over all n_all rows of a record (rows rc, workers ws), then store over its n go rows (go_idx), in the
// Python path's arithmetic (float64 rewards, float32 gains as numpy computes them); kk_out [n] (-1: dropped), snap_out
// [n_all, N_SNAP] (the accumulators of a row whose episode ends, before their reset). amu guards the arena.
inline void consume(const ConsumerCfg &c, RollSlot &R, std::mutex &amu, const RowLayout &L, const uint8_t *rc, i64 n_all,
                    const i64 *ws, const i64 *go_idx, i64 n, const float *out, const i64 *act, const i64 *pend, i64 ver,
                    const uint8_t *privp, i64 *kk_out, double *snap) {
  auto u8 = [&](const uint8_t *src, i64 o) { return (i64)src[o]; };
  auto f32 = [&](const uint8_t *src, i64 o) { float v; memcpy(&v, src+o, 4); return v; };
  auto cur_gains = [&](const uint8_t *src, double *g) {
    float p7 = f32(src, c.o_player+7*4), p8 = f32(src, c.o_player+8*4), p9 = f32(src, c.o_player+9*4), p10 = f32(src, c.o_player+10*4);
    float a = std::nearbyint(p7*12.0f);
    float b = std::nearbyint(p8*10.0f)+std::nearbyint(p9*10.0f);
    b = b+std::nearbyint(p10*50.0f);
    g[0] = (double)a; g[1] = (double)b;
  };
  for (i64 i = 0; i < n_all; i++) {
    i64 w = ws[i];
    const uint8_t *src = rc+i*L.row_stride;
    bool first = u8(src, c.o_first) != 0;
    i64 d = u8(src, c.o_done);
    i64 k = load_i64(&R.open[w]);
    i64 ix;
    if (first && k >= 0) {   // the episode before broke off
      ix = w*R.cap+k;
      R.reward[ix] = 0.0f; R.terminal[ix] = 1;
      if (R.close_hurt) { R.close_hurt[ix] = 0.0f; R.close_ev[ix] = 0; R.close_done[ix] = 255; }
      store_i64(&R.open[w], -1); k = -1;
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
      R.reward[ix] = (float)reward; R.terminal[ix] = d != 0 && d != c.done_branch;
      store_i64(&R.open[w], -1);
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
  i64 ow = L.nh+3;
  for (i64 j = 0; j < n; j++) {
    i64 i = go_idx[j], w = ws[i];
    const uint8_t *src = rc+i*L.row_stride;
    i64 k = load_i64(&R.length[w]);
    kk_out[j] = -1;
    if (k >= R.cap) continue;   // (cannot happen with the trainer's capacity; dropped rather than overrun, as store)
    i64 ix = w*R.cap+k;
    uint8_t *dst = R.rows+ix*R.rowc_size;
    memset(dst, 0, R.rowc_size);
    memcpy(dst, src, c.row_frame_at);
    i64 nbytes = rd_int(src+L.f_bytes, 4);
    i64 off;
    { std::lock_guard<std::mutex> g(amu); off = R.arena.append(src+L.f_frame, std::max<i64>(0, std::min<i64>(nbytes, L.frame_bytes))); }
    memcpy(dst+R.off_visible_off, &off, 8);
    memcpy(R.pending+ix*L.nh, pend+j*L.nh, L.nh*8);
    memcpy(R.actions+ix*L.nh, act+j*L.nh, L.nh*8);
    R.logp[ix] = out[j*ow+L.nh]; R.value[ix] = out[j*ow+L.nh+1];
    if (R.value_g) R.value_g[ix] = out[j*ow+L.nh+2];
    R.version[ix] = ver;
    R.boot[ix] = u8(src, c.o_done) == c.done_branch;
    if (R.priv && privp) memcpy(R.priv+ix*R.priv_size, privp+i*L.priv_itemsize, R.priv_size);
    store_i64(&R.length[w], k+1); store_i64(&R.open[w], k);
    kk_out[j] = k;
  }
}

}  // namespace abpc
