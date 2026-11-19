// P package (2026-11-19): the host side of one TokPolicy actor call in C++, torch-free -- the thin driver's
// (abp_tok_driver.cpp) checks and packing, held to the Python reference by the same golden samples
// (bridge/python/tools/p_golden.py; tools/p_golden_cpp.py runs them through this header):
//   check_row / check_batch   = v3_input._reject_row / reject_code / check_batch (the REJECT codes, the record named,
//                               the counts (p, e, n, s), the N / S truncations, the E bucket)
//   crn_row                   = tok_policy.crn_uniforms, bit for bit (as abp_actor.cpp's)
//   fold_pack                 = tok_policy.fold_pack (the bytes fold_defined names; the others keep what the buffers held)
// Every constant comes from Python at setup (schema_v3, v3_input, ROW's field offsets): nothing of the schema is
// written here, so a schema batch changes no line of this file.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace abptok {
using i64 = int64_t;

// ---------------------------------------------------------------- the v3 frame check (v3_input.REJECT's codes)
enum Reject { REJ_OK = 0, REJ_SCHEMA = 1, REJ_BYTES = 2, REJ_NONWHOLE = 3, REJ_MAGIC = 4, REJ_VERSION_WORD = 5, REJ_COUNTS = 6,
              REJ_SIZE = 7, REJ_P0 = 8, REJ_P_MAX = 9 };

struct V3 {   // schema_v3 / v3_input's numbers (set from Python)
  i64 header = 8, h_p = 2, fixed = 0, body = 0, ef = 0, nf = 0, sf = 0, frame_bytes = 0;
  double magic = 0, version = 0, version_word = 0, main_role = 1;
  i64 cap_body = 0, cap_entity = 0, n_max = 0, s_max = 0, p_max = 0;
  i64 p0_role = 0, p0_known = 0;   // (frame float indices of P row 0's body_role / position_known: v3_input.P0_WORDS)
  std::vector<i64> e_buckets;
};

static inline bool whole(double x) { return std::isfinite(x) && x == std::trunc(x); }

// v3_input._reject_row: one record's 8 header words, its visible_schema and its byte count (as Python floats)
static inline int check_row(const V3 &c, const float *h, double version, double nbytes, i64 cnt[4]) {
  if (!std::isfinite(version) || version != c.version) return REJ_SCHEMA;
  if (!std::isfinite(nbytes)) return REJ_BYTES;
  double nb = std::trunc(nbytes);   // (int(nbytes))
  if (nb < (double)(c.header*4) || nb > (double)c.frame_bytes) return REJ_BYTES;
  i64 n = (i64)nb;
  if (n%4) return REJ_BYTES;
  double w[8];
  for (int j = 0; j < 8; j++) { w[j] = (double)h[j]; if (!whole(w[j])) return REJ_NONWHOLE; }
  if (w[0] != c.magic) return REJ_MAGIC;
  if (w[6] != c.version_word) return REJ_VERSION_WORD;
  double p = w[2], e = w[3], nn = w[4], ss = w[5];
  if (!(p >= 1 && p <= (double)c.cap_body && e >= 0 && e <= (double)c.cap_entity && nn >= 0 && ss >= 0)) return REJ_COUNTS;
  // (frame_floats; n / s past the byte budget on their own: 'size' as in Python, whatever their size)
  if (nn > (double)c.frame_bytes || ss > (double)c.frame_bytes) return REJ_SIZE;
  i64 want = c.fixed+(i64)p*c.body+(i64)e*c.ef+(i64)nn*c.nf+(i64)ss*c.sf;
  if (w[1] != (double)want || want*4 > c.frame_bytes || n/4 != want) return REJ_SIZE;
  cnt[0] = (i64)p; cnt[1] = (i64)e; cnt[2] = (i64)nn; cnt[3] = (i64)ss;
  return REJ_OK;
}

struct CheckOut { int code = REJ_OK; i64 record = -1, trunc_n = 0, trunc_s = 0, e_bucket = -1; };

// reject_code + check_batch: headers [b, hstride] float32, versions / sizes [b] (as doubles), p0 [b, 2] float32 or
// null, p_max (has_p_max false: none); counts [b, 4] out (defined where the batch is accepted)
static inline CheckOut check_batch(const V3 &c, const float *headers, i64 hstride, const double *versions,
                                   const double *sizes, const float *p0, i64 b, bool has_p_max, i64 p_max,
                                   i64 *counts) {
  CheckOut o;
  for (i64 i = 0; i < b; i++) {
    int r = check_row(c, headers+i*hstride, versions[i], sizes[i], counts+4*i);
    if (r) { o.code = r; o.record = i; return o; }
  }
  if (p0)
    for (i64 i = 0; i < b; i++)
      if (!(p0[2*i] == (float)c.main_role && p0[2*i+1] == 1.0f)) { o.code = REJ_P0; o.record = i; return o; }
  i64 emax = 0;
  for (i64 i = 0; i < b; i++) {
    if (has_p_max && counts[4*i] > p_max) { o.code = REJ_P_MAX; o.record = -1; return o; }
    if (counts[4*i+1] > emax) emax = counts[4*i+1];
    o.trunc_n += counts[4*i+2] > c.n_max;
    o.trunc_s += counts[4*i+3] > c.s_max;
  }
  if (b > 0) {
    o.e_bucket = -2;   // (past the largest bucket: never after the counts check, cap_entity being the largest)
    for (i64 eb : c.e_buckets) if (emax <= eb) { o.e_bucket = eb; break; }
  }
  return o;
}

// ---------------------------------------------------------------- common random numbers (tok_policy.crn_uniforms)
static inline uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27))*0x94D049BB133111EBULL; return z ^ (z >> 31);
}
static inline void crn_row(i64 key, i64 t, i64 columns, float *out) {
  const uint64_t G = 0x9E3779B97F4A7C15ULL;
  uint64_t seed = mix64((uint64_t)key*G+0x632BE59BD9B4E019ULL);
  for (i64 col = 0; col < columns; col++) {
    uint64_t ctr = ((uint64_t)t << 8)+(uint64_t)col+1;
    uint64_t z = mix64(seed+ctr*G);
    out[col] = (float)(((double)(z >> 41)+0.5)/8388608.0);
  }
}

// ---------------------------------------------------------------- the fold staging (tok_policy.FoldLayout / fold_pack)
struct Fold {
  i64 n = 0, nh = 0, H = 0, cw = 0, dest_at = 0, row_bytes = 0;
  bool mem = false, t_on = false, crn = false;
  std::vector<i64> sizes, rsizes;
  i64 off_e = 0, per_e = 0, end_e = 0;           // ROW 'ent': offset, bytes per entity row, end
  i64 off_n_ent = 0, off_crn = -1, off_t = 0;     // ROW fields read here (n_ent int32, crn int64 (-1: none), t int32)
  i64 stage_w() const { return row_bytes+8*nh; }
};

struct FoldOut { i64 g = 0, r = 0, graph = 0, crn_rows = 0; };

static inline i64 rd_i32(const uint8_t *p) { int32_t v; std::memcpy(&v, p, 4); return v; }
static inline i64 rd_i64(const uint8_t *p) { int64_t v; std::memcpy(&v, p, 8); return v; }
template <class T> static inline void wr(uint8_t *p, T v) { std::memcpy(p, &v, sizeof v); }

// first index of sizes >= x (numpy searchsorted, side left); sizes.size() when none
static inline i64 search_left(const std::vector<i64> &v, i64 x) {
  i64 k = 0; while (k < (i64)v.size() && v[k] < x) k++; return k;
}

// rows [k, row_bytes] (each record's ROW bytes from its start; prefix > 0: only the first `prefix[i]` bytes of record i
// past the entity span are copied -- the driver's frame-prefix rule; null: whole rows, as fold_pack), pending [k, nh]
// int64, widx [k] (mem / t_on), reset [k] (mem; null: none), dest [k] (t_on with a rollout; null: none),
// ukey / ut [k] (null: none), eb (< 0: none); stage [n, stage_w] and ctrl [n, cw] are written in place.
// Returns g / r / graph / crn_rows; -1 in g: n_ent past the largest entity graph (an error for the caller).
static inline FoldOut fold_pack(const Fold &L, const uint8_t *rows, i64 k, const i64 *pending, const i64 *widx,
                                const uint8_t *reset, const i64 *dest, const i64 *ukey, const i64 *ut, i64 eb,
                                uint8_t *stage, uint8_t *ctrl, const i64 *prefix = nullptr) {
  FoldOut o;
  i64 most = 0;
  for (i64 i = 0; i < k; i++) { i64 ne = rd_i32(rows+i*L.row_bytes+L.off_n_ent); if (i == 0 || ne > most) most = ne; }
  i64 g = search_left(L.sizes, most);
  if (g >= (i64)L.sizes.size()) { o.g = -1; return o; }
  i64 cut = L.off_e+L.per_e*L.sizes[g];
  if (cut > L.row_bytes) cut = L.row_bytes;
  const i64 SW = L.stage_w();
  for (i64 i = 0; i < k; i++) {
    const uint8_t *src = rows+i*L.row_bytes;
    uint8_t *dst = stage+i*SW;
    std::memcpy(dst, src, (size_t)cut);
    if (L.end_e < L.row_bytes) {
      i64 m = L.row_bytes-L.end_e;
      if (prefix && prefix[i] >= 0 && prefix[i] < m) m = prefix[i];
      std::memcpy(dst+L.end_e, src+L.end_e, (size_t)m);
    }
    std::memcpy(dst+L.row_bytes, pending+i*L.nh, (size_t)(8*L.nh));
  }
  for (i64 i = 0; i < L.n; i++) {
    uint8_t *c = ctrl+i*L.cw;
    if (L.mem || L.t_on) wr<int64_t>(c, i < k ? widx[i] : widx[0]);
    if (L.mem && i < k) wr<float>(c+8, (reset && reset[i]) ? 0.0f : 1.0f);
    if (dest && i < k) wr<int64_t>(c+L.dest_at, dest[i]);
    wr<float>(c+12, i < k ? 0.0f : 1.0f);   // dup
    wr<float>(c+16, 0.0f);                    // cf (all rows cleared first)
  }
  std::vector<float> u((size_t)L.H);
  if (L.crn && L.off_crn >= 0)   // (Phase B2: a record with a crn key: its uniforms from it)
    for (i64 i = 0; i < k; i++) {
      const uint8_t *src = rows+i*L.row_bytes;
      i64 key = rd_i64(src+L.off_crn);
      if (!key) continue;
      crn_row(key, rd_i32(src+L.off_t), L.H, u.data());
      std::memcpy(ctrl+i*L.cw+20, u.data(), (size_t)(4*L.H));
      wr<float>(ctrl+i*L.cw+16, 1.0f);
      o.crn_rows++;
    }
  if (ukey)   // (B36: every record's uniforms from its key)
    for (i64 i = 0; i < k; i++) {
      crn_row(ukey[i], ut[i], L.H, u.data());
      std::memcpy(ctrl+i*L.cw+20, u.data(), (size_t)(4*L.H));
      wr<float>(ctrl+i*L.cw+16, 1.0f);
    }
  if (eb >= 0) {   // (v3: the entity bucket's graph)
    g = -1;
    for (i64 j = 0; j < (i64)L.sizes.size(); j++) if (L.sizes[j] == eb) g = j;
    if (g < 0) { o.g = -1; return o; }
  }
  i64 ri = search_left(L.rsizes, k);
  if (ri >= (i64)L.rsizes.size()) { o.g = -1; return o; }
  o.g = g; o.r = L.rsizes[ri]; o.graph = g+ri*(i64)L.sizes.size();
  return o;
}

}  // namespace abptok
