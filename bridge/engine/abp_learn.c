/* B60: the block learner's host packing in C (bridge/python/isaac_bridge/learn_native.py loads it with ctypes.CDLL,
 * which releases the GIL for the whole call). Every function copies bytes only (no float arithmetic), so its outputs
 * equal the Python reference's byte for byte:
 *   abp_learn_scan    structured_input.check_headers on per-row header addresses: counts (p, p, e, n, s) per row;
 *   abp_learn_gather  decode_frames_compact(packed=True)'s gathers: room, minimap and the five compact lists;
 *   abp_learn_concat  block_learner._fill_merged: pieces concatenated into one destination, a zero tail;
 *   abp_learn_pages   (B62 N3) PagedStore._gather: page rows by global row id (page = id / rows per page);
 *   abp_learn_reader  (B62 N4) one Reader bucket's index tables (block_learner._prepare_packed's per-chunk loop).
 * Built by learn_native.build() with gcc -O2 on first use (cache keyed by this file's hash). */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define ABP_LEARN_VERSION 6004

int abp_learn_version(void) { return ABP_LEARN_VERSION; }

static int to_count(float x, int64_t limit, int64_t *out) {
    /* numpy astype(int64) of a finite value truncates toward zero; anything outside [0, limit] is invalid here
     * (the reference rejects it: negative counts, or a size check no count above the frame limit can pass) */
    if (!isfinite(x) || x < 0.0f || x > (float)limit) return 0;
    *out = (int64_t)x;
    return 1;
}

/* Header check and counts of m rows. head[r]: address of row r's frame header (32 bytes); schema / size: the
 * records' visible_schema / visible_bytes; avail[r]: bytes readable from head[r]. counts: [m, 5] (p, p, e, n, s).
 * Returns -1 when every row is valid, else the first invalid row (the caller then runs the reference, which raises
 * the reference's error). */
long abp_learn_scan(long m, const uint64_t *head, const int64_t *schema, const int64_t *size, const int64_t *avail,
                    long version, long fixed, long pf, long pl, long ef, long nf, long sf, long frame_bytes,
                    int64_t *counts) {
    for (long r = 0; r < m; r++) {
        if (schema[r] != version || avail[r] < 32) return r;
        float h[6];
        memcpy(h, (const void *)(uintptr_t)head[r], sizeof h);
        int64_t p, e, n, s;
        if (!to_count(h[2], frame_bytes, &p) || !to_count(h[3], frame_bytes, &e) ||
            !to_count(h[4], frame_bytes, &n) || !to_count(h[5], frame_bytes, &s))
            return r;
        int64_t expected = fixed + p * (pf + pl) + e * ef + n * nf + s * sf;
        if (!(h[0] == 61107.0f) || !((double)h[1] == (double)expected) || size[r] != expected * 4 || p <= 0 ||
            e > 128 || size[r] > frame_bytes || size[r] > avail[r])
            return r;
        int64_t *c = counts + 5 * r;
        c[0] = p; c[1] = p; c[2] = e; c[3] = n; c[4] = s;
    }
    return -1;
}

/* The compact decode of m checked rows. data[r]: address of row r's first float (the reference reads floats from
 * floor(offset / 4) * 4 of an arena, from the row start of ROW records); counts: abp_learn_scan's. room [m, room_n],
 * map [m, map_n]; out[k] receives list k's entries [sum counts[:, k], widths[k]] in row order. */
void abp_learn_gather(long m, const uint64_t *data, const int64_t *counts, long header, long room_n, long map_n,
                      long fixed, const int64_t *widths, float *room, float *map, float *out0, float *out1,
                      float *out2, float *out3, float *out4) {
    float *out[5] = {out0, out1, out2, out3, out4};
    int64_t at[5] = {0, 0, 0, 0, 0};
    for (long r = 0; r < m; r++) {
        const char *f = (const char *)(uintptr_t)data[r];
        memcpy(room + r * room_n, f + 4 * header, 4 * room_n);
        memcpy(map + r * map_n, f + 4 * (header + room_n), 4 * map_n);
        int64_t off = fixed;
        for (int k = 0; k < 5; k++) {
            int64_t c = counts[5 * r + k], w = widths[k];
            if (c) memcpy(out[k] + at[k] * w, f + 4 * off, 4 * c * w);
            at[k] += c;
            off += c * w;
        }
    }
}

/* n pieces (address, bytes) one after another into dst, then zero bytes of zeros. */
void abp_learn_concat(long n, const uint64_t *src, const int64_t *nbytes, char *dst, long zero) {
    char *at = dst;
    for (long i = 0; i < n; i++) {
        if (nbytes[i]) memcpy(at, (const void *)(uintptr_t)src[i], nbytes[i]);
        at += nbytes[i];
    }
    if (zero > 0) memset(at, 0, zero);
}

/* Rows ids[i] of a paged table (pages[p]: address of page p, rows_per_page rows of row_bytes) into out. Returns -1,
 * else the first id outside the pages (the caller raises). The pages are never moved or freed while a learner part
 * reads them (B62: page tensors are allocated once, the rows read belong to live snapshots). */
long abp_learn_pages(long n, const int64_t *ids, long rows_per_page, long n_pages, const uint64_t *pages,
                     long row_bytes, char *out) {
    for (long i = 0; i < n; i++) {
        int64_t id = ids[i];
        if (id < 0 || id / rows_per_page >= n_pages) return i;
        const char *pg = (const char *)(uintptr_t)pages[id / rows_per_page];
        memcpy(out + i * row_bytes, pg + (id % rows_per_page) * row_bytes, row_bytes);
    }
    return -1;
}

/* (B62 N4) One Reader bucket's tables, block_learner._prepare_packed's per-chunk loop: for its nb chunks ids[r] (row r),
 * the Reader rows (zidx [Bb, T], filled with n_main; the chunk's decisions at row_start + 0..n-1), the real positions
 * (r*T + 0..n-1, in row order), the prefix K/V rows (kvidx [Bb, Pd], filled with n_ukv; kinv of the chunk's kvs), their
 * steps (kp [Bb, Pd*lat], filled with -1; each kv step repeated lat times), the bank's block-end table (ends [Bb, max_m],
 * filled with -1; the chunk's old blocks' ends then its new blocks', each mt times) and the bank scatter (dst = r*max_m +
 * 0..k-1, src = block value index * mt + 0..mt-1 over the old blocks old_start + 0..n_old-1 then the new ones
 * n_old_total + new_start + 0..complete-1). The caller has filled zidx / kvidx / kp / ends with their pad values.
 * Returns the number of dst / src entries written. */
long abp_learn_reader(long nb, const int64_t *ids, long T, long max_m, long Pd, long lat, long mt, long n_old_total,
                      const int64_t *lens, const int64_t *row_start, const int64_t *at_k, const int64_t *nk,
                      const int64_t *kinv, const int64_t *kv_steps, const int64_t *old_start, const int64_t *n_old,
                      const int64_t *old_ends, const int64_t *new_start, const int64_t *complete,
                      const int64_t *new_ends, int64_t *zidx, int64_t *real, int64_t *kvidx, int64_t *kp,
                      int64_t *ends, int64_t *dst, int64_t *src) {
    long at_real = 0, at_dst = 0;
    for (long r = 0; r < nb; r++) {
        int64_t i = ids[r], n = lens[i];
        for (int64_t j = 0; j < n; j++) {
            zidx[r * T + j] = row_start[i] + j;
            real[at_real++] = r * T + j;
        }
        int64_t k0 = at_k[i];
        for (int64_t j = 0; j < nk[i]; j++) {
            kvidx[r * Pd + j] = kinv[k0 + j];
            for (long l = 0; l < lat; l++) kp[r * Pd * lat + j * lat + l] = kv_steps[k0 + j];
        }
        int64_t t = 0;
        for (int64_t b = 0; b < n_old[i]; b++, t++) {
            int64_t v = old_start[i] + b, e = old_ends[old_start[i] + b];
            for (long q = 0; q < mt; q++) {
                ends[r * max_m + t * mt + q] = e;
                dst[at_dst] = r * max_m + t * mt + q;
                src[at_dst++] = v * mt + q;
            }
        }
        for (int64_t b = 0; b < complete[i]; b++, t++) {
            int64_t v = n_old_total + new_start[i] + b, e = new_ends[new_start[i] + b];
            for (long q = 0; q < mt; q++) {
                ends[r * max_m + t * mt + q] = e;
                dst[at_dst] = r * max_m + t * mt + q;
                src[at_dst++] = v * mt + q;
            }
        }
    }
    return at_dst;
}
