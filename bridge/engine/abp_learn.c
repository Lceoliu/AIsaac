/* B60: the block learner's host packing in C (bridge/python/isaac_bridge/learn_native.py loads it with ctypes.CDLL,
 * which releases the GIL for the whole call). Every function copies bytes only (no float arithmetic), so its outputs
 * equal the Python reference's byte for byte:
 *   abp_learn_scan    structured_input.check_headers on per-row header addresses: counts (p, p, e, n, s) per row;
 *   abp_learn_gather  decode_frames_compact(packed=True)'s gathers: room, minimap and the five compact lists;
 *   abp_learn_concat  block_learner._fill_merged: pieces concatenated into one destination, a zero tail.
 * (Page gathers stay in Python until the native page store of the cpp-actor line is merged: B60 N3.)
 * Built by learn_native.build() with gcc -O2 on first use (cache keyed by this file's hash). */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define ABP_LEARN_VERSION 6002

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
