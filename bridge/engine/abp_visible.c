// B45 (2026-11-09): structured_input.observation_frame in C, for the sampler workers (ctypes; not loaded by the game).
//
// abp_visible_encode writes the visible-v2 frame (schema 2026110702, isaac-visible-prlmens-g-v2) of one lean
// observation byte for byte as structured_input.observation_frame + pack_frame do under NumPy 2 (NEP 50: a Python
// float meets a float32 value as float32; float64 records stay float64): same float32 / float64 operations in the same
// order, the same nearest-cell rule (first minimum), the same entity priority order (np.lexsort) and the same lossless
// canvas packing with its range / integrality check. The caller (isaac_bridge.visible_native) has decoded the lean
// payload (abplus_lean.LeanDecoder: the record arrays are views of the payload, the terrain / grid / map blocks the
// decoder keeps) and passes the arrays; the JSON-only parts (bodies' controller / animation, inventory, slots) come as
// small arrays. Returns the frame's byte count, or a negative code when the reference would raise (the caller then
// calls the reference to raise its exact error):
//   -1 a canvas value outside the lossless schema, -2 no body / more than 128 kept entities (impossible), -3 the frame
//   exceeds out_cap (FRAME_BYTES), -4 a room canvas larger than 16 x 28.
// Build: gcc -O2 -shared -fPIC -ffp-contract=off -o libabp_visible.so abp_visible.c
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PF 80
#define EF 48
#define NF 4
#define SF 13
#define RC 16
#define RH 16
#define RW 28
#define HEADER 8
#define STAGE_INDEX 42
#define CURSE_START 43
#define OVERFLOW_START 75
#define ENT_KEEP 128

// abplus_lean record layouts
#define PLAYER_BYTES 312   // 39 doubles: PLAYER_FIELDS
#define DOOR_BYTES 16      // slot u1, open u1, locked u1, seen u1, x f4, y f4, target_type i4
#define ENTITY_BYTES 148
#define MAP_BYTES 5        // idx, shape, type, display, flags

static inline double rd_d(const unsigned char *p) { double v; memcpy(&v, p, 8); return v; }
static inline float rd_f(const unsigned char *p) { float v; memcpy(&v, p, 4); return v; }
static inline int32_t rd_i(const unsigned char *p) { int32_t v; memcpy(&v, p, 4); return v; }
static inline int64_t rd_q(const unsigned char *p) { int64_t v; memcpy(&v, p, 8); return v; }

// PLAYER_FIELDS indices of observation_frame's P_FIELDS (-1: 'unknown_*', always 0)
static const int P_SRC[35] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              27, 28, 29, 30, 31, 32, 33, -1, -1, -1, 37, 38};
static const float P_SCALES[35] = {520, 280, 10, 10, 20, 12, 12, 12, 65535, 6, 2, 2, 4, 100, 100, 100, 10, 20, 2, 500,
                                   2, 5, 1, 1, 1, 4, 4, 4, 60, 1, 20, 60, 60, 2048, 32};
static const int P_HEALTH[7] = {5, 6, 7, 8, 9, 10, 11};   // hearts, max_hearts, soul, black, bone, eternal, golden
static const int ROOM_BITS[13] = {1, 3, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1};
static const int ROOM_SCALES[13] = {1, 6, 1, 1, 1, 1, 1, 1, 1, 30, 1, 1, 1};
static const int TFLAG_BITS[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 16, 19, 22};

// tok_obs.SHAPE_CELLS (shape -> cell deltas; observation_frame walks (0,) + SHAPE_CELLS.get(shape, ()))
static int shape_cells(int shape, int *out) {
    static const int s4[] = {0, 13}, s6[] = {0, 1}, s8[] = {0, 1, 13, 14}, s9[] = {1, 13, 14}, s10[] = {0, 13, 14},
                     s11[] = {0, 1, 14}, s12[] = {0, 1, 13};
    const int *c = NULL;
    int n = 0;
    switch (shape) {
    case 4: case 5: c = s4; n = 2; break;
    case 6: case 7: c = s6; n = 2; break;
    case 8: c = s8; n = 4; break;
    case 9: c = s9; n = 3; break;
    case 10: c = s10; n = 3; break;
    case 11: c = s11; n = 3; break;
    case 12: c = s12; n = 3; break;
    default: break;
    }
    out[0] = 0;
    for (int i = 0; i < n; i++) out[1 + i] = c[i];
    return 1 + n;
}

// pack_canvas's per-cell word: rint(value * scale) << shift, summed; 0 / error past the lossless schema
// (np.allclose(values, integer, atol=1e-5) with rtol 1e-5; 0 <= integer < 2**bits)
static int pack_word(const float *ch, int stride, const int *bits, const int *scales, int n, float *out) {
    int64_t word = 0;
    int shift = 0;
    for (int k = 0; k < n; k++) {
        double v = (double)ch[k * stride] * (double)scales[k];
        double r = nearbyint(v);
        if (!(fabs(v - r) <= 1e-5 + 1e-5 * fabs(r))) return -1;
        int64_t iv = (int64_t)r;
        if (iv < 0 || iv >= ((int64_t)1 << bits[k])) return -1;
        word += iv << shift;
        shift += bits[k];
    }
    *out = (float)word;
    return 0;
}

// pack_room of a [RC][h][w] canvas (row stride w): for each cell its word and channels 13..15
static int pack_room(const float *x, int h, int w, float *out) {
    for (int iy = 0; iy < h; iy++)
        for (int ix = 0; ix < w; ix++) {
            const float *c = x + iy * w + ix;
            if (pack_word(c, h * w, ROOM_BITS, ROOM_SCALES, 13, out)) return -1;
            out[1] = c[13 * h * w];
            out[2] = c[14 * h * w];
            out[3] = c[15 * h * w];
            out += 4;
        }
    return 0;
}

// observation_frame's cell(x, y): the first cell of the smallest squared distance. f32: the query point is a float32
// value (a door record) and the arithmetic float32 (NEP 50); otherwise float64.
static void nearest(const double *cells, int n_cells, int width, double x, double y, int f32, int *iy, int *ix) {
    int best = 0;
    if (f32) {
        float fx = (float)x, fy = (float)y, bd = 0;
        for (int i = 0; i < n_cells; i++) {
            float dx = (float)cells[10 * i + 1] - fx, dy = (float)cells[10 * i + 2] - fy;
            float d = dx * dx + dy * dy;
            if (i == 0 || d < bd) { bd = d; best = i; }
        }
    } else {
        double bd = 0;
        for (int i = 0; i < n_cells; i++) {
            double dx = cells[10 * i + 1] - x, dy = cells[10 * i + 2] - y;
            double d = dx * dx + dy * dy;
            if (i == 0 || d < bd) { bd = d; best = i; }
        }
    }
    long index = (long)cells[10 * best];
    *iy = (int)(index / width);
    *ix = (int)(index % width);
}

int abp_visible_version(void) { return 2026110702; }

// players: n_players lean player records; room: the lean room tuple's top-left x / y (room[8], room[9]) and the room
// index (room[4]); doors / entities / map: lean records; cells: [n_cells, 10] float64 (terrain['cells']); grid:
// [n_grid, 7] float64 (terrain grid records g[0..6]); width / height: terrain['width' / 'height']; bodies: [n_players, 4]
// float64 (controller, controller_known, animation, animation_known); inv: [n_inv, 4] float32, slots: [n_slots, 13]
// float32 (np.asarray of the JSON lists); curses / stage / capabilities: ext's. out: out_cap bytes.
long abp_visible_encode(const unsigned char *players, int n_players, double room_x, double room_y, long room_idx,
                        const unsigned char *doors, int n_doors, const unsigned char *entities, int n_ent,
                        const unsigned char *map, int n_map, const double *cells, int n_cells, const double *grid,
                        int n_grid, int width, int height, const double *bodies, const float *inv, int n_inv,
                        const float *slots, int n_slots, long curses, long stage, double capabilities,
                        unsigned char *out, long out_cap) {
    if (n_players < 1) return -2;
    if (width > RW || height > RH) return -4;
    if (width < 1) return -4;
    // ---- room canvas [RC][RH][RW]
    static float room[RC * RH * RW];
    memset(room, 0, sizeof room);
#define R(c, y, x) room[((c) * RH + (y)) * RW + (x)]
    for (int i = 0; i < n_cells; i++) {
        const double *c = cells + 10 * i;
        long index = (long)c[0];
        int iy = (int)(index / width), ix = (int)(index % width);
        if (iy < 0 || iy >= RH || ix < 0 || ix >= RW) return -4;
        // (inside, collision / 6, solid, pit, destructible, hazard)
        R(0, iy, ix) = (float)c[4];
        R(1, iy, ix) = (float)(c[3] / 6);
        R(2, iy, ix) = (float)c[8];
        R(3, iy, ix) = (float)c[6];
        R(4, iy, ix) = (float)c[9];
        R(5, iy, ix) = (float)c[7];
    }
    if (n_cells < 1 && (n_doors || n_grid || n_players)) return -4;
    for (int i = 0; i < n_doors; i++) {
        const unsigned char *d = doors + DOOR_BYTES * i;
        int iy, ix;
        nearest(cells, n_cells, width, (double)rd_f(d + 4), (double)rd_f(d + 8), 1, &iy, &ix);
        int seen = d[3];
        R(6, iy, ix) = 1;
        R(7, iy, ix) = (float)d[1];
        R(8, iy, ix) = (float)d[2];
        R(9, iy, ix) = (float)((double)rd_i(d + 12) / 30);
        R(10, iy, ix) = (float)(seen & 1);
        R(11, iy, ix) = (float)((seen >> 1) & 1);
    }
    for (int i = 0; i < n_grid; i++) {
        const double *g = grid + 7 * i;
        int iy, ix;
        nearest(cells, n_cells, width, g[5], g[6], 0, &iy, &ix);
        R(13, iy, ix) = (float)(g[1] / 32);
        R(14, iy, ix) = (float)(g[2] / 256);
        R(15, iy, ix) = (float)(g[3] / 256);
        if (g[1] == 17 || g[1] == 18) R(12, iy, ix) = 1;
    }
    // ---- entities: priority order (np.lexsort((index, distance, priority))[:128]) and per-class overflow
    int known = !(curses & 8);
    long count_all[5] = {0}, count_kept[5] = {0};
    int *prio = NULL, *order = NULL;
    double *dist = NULL;
    if (n_ent > 0) {
        prio = (int *)malloc(sizeof(int) * n_ent);
        order = (int *)malloc(sizeof(int) * n_ent);
        dist = (double *)malloc(sizeof(double) * n_ent);
        if (!prio || !order || !dist) { free(prio); free(order); free(dist); return -2; }
    }
    for (int i = 0; i < n_ent; i++) {
        const unsigned char *e = entities + ENTITY_BYTES * i;
        int type = rd_i(e + 8), kind = e[84], flags = e[85];
        float cdmg = rd_f(e + 72);
        int p = 4;
        if ((kind >= 1 && kind <= 4) || type == 2 || type == 4 || type == 7 || type == 8 || type == 9 ||
            (type == 1000 && cdmg > 0))
            p = 3;
        if (type == 3) p = 2;
        if (kind == 6 || (flags & 1)) p = 1;
        if (kind == 5 || type == 5 || type == 6 || type == 1022 || type == 1023) p = 0;
        prio[i] = p;
        count_all[p]++;
        double ex = rd_d(e + 20), ey = rd_d(e + 28), best = 0;
        for (int k = 0; k < n_players; k++) {
            const unsigned char *pl = players + PLAYER_BYTES * k;
            double dx = ex - rd_d(pl + 8), dy = ey - rd_d(pl + 16);
            double d = dx * dx + dy * dy;
            if (k == 0 || d < best) best = d;   // np.min over players (NaN aside)
        }
        dist[i] = best;
        order[i] = i;
    }
    // insertion sort, stable on (priority, distance) with the index as the last key
    for (int i = 1; i < n_ent; i++) {
        int v = order[i], j = i - 1;
        while (j >= 0 && (prio[order[j]] > prio[v] || (prio[order[j]] == prio[v] && dist[order[j]] > dist[v]))) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = v;
    }
    int n_keep = n_ent < ENT_KEEP ? n_ent : ENT_KEEP;
    for (int i = 0; i < n_keep; i++) count_kept[prio[order[i]]]++;
    // ---- frame layout
    long n_room = RH * RW * 4, n_map_f = 13 * 13;
    // local patches: pack_room of [P][RC][9][9] -> P * 81 * 4 floats
    long size = HEADER + n_room + n_map_f + (long)n_players * PF + (long)n_players * 81 * 4 + (long)n_keep * EF +
           (long)n_inv * NF + (long)n_slots * SF;
    if (size * 4 > out_cap) { free(prio); free(order); free(dist); return -3; }
    float *f = (float *)out;   // the caller's buffer is 4-byte aligned
    float head[HEADER] = {61107, (float)size, (float)n_players, (float)n_keep, (float)n_inv, (float)n_slots, 1, 0};
    memcpy(f, head, sizeof head);
    float *at = f + HEADER;
    if (pack_room(room, RH, RW, at)) { free(prio); free(order); free(dist); return -1; }
    at += n_room;
    // ---- minimap [8][13][13], packed 1 bit per channel
    {
        float mm[8 * 169];
        memset(mm, 0, sizeof mm);
        int deltas[5];
        for (int i = 0; i < n_map; i++) {
            const unsigned char *m = map + MAP_BYTES * i;
            int idx = m[0], kind = m[2], flg = m[4];
            int nd = shape_cells(m[1], deltas);
            for (int k = 0; k < nd; k++) {
                int j = idx + deltas[k];
                if (j < 0 || j >= 169) continue;
                float v[8] = {1, (float)(flg & 1), (float)((flg >> 1) & 1), (float)(idx == room_idx), (float)(kind == 5),
                              (float)(kind == 4), (float)(kind == 2),
                              (float)!(kind == 1 || kind == 2 || kind == 4 || kind == 5)};
                for (int c = 0; c < 8; c++) mm[c * 169 + j] = v[c];
            }
        }
        static const int ones[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        for (int j = 0; j < 169; j++)
            if (pack_word(mm + j, 169, ones, ones, 8, at + j)) { free(prio); free(order); free(dist); return -1; }
        at += n_map_f;
    }
    // ---- bodies [P][PF]
    float *body = at;
    memset(body, 0, sizeof(float) * PF * n_players);
    int *pcell = (int *)malloc(sizeof(int) * 2 * n_players);
    if (!pcell) { free(prio); free(order); free(dist); return -2; }
    for (int i = 0; i < n_players; i++) {
        const unsigned char *pl = players + PLAYER_BYTES * i;
        float values[35];
        for (int k = 0; k < 35; k++) values[k] = P_SRC[k] < 0 ? 0.0f : (float)rd_d(pl + 8 * P_SRC[k]);
        values[0] = values[0] - (float)room_x;
        values[1] = values[1] - (float)room_y;
        if (!known)
            for (int k = 0; k < 7; k++) values[P_HEALTH[k]] = 0;
        float *b = body + PF * i;
        for (int k = 0; k < 35; k++) b[k] = values[k] / P_SCALES[k];
        const double *meta = bodies + 4 * i;
        b[35] = (float)i;
        b[36] = (float)meta[0];
        b[37] = (float)meta[1];
        b[38] = (float)(meta[2] / 32);
        b[39] = (float)meta[3];
        b[40] = (float)known;
        b[41] = (float)capabilities;
        b[STAGE_INDEX] = (float)((double)stage / 16);
        for (int bit = 0; bit < 32; bit++) b[CURSE_START + bit] = (float)((curses >> bit) & 1);
        for (int c = 0; c < 5; c++) b[OVERFLOW_START + c] = (float)((double)(count_all[c] - count_kept[c]) / 128);
        nearest(cells, n_cells, width, rd_d(pl + 8), rd_d(pl + 16), 0, pcell + 2 * i, pcell + 2 * i + 1);
    }
    at += PF * n_players;
    // ---- local patches: padded (4 each side) room[:, iy:iy+9, ix:ix+9], packed per body
    for (int i = 0; i < n_players; i++) {
        float patch[RC * 81];
        int iy = pcell[2 * i], ix = pcell[2 * i + 1];
        for (int c = 0; c < RC; c++)
            for (int y = 0; y < 9; y++)
                for (int x = 0; x < 9; x++) {
                    int ry = iy + y - 4, rx = ix + x - 4;
                    patch[(c * 9 + y) * 9 + x] = (ry >= 0 && ry < RH && rx >= 0 && rx < RW) ? R(c, ry, rx) : 0.0f;
                }
        if (pack_room(patch, 9, 9, at)) { free(pcell); free(prio); free(order); free(dist); return -1; }
        at += 81 * 4;
    }
    free(pcell);
    // ---- entities [E][EF]
    for (int r = 0; r < n_keep; r++) {
        const unsigned char *e = entities + ENTITY_BYTES * order[r];
        float v[32];
        // ef: type variant subtype x y vx vy size size_mx size_my coll gcoll cdmg aframe age kind flags anim flip
        //     champion hp height fall scale l_cos l_sin l_len l_ex l_ey l_radius l_circle price (ENTITY offsets)
        v[0] = (float)rd_i(e + 8); v[1] = (float)rd_i(e + 12); v[2] = (float)rd_i(e + 16);
        v[3] = (float)rd_d(e + 20); v[4] = (float)rd_d(e + 28); v[5] = (float)rd_d(e + 36); v[6] = (float)rd_d(e + 44);
        v[7] = rd_f(e + 52); v[8] = rd_f(e + 56); v[9] = rd_f(e + 60);
        v[10] = (float)rd_i(e + 64); v[11] = (float)rd_i(e + 68); v[12] = rd_f(e + 72);
        v[13] = (float)rd_i(e + 76); v[14] = (float)rd_i(e + 80);
        v[15] = (float)e[84]; v[16] = (float)e[85]; v[17] = (float)e[86]; v[18] = (float)e[87];
        v[19] = (float)rd_i(e + 88); v[20] = rd_f(e + 92); v[21] = rd_f(e + 96); v[22] = rd_f(e + 100);
        v[23] = rd_f(e + 104);
        v[24] = rd_f(e + 116); v[25] = rd_f(e + 120); v[26] = rd_f(e + 124); v[27] = rd_f(e + 128);
        v[28] = rd_f(e + 132); v[29] = rd_f(e + 136); v[30] = (float)rd_i(e + 140); v[31] = (float)rd_i(e + 144);
        int type = rd_i(e + 8), variant = rd_i(e + 12), flags = e[85];
        int identity_known = !((curses & 64) && type == 5 && variant == 100);
        if (!identity_known) v[2] = 0;
        v[3] = (v[3] - (float)room_x) / 520.0f;
        v[4] = (v[4] - (float)room_y) / 280.0f;
        for (int k = 5; k < 10; k++) v[k] = v[k] / 20.0f;
        for (int k = 13; k < 15; k++) v[k] = v[k] / 60.0f;
        v[20] = (flags & 4) ? v[20] : 0.0f;
        for (int k = 21; k < 24; k++) v[k] = v[k] / 30.0f;
        for (int k = 26; k < 30; k++) v[k] = v[k] / 520.0f;
        v[31] = v[31] / 100.0f;
        float *row = at + EF * r;
        memcpy(row, v, sizeof v);
        int64_t word = rd_q(e + 108);   // tflags
        for (int k = 0; k < 15; k++) row[32 + k] = (float)((word >> TFLAG_BITS[k]) & 1);
        row[47] = (float)identity_known;
    }
    at += EF * n_keep;
    free(prio); free(order); free(dist);
    if (n_inv) memcpy(at, inv, sizeof(float) * NF * n_inv);
    at += NF * n_inv;
    if (n_slots) memcpy(at, slots, sizeof(float) * SF * n_slots);
    at += SF * n_slots;
#undef R
    return size * 4;
}

// B46: structured_input.decode_frame_host(rows, packed=True) for the actor's step graph, written straight into its
// fixed-width input slab: n visible-v2 frames (frame r at frames + r * stride, already header-checked by the caller:
// check_frames) into room [n][1792], map [n][169] and per ragged list (body / local share the body count) values
// [n][w][width] + masks [n][w] (uint8), entries past a row's count zero. Returns 0, or -1 when a count exceeds its
// width (the caller's bucket widths are wrong).
int abp_visible_unpack(const unsigned char *frames, long stride, int n, int wb, int we, int wn, int ws, float *room,
                       float *map, float *body, unsigned char *body_mask, float *local, unsigned char *local_mask,
                       float *ent, unsigned char *ent_mask, float *inv, unsigned char *inv_mask, float *slots,
                       unsigned char *slots_mask) {
    static const int RAG_W[5] = {PF, 324, EF, NF, SF};
    for (int r = 0; r < n; r++) {
        const float *f = (const float *)(frames + (long)r * stride);   // frames are 4-byte aligned in ROW
        int counts[5] = {(int)f[2], (int)f[2], (int)f[3], (int)f[4], (int)f[5]};
        memcpy(room + (long)r * 1792, f + HEADER, sizeof(float) * 1792);
        memcpy(map + (long)r * 169, f + HEADER + 1792, sizeof(float) * 169);
        long offset = HEADER + 1792 + 169;
        float *vals[5] = {body, local, ent, inv, slots};
        unsigned char *masks[5] = {body_mask, local_mask, ent_mask, inv_mask, slots_mask};
        int widths[5] = {wb, wb, we, wn, ws};
        for (int k = 0; k < 5; k++) {
            int w = widths[k], c = counts[k], width = RAG_W[k];
            if (c > w) return -1;
            float *v = vals[k] + (long)r * w * width;
            unsigned char *m = masks[k] + (long)r * w;
            if (c) memcpy(v, f + offset, sizeof(float) * width * c);
            if (w > c) memset(v + (long)c * width, 0, sizeof(float) * width * (w - c));
            memset(m, 1, c);
            if (w > c) memset(m + c, 0, w - c);
            offset += (long)c * width;
        }
    }
    return 0;
}
