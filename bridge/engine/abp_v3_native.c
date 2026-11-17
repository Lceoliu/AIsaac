// visible-v3 native frame (package C, 2026-11-17; included by abp_turbo.c just before obs_init, so it uses the
// native observation's helpers: FN / G0 / G1 / FIELD, the OBS_FUNC_LIST getters, obs_dyncast, vis_*).
//
// Lua: frame = abp_native_v3() -- the v3 frame bytes (schema_v3.py, abp_schema_v3.h) exactly as
// visible_producer.observation_frame_v3(visible_v3_raw()) gives them, built here from the same engine functions the
// bridge's Lua code calls; or nil, why (the bridge then sends its Lua raw block). Every column number, size, range and
// the decorative list come from the generated header: nothing hand-written twice.
// Stages (each checked against the reference before the next): 1 the room canvas; 2 bodies; 3
// entities; 4 inventory and slots; 5 the minimap. Until a stage exists its section is written empty (counts 0).
#include "abp_schema_v3.h"

#define V3_FUNC_LIST(X) \
    X(V3_GRID_POS, 0x82f3c0, "_ZNK4Room15GetGridPositionEi") \
    X(V3_IN_ROOM, 0x83be00, "_ZNK4Room16IsPositionInRoomERKN4KAGE4Math7Vector2Ef") \
    X(V3_GE_STATE, 0x780d80, "_ZNK10GridEntity8GetStateEv") \
    X(V3_DISTANCE, 0x88a5a0, "_ZN4KAGE4Math7Vector28DistanceERKS1_S3_") \
    X(V3_STAGE_TYPE, 0x7814b0, "_ZNK5Level12GetStageTypeEv") \
    X(V3_CURSES, 0x732540, "_ZNK5Level9GetCursesEv") \
    X(V3_DIFFICULTY, 0x782160, "_ZNK4Game13GetDifficultyEv") \
    X(V3_EFLAGS, 0x7808e0, "_ZNK6Entity14GetEntityFlagsEv") \
    X(V3_INIT_SEED, 0x780990, "_ZNK6Entity11GetInitSeedEv") \
    X(V3_BOMB_FLAGS, 0x780aa0, "_ZNK11Entity_Bomb12GetTearFlagsEv") \
    X(V3_BONE_HEART, 0x6673d0, "_ZNK13Entity_Player11IsBoneHeartEi") \
    X(V3_SECONDARY, 0x781b90, "_ZN13Entity_Player18GetSecondaryPlayerEv")

#define V3_ENUM(id, va, name) VF3_##id,
enum { V3_FUNC_LIST(V3_ENUM) VF3_COUNT };
#define V3_ENTRY(id, va, name) [VF3_##id] = {(uintptr_t)va, name},
static const struct { uintptr_t va; const char *name; } V3_FUNCS[VF3_COUNT] = { V3_FUNC_LIST(V3_ENTRY) };
#define V3FN(id) ((void *)V3_FUNCS[VF3_##id].va)
#define V3G1(T, id, obj, a) (((T (*)(const void *, int))V3FN(id))((obj), (a)))

static int v3_ready = -1;          // -1 not checked, 0 refused, 1 ready
static char v3_refusal[200];
static const void *ti_grid, *ti_rock, *ti_poop, *ti_tnt;
static float *v3_buf;              // the frame being built, float32 little-endian (the host's order)
static size_t v3_cap;
static uint64_t v3_calls, v3_fallbacks;
static char v3_last_fallback[96];

static int v3_check(void) {
    if (obs_ready != 1) {
        snprintf(v3_refusal, sizeof v3_refusal, "the native observation is not ready");
        return 0;
    }
    if (vis_ready < 0) vis_ready = vis_check();   // (stages 2 / 4: the controller and item getters of VIS_FUNC_LIST)
    if (vis_ready != 1) {
        snprintf(v3_refusal, sizeof v3_refusal, "the native visible getters are not ready");
        return 0;
    }
    for (int i = 0; i < VF3_COUNT; i++) {
        Dl_info info;
        if (!dladdr((void *)V3_FUNCS[i].va, &info) || (uintptr_t)info.dli_saddr != V3_FUNCS[i].va || !info.dli_sname ||
            strcmp(info.dli_sname, V3_FUNCS[i].name) != 0) {
            snprintf(v3_refusal, sizeof v3_refusal, "%s not at 0x%lx", V3_FUNCS[i].name, (unsigned long)V3_FUNCS[i].va);
            return 0;
        }
    }
    ti_grid = dlsym(RTLD_DEFAULT, "_ZTI10GridEntity");
    ti_rock = dlsym(RTLD_DEFAULT, "_ZTI15GridEntity_Rock");
    ti_poop = dlsym(RTLD_DEFAULT, "_ZTI15GridEntity_Poop");
    ti_tnt = dlsym(RTLD_DEFAULT, "_ZTI14GridEntity_TNT");
    if (!ti_grid || !ti_rock || !ti_poop || !ti_tnt) {
        snprintf(v3_refusal, sizeof v3_refusal, "grid typeinfo not found");
        return 0;
    }
    return 1;
}

static int v3_is_decorative(int variant) {
    for (int k = 0; k < ABP_V3_DECORATIVE_N; k++)
        if (ABP_V3_DECORATIVE[k] == variant) return 1;
    return 0;
}

// a bit field of a room / minimap word: value << shift, or -1 when it does not fit (the reference raises)
static inline int v3_bits(uint32_t *word, long long v, int shift, int bits) {
    if (v < 0 || v >= (1LL << bits)) return -1;
    *word |= (uint32_t)v << shift;
    return 0;
}
#define W0(word, name, v) v3_bits(&(word), (v), ABP_V3_W0_##name##_SHIFT, ABP_V3_W0_##name##_BITS)
#define W1(word, name, v) v3_bits(&(word), (v), ABP_V3_W1_##name##_SHIFT, ABP_V3_W1_##name##_BITS)

// The room's own facts the stages share.
typedef struct {
    uintptr_t game;
    const void *room, *level;
    double ox, oy;            // the canvas origin: GetGridPosition(0) - (CELL / 2, CELL / 2), as the bridge's doubles
    int gw, gh, n;
} v3_room;

// the hazard effects of visible_v3_raw's v3_cells: visible effects (type 1000) with CollisionDamage > 0
#define V3_HAZARD_CAP 256
static const void *v3_hazards[V3_HAZARD_CAP];

// Stage 1: the room canvas (canvas[16][28][5], zero outside the grid) as visible_producer packs v3_cells / doors.
static const char *v3_canvas(const v3_room *r, float *canvas) {
    const void *room = r->room;
    // the hazard effects (the order does not matter: any one within reach sets the bit)
    int n_haz = 0;
    uintptr_t list = FIELD(uintptr_t, room, ROOM_ENTITIES);
    uint32_t count = FIELD(uint32_t, room, ROOM_ENTITY_COUNT);
    for (uint32_t k = 0; k < count; k++) {
        const void *e = ((const void *const *)list)[k];
        if (!e || G0(int, E_TYPE, e) != 1000 || !gbool(FN(E_VISIBLE), e)) continue;
        if (!(G0(float, E_CDMG, e) > 0)) continue;
        if (n_haz == V3_HAZARD_CAP) return "v3: too many hazard effects";
        v3_hazards[n_haz++] = e;
    }
    uint32_t w0[ABP_V3_ABP_GRID_CELLS], w1[ABP_V3_ABP_GRID_CELLS];
    int gt_[ABP_V3_ABP_GRID_CELLS], gv_[ABP_V3_ABP_GRID_CELLS], gs_[ABP_V3_ABP_GRID_CELLS];
    int cells = r->n < ABP_V3_ABP_GRID_CELLS ? r->n : ABP_V3_ABP_GRID_CELLS;
    if (cells < r->gw * r->gh) return "v3: grid size";
    for (int i = 0; i < cells; i++) {
        vec2f pos = V3G1(vec2f, V3_GRID_POS, room, i);
        int collision = G1(int, R_COLLISION, room, i);
        const void *g = G1(const void *, R_GRID_ENTITY, room, i);
        int gt = 0, gv = 0, gs = 0;
        if (g) {
            gt = G0(int, GE_TYPE, g);
            if (gt == 16 && (int)G0(uint32_t, GE_VARIANT, g) == 7) gt = 15;      // a hidden door: seen as a wall
            else if (gt != 0 && gt != 1) { gv = (int)G0(uint32_t, GE_VARIANT, g); gs = (int)((uint32_t (*)(const void *))V3FN(V3_GE_STATE))(g); }
            else gt = 0;
        }
        int pit = collision == 1;
        int solid = collision != 0 && !pit;
        int destructible = solid && g && (obs_dyncast(g, ti_grid, ti_rock, -1) || obs_dyncast(g, ti_grid, ti_poop, -1) ||
                                          obs_dyncast(g, ti_grid, ti_tnt, -1));
        int hazard = gt == 8 || gt == 9;
        for (int k = 0; k < n_haz && !hazard; k++) {
            vec2f ep = G0(vec2f, E_POSITION, v3_hazards[k]);
            float d = ((float (*)(const vec2f *, const vec2f *))V3FN(V3_DISTANCE))(&pos, &ep);
            if ((double)d <= (double)G0(float, E_SIZE, v3_hazards[k]) + 28.3) hazard = 1;
        }
        int inside = ((char (*)(const void *, const vec2f *, float))V3FN(V3_IN_ROOM))(room, &pos, 0.0f) != 0;
        uint32_t a = 0;
        if (W0(a, INSIDE, inside) || W0(a, COLLISION, collision) || W0(a, SOLID, solid) || W0(a, PIT, pit) ||
            W0(a, DESTRUCTIBLE, destructible) || W0(a, HAZARD, hazard) || W0(a, EXIT, gt == 17 || gt == 18))
            return "v3: a room bit does not fit";
        w0[i] = a;
        w1[i] = 0;
        gt_[i] = gt; gv_[i] = gv; gs_[i] = gs;
    }
    for (int slot = 0; slot < ABP_V3_ABP_DOOR_SLOTS; slot++) {
        const void *d = G1(const void *, R_DOOR, room, slot);
        if (!d || (int)G0(uint32_t, GE_VARIANT, d) == 7) continue;
        vec2f p = G0(vec2f, GE_POSITION, d);
        double cx = floor(((double)p.x - r->ox) / ABP_V3_CELL), cy = floor(((double)p.y - r->oy) / ABP_V3_CELL);
        if (cx < 0 || cx >= r->gw || cy < 0 || cy >= r->gh) continue;
        int i = (int)cy * r->gw + (int)cx;
        int visited = 0, clear = 0;
        const void *behind = G1(const void *, L_ROOM_BY_IDX, r->level, FIELD(int32_t, d, DOOR_TARGET_INDEX));
        if (behind && FIELD(uint32_t, behind, DESC_VISITED) > 0) { visited = 1; clear = gbool(FN(D_CLEAR), behind); }
        // (visible_producer: a door's bits are set on top of the cell's; a second door on the same cell overwrites)
        uint32_t a = w0[i] & ~(((1u << ABP_V3_W0_DOOR_BITS) - 1) << ABP_V3_W0_DOOR_SHIFT) &
                     ~(((1u << ABP_V3_W0_DOOR_OPEN_BITS) - 1) << ABP_V3_W0_DOOR_OPEN_SHIFT) &
                     ~(((1u << ABP_V3_W0_DOOR_LOCKED_BITS) - 1) << ABP_V3_W0_DOOR_LOCKED_SHIFT) &
                     ~(((1u << ABP_V3_W0_DOOR_TARGET_BITS) - 1) << ABP_V3_W0_DOOR_TARGET_SHIFT) &
                     ~(((1u << ABP_V3_W0_VISITED_BITS) - 1) << ABP_V3_W0_VISITED_SHIFT) &
                     ~(((1u << ABP_V3_W0_CLEAR_BITS) - 1) << ABP_V3_W0_CLEAR_SHIFT);
        uint32_t b = 0;
        if (W0(a, DOOR, 1) || W0(a, DOOR_OPEN, gbool(FN(DOOR_OPEN), d)) || W0(a, DOOR_LOCKED, gbool(FN(DOOR_LOCKED), d)) ||
            W0(a, DOOR_TARGET, FIELD(int32_t, d, DOOR_TARGET_TYPE)) || W0(a, VISITED, visited) || W0(a, CLEAR, clear) ||
            W1(b, DOOR_VARIANT, (long long)G0(uint32_t, GE_VARIANT, d)) ||
            W1(b, DOOR_STATE, (long long)((uint32_t (*)(const void *))V3FN(V3_GE_STATE))(d)))
            return "v3: a door bit does not fit";
        w0[i] = a;
        w1[i] = b;
    }
    for (int i = 0; i < r->gw * r->gh; i++) {
        int row = i / r->gw, col = i % r->gw;
        float *c = canvas + ((size_t)row * ABP_V3_CANVAS_W + col) * ABP_V3_ROOM_WORDS;
        c[0] = (float)w0[i]; c[1] = (float)w1[i]; c[2] = (float)gt_[i]; c[3] = (float)gv_[i]; c[4] = (float)gs_[i];
    }
    return NULL;
}

// Stage 2: the bodies (visible_v3_raw's v3_bodies / v3_body; observation_frame_v3's P rows and local patches).
#define V3_BODY_CAP 8
typedef struct { const void *p; int index, role, linked, in_room; } v3_body_t;

static inline uint32_t v3_seed(const void *e) { return ((uint32_t (*)(const void *))V3FN(V3_INIT_SEED))(e); }

static int v3_bodies(const v3_room *r, v3_body_t *b) {
    uint32_t seeds[64];
    int n_seeds = 0;   // the room's player entities (position_known)
    uintptr_t list = FIELD(uintptr_t, r->room, ROOM_ENTITIES);
    uint32_t count = FIELD(uint32_t, r->room, ROOM_ENTITY_COUNT);
    for (uint32_t k = 0; k < count && n_seeds < 64; k++) {
        const void *e = ((const void *const *)list)[k];
        if (e && G0(int, E_TYPE, e) == 1) seeds[n_seeds++] = v3_seed(e);
    }
    int n = 0, np = (int)G0(long, G_NUM_PLAYERS, (const void *)r->game);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < np; i++) {
            const void *p = ((const void *(*)(int))FN(P_GET))(i);
            if (!p) return -1;
            const void *q = pass == 0 ? p : ((const void *(*)(const void *))V3FN(V3_SECONDARY))(p);
            if (!q) continue;
            uint32_t seed = v3_seed(q);
            int seen = 0;
            for (int k = 0; k < n; k++)
                if (v3_seed(b[k].p) == seed) seen = 1;
            if (seen) continue;
            if (n == V3_BODY_CAP) return -1;
            b[n].p = q; b[n].index = pass == 0 ? i : -1; b[n].role = pass == 0 ? 1 : 3; b[n].linked = 0;
            b[n].in_room = 0;
            for (int k = 0; k < n_seeds; k++) if (seeds[k] == seed) b[n].in_room = 1;
            if (pass == 1) {   // linked both ways with its main body
                uint32_t ms = v3_seed(p);
                for (int k = 0; k < n; k++)
                    if (v3_seed(b[k].p) == ms) { b[n].linked = k + 1; b[k].linked = n + 1; }
            }
            n++;
        }
    }
    return n;
}

static void v3_body_row(const v3_room *r, const v3_body_t *bd, int bi, float *P, float *local, const float *canvas,
                        int stage, int stage_type, int room_type, int difficulty, uint32_t curses) {
    static const char *const ANIMS[14] = {"WalkDown", "WalkUp", "WalkLeft", "WalkRight", "HeadDown", "HeadUp", "HeadLeft",
                                          "HeadRight", "Hit", "Death", "Appear", "TeleportUp", "TeleportDown", "Trapdoor"};
    const void *p = bd->p;
    int unknown = (curses & 8) != 0;
    vec2f pos = G0(vec2f, E_POSITION, p), vel = G0(vec2f, E_VELOCITY, p);
    double x = (double)pos.x - r->ox, y = (double)pos.y - r->oy;
    if (bd->in_room) {
        P[ABP_V3_P_X] = (float)x; P[ABP_V3_P_Y] = (float)y;
        P[ABP_V3_P_VX] = vel.x; P[ABP_V3_P_VY] = vel.y;
    }
    P[ABP_V3_P_SIZE] = G0(float, E_SIZE, p);
    int max = G0(int, P_MAXHEARTS, p), soul = G0(int, P_SOUL, p), bone = G0(int, P_BONE, p);
    if (!unknown) {
        P[ABP_V3_P_HEARTS] = (float)G0(int, P_HEARTS, p); P[ABP_V3_P_MAX_HEARTS] = (float)max;
        P[ABP_V3_P_SOUL] = (float)soul; P[ABP_V3_P_BONE] = (float)bone;
        P[ABP_V3_P_ETERNAL] = (float)G0(int, P_ETERNAL, p); P[ABP_V3_P_GOLDEN] = (float)G0(int, P_GOLDEN, p);
    }
    P[ABP_V3_P_EXTRA_LIVES] = (float)G0(int, P_LIVES, p);
    P[ABP_V3_P_COINS] = (float)G0(int, P_COINS, p); P[ABP_V3_P_BOMBS] = (float)G0(int, P_BOMBS, p);
    P[ABP_V3_P_KEYS] = (float)G0(int, P_KEYS, p);
    P[ABP_V3_P_DAMAGE] = FIELD(float, p, PL_DAMAGE);
    P[ABP_V3_P_MAX_FIRE_DELAY] = (float)FIELD(int32_t, p, PL_MAX_FIRE_DELAY);
    P[ABP_V3_P_SHOT_SPEED] = FIELD(float, p, PL_SHOT_SPEED);
    double h = (double)FIELD(float, p, PL_TEAR_HEIGHT);
    P[ABP_V3_P_RANGE] = (float)(h < 0 ? 260 * h / -23.75 : 260);
    P[ABP_V3_P_MOVE_SPEED] = FIELD(float, p, PL_MOVE_SPEED);
    P[ABP_V3_P_LUCK] = FIELD(float, p, PL_LUCK);
    P[ABP_V3_P_CAN_FLY] = FIELD(unsigned char, p, PL_CAN_FLY) != 0;
    P[ABP_V3_P_INVULNERABLE] = G0(int, P_DMG_COOLDOWN, p) > 0;
    P[ABP_V3_P_CONTROLS] = gbool(FN(P_CONTROLS), p);
    P[ABP_V3_P_DEAD] = gbool(FN(E_IS_DEAD), p);   // (the bridge sends the Lua block while a lethal hit is cancelled)
    P[ABP_V3_P_HEAD_DIR] = (float)G0(int, P_HEAD_DIR, p);
    P[ABP_V3_P_FIRE_DIR] = (float)G0(int, P_FIRE_DIR, p);
    P[ABP_V3_P_MOVE_DIR] = (float)G0(int, P_MOVE_DIR, p);
    const void *sprite = G0(const void *, E_SPRITE, p);
    P[ABP_V3_P_SPRITE_FRAME] = (float)G0(int, ANM2_FRAME, sprite);
    if (bd->index >= 0) {   // (abp_player_charge(i): Isaac.GetPlayer(i)'s counter; a sub-player has none: 0, known 0)
        P[ABP_V3_P_WEAPON_CHARGE] = (float)FIELD(int32_t, p, PL_CHARGE);
        P[ABP_V3_P_WEAPON_CHARGE_KNOWN] = 1;
    }
    P[ABP_V3_P_BODY_INDEX] = (float)bi;
    P[ABP_V3_P_LINKED_BODY] = (float)bd->linked;
    P[ABP_V3_P_GRID_W] = (float)r->gw; P[ABP_V3_P_GRID_H] = (float)r->gh;
    P[ABP_V3_P_HEALTH_KNOWN] = unknown ? 0 : 1;
    P[ABP_V3_P_CONTROLLER_KNOWN] = 1;
    int animation = 0;
    for (int k = 0; k < 14 && sprite; k++) {
        if (((char (*)(const void *, const char *))FN(ANM2_PLAYING))(sprite, ANIMS[k]) ||
            ((char (*)(const void *, const char *))FN(ANM2_FINISHED))(sprite, ANIMS[k])) {
            animation = k + 1;
            break;
        }
    }
    P[ABP_V3_P_ANIMATION_KNOWN] = animation != 0;
    P[ABP_V3_P_BONE_SLOTS_KNOWN] = 1;
    P[ABP_V3_P_POSITION_KNOWN] = (float)bd->in_room;
    P[ABP_V3_P_PTYPE] = (float)(G0(int, P_PLAYER_TYPE, p) + 2);
    P[ABP_V3_P_BODY_ROLE] = (float)bd->role;
    P[ABP_V3_P_CONTROLLER] = (float)((int)((unsigned (*)(const void *))VFN(V_CTRL))(p) + 2);
    P[ABP_V3_P_ANIMATION] = (float)animation;
    P[ABP_V3_P_STAGE] = (float)stage; P[ABP_V3_P_STAGE_TYPE] = (float)(stage_type + 1);
    P[ABP_V3_P_ROOM_TYPE] = (float)room_type; P[ABP_V3_P_DIFFICULTY] = (float)(difficulty + 1);
    P[ABP_V3_P_GAME_VERSION] = 1; P[ABP_V3_P_CONTROL_SCHEME] = 0;
    for (int k = 0; k < ABP_V3_P_CURSE_N; k++) P[ABP_V3_P_CURSE_FIRST + k] = (float)((curses >> k) & 1);
    if (!unknown) {
        int black_bits = (soul + 1) / 2 < ABP_V3_ABP_HEART_SLOTS ? (soul + 1) / 2 : ABP_V3_ABP_HEART_SLOTS;
        uint32_t black = (uint32_t)G0(int, P_BLACK, p) & (uint32_t)((1ULL << black_bits) - 1);
        int slots = (max + soul + 1) / 2 + bone;
        if (slots > ABP_V3_ABP_HEART_SLOTS) slots = ABP_V3_ABP_HEART_SLOTS;
        for (int k = 0; k < ABP_V3_P_BLACK_SLOT_N; k++) P[ABP_V3_P_BLACK_SLOT_FIRST + k] = (float)((black >> k) & 1);
        for (int k = 0; k < slots; k++)
            P[ABP_V3_P_BONE_SLOT_FIRST + k] = ((char (*)(const void *, int))V3FN(V3_BONE_HEART))(p, k) != 0;
    }
    for (int k = 0; k <= ABP_V3_ABP_WEAPON_TYPE_MAX; k++)
        P[ABP_V3_P_WEAPON_FIRST + k] = FIELD(unsigned char, p, PL_WEAPON_TYPES + k) != 0;
    if (bd->in_room) {   // the 9 x 9 patch around floor(pos / CELL), clamped to the grid; the canvas beyond is 0
        int col = (int)floor(x / ABP_V3_CELL), row = (int)floor(y / ABP_V3_CELL);
        if (col < 0) col = 0;
        if (col > r->gw - 1) col = r->gw - 1;
        if (row < 0) row = 0;
        if (row > r->gh - 1) row = r->gh - 1;
        for (int dy = 0; dy < ABP_V3_LOCAL_SIDE; dy++)
            for (int dx = 0; dx < ABP_V3_LOCAL_SIDE; dx++) {
                int cy = row - ABP_V3_LOCAL_SIDE / 2 + dy, cx = col - ABP_V3_LOCAL_SIDE / 2 + dx;
                if (cy < 0 || cy >= ABP_V3_CANVAS_H || cx < 0 || cx >= ABP_V3_CANVAS_W) continue;
                memcpy(local + ((size_t)dy * ABP_V3_LOCAL_SIDE + dx) * ABP_V3_ROOM_WORDS,
                       canvas + ((size_t)cy * ABP_V3_CANVAS_W + cx) * ABP_V3_ROOM_WORDS, ABP_V3_ROOM_WORDS * sizeof(float));
            }
    }
}

// The frame into v3_buf; its size in floats in *n_floats. NULL, or why the Lua block is sent instead.
static const char *v3_build(size_t *n_floats) {
    v3_room r;
    r.game = obs_game();
    r.room = r.game ? G0(const void *, G_ROOM, (const void *)r.game) : NULL;
    r.level = r.game ? G0(const void *, G_LEVEL, (const void *)r.game) : NULL;
    if (!r.room || !r.level) return "v3: no room";
    vec2f g0 = V3G1(vec2f, V3_GRID_POS, r.room, 0);
    r.ox = (double)g0.x - 20.0;
    r.oy = (double)g0.y - 20.0;
    r.gw = G0(int, R_WIDTH, r.room);
    r.gh = G0(int, R_HEIGHT, r.room);
    r.n = G0(int, R_SIZE, r.room);
    if (r.gw <= 0 || r.gh <= 0 || r.gw > ABP_V3_CANVAS_W || r.gh > ABP_V3_CANVAS_H) return "v3: grid past the canvas";
    v3_body_t bodies[V3_BODY_CAP];
    int p = v3_bodies(&r, bodies);
    if (p <= 0) return "v3: bodies";
    int e = 0, n = 0, s = 0;   // (stages 3-4)
    size_t total = (size_t)ABP_V3_FIXED + (size_t)p * (ABP_V3_PF + ABP_V3_LOCAL) + (size_t)e * ABP_V3_EF +
                   (size_t)n * ABP_V3_NF + (size_t)s * ABP_V3_SF;
    if (total * 4 > ABP_V3_FRAME_BYTES) return "v3: frame past the byte budget";
    if (total > v3_cap) {
        float *b = realloc(v3_buf, total * sizeof *b);
        if (!b) abort();
        v3_buf = b;
        v3_cap = total;
    }
    memset(v3_buf, 0, total * sizeof *v3_buf);
    float *h = v3_buf;
    h[ABP_V3_H_MAGIC] = ABP_V3_MAGIC; h[ABP_V3_H_SIZE] = (float)total; h[ABP_V3_H_P] = (float)p; h[ABP_V3_H_E] = (float)e;
    h[ABP_V3_H_N] = (float)n; h[ABP_V3_H_S] = (float)s; h[ABP_V3_H_GEN] = ABP_V3_LAYOUT_GEN;
    h[ABP_V3_H_PRODUCER] = ABP_V3_PRODUCER_ABPLUS;
    float *canvas = v3_buf + ABP_V3_HEADER;
    const char *why = v3_canvas(&r, canvas);
    if (why) return why;
    int stage = G0(int, L_STAGE, r.level), stage_type = ((int (*)(const void *))V3FN(V3_STAGE_TYPE))(r.level);
    int difficulty = ((int (*)(const void *))V3FN(V3_DIFFICULTY))((const void *)r.game);
    uint32_t curses = ((uint32_t (*)(const void *))V3FN(V3_CURSES))(r.level);
    float *P = v3_buf + ABP_V3_FIXED, *local = P + (size_t)p * ABP_V3_PF;
    for (int i = 0; i < p; i++)
        v3_body_row(&r, &bodies[i], i, P + (size_t)i * ABP_V3_PF, local + (size_t)i * ABP_V3_LOCAL, canvas, stage,
                    stage_type, G0(int, R_TYPE, r.room), difficulty, curses);
    *n_floats = total;
    return NULL;
}

static int v3_lua(lua_State *L) {
    v3_calls++;
    size_t n = 0;
    const char *why = v3_build(&n);
    if (why) {
        v3_fallbacks++;
        snprintf(v3_last_fallback, sizeof v3_last_fallback, "%s", why);
        return obs_fallback(L, why);
    }
    LPUSHSTR(L, (const char *)v3_buf, n * sizeof *v3_buf);
    return 1;
}
