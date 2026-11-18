// visible-v3 native frame (package C, 2026-11-17; included by abp_turbo.c just before obs_init, so it uses the
// native observation's helpers: FN / G0 / G1 / FIELD, the OBS_FUNC_LIST getters, obs_dyncast, vis_*).
//
// Lua: frame = abp_native_v3(map, lasers) -- the v3 frame bytes (schema_v3.py, abp_schema_v3.h) exactly as
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
    X(V3_SECONDARY, 0x781b90, "_ZN13Entity_Player18GetSecondaryPlayerEv") \
    X(V3_KNIFE_FLAGS, 0x780cb0, "_ZNK12Entity_Knife12GetTearFlagsEv") \
    X(V3_TOLSTRING, 0x92de40, "lua_tolstring")

#define V3_ENUM(id, va, name) VF3_##id,
enum { V3_FUNC_LIST(V3_ENUM) VF3_COUNT };
#define V3_ENTRY(id, va, name) [VF3_##id] = {(uintptr_t)va, name},
static const struct { uintptr_t va; const char *name; } V3_FUNCS[VF3_COUNT] = { V3_FUNC_LIST(V3_ENTRY) };
#define V3FN(id) ((void *)V3_FUNCS[VF3_##id].va)
#define V3G1(T, id, obj, a) (((T (*)(const void *, int))V3FN(id))((obj), (a)))

static int v3_ready = -1;          // -1 not checked, 0 refused, 1 ready
static char v3_refusal[200];
static const void *ti_grid, *ti_rock, *ti_poop, *ti_tnt, *ti_bomb, *ti_knife;
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
    ti_bomb = dlsym(RTLD_DEFAULT, "_ZTI11Entity_Bomb");
    ti_knife = dlsym(RTLD_DEFAULT, "_ZTI12Entity_Knife");
    if (!ti_grid || !ti_rock || !ti_poop || !ti_tnt || !ti_bomb || !ti_knife) {
        snprintf(v3_refusal, sizeof v3_refusal, "grid / entity typeinfo not found");
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
#define V3_BODY_CAP ABP_V3_CAP_BODY   // (schema_v3.CAP_BODY, 2026-11-18)
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
    // (schema batch 4: player 0's controlled body first; each sub-player right after its own player)
    int n = 0, np = (int)G0(long, G_NUM_PLAYERS, (const void *)r->game);
    for (int i = 0; i < np; i++) {
        for (int pass = 0; pass < 2; pass++) {
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

// Stage 3: the entities (visible_v3_raw's v3_entities; observation_frame_v3's priority cap, E rows, overflow counts).
// A laser's geometry comes from the bridge (abp_native_v3's second argument: laser_record's values per laser, 53 bytes:
// InitSeed u32, circle u8, radius, angle (degrees), length, end x, end y (doubles), TearFlags i64), found by InitSeed;
// a laser missing there hands the frame back.
typedef struct {
    const void *e;
    int prio, idx;
    double dist;
} v3_ent_t;
static v3_ent_t *v3_ents;
static size_t v3_ents_cap;

static int v3_ent_cmp(const void *a_, const void *b_) {
    const v3_ent_t *a = a_, *b = b_;
    if (a->prio != b->prio) return a->prio < b->prio ? -1 : 1;
    if (a->dist != b->dist) return a->dist < b->dist ? -1 : 1;
    return a->idx < b->idx ? -1 : a->idx > b->idx;
}

static const char *const V3_ANIM_MONSTRO[6] = {"Walk", "JumpUp", "JumpDown", "Taunt", "Appear", "Death"};
static const char *const V3_ANIM_BOMB[3] = {"Pulse", "Idle", "Explode"};

// the raw kind of visible_v3_raw (0 other, 1 tear, 2 projectile, 3 laser, 4 bomb, 5 pickup, 6 npc, 7 knife, 8 familiar,
// 9 effect) and its npc flags (for the priority)
static int v3_kind(const void *e, int t, const void **npc_out) {
    const void *npc = (t >= 10 && t < 1000) ? obs_dyncast(e, ti_entity, ti_npc, 0) : NULL;
    *npc_out = npc;
    if (t == 2) return 1;
    if (t == 9) return 2;
    if (t == 7) return 3;
    if (t == 4) return 4;
    if (t == 5) return 5;
    if (t == 8) return 7;
    if (t == 3) return 8;
    if (t == 1000) return 9;
    if (npc) return 6;
    return 0;
}

static int v3_npc_flags(const void *e) {
    int f = 0;
    if (gbool(FN(E_IS_ENEMY), e)) f += 1;
    if (gbool(FN(E_IS_VULNERABLE), e)) f += 2;
    if (vcall_bool(e, VT_IS_BOSS)) f += 4;
    if (vcall_bool(e, VT_CAN_SHUT_DOORS) && !gbool(FN(E_IS_DEAD), e)) f += 8;
    return f;
}

// visible_producer.v3_priority on the raw values
static int v3_prio(int kind, int t, int npc_flags, float cdmg) {
    if (kind == 5 || t == 5 || t == 6 || t == 1022 || t == 1023) return 0;
    if (kind == 6 || (npc_flags & 1)) return 1;
    if (t == 3) return 2;
    if (kind == 1 || kind == 2 || kind == 3 || kind == 4 || kind == 7 || t == 2 || t == 4 || t == 7 || t == 8 || t == 9 ||
        (t == 1000 && cdmg > 0)) return 3;
    return 4;
}

static void v3_words16(float *dst, uint64_t v, int n) {
    for (int k = 0; k < n; k++) dst[k] = (float)((v >> (16 * k)) & 0xffff);
}

// The entities sent, in priority order (at most CAP); their count in *n_out, the dropped per class in overflow[5].
static const char *v3_select(const v3_room *r, const v3_body_t *bodies, int n_bodies, int *n_out, int overflow[5]) {
    uintptr_t list = FIELD(uintptr_t, r->room, ROOM_ENTITIES);
    uint32_t count = FIELD(uint32_t, r->room, ROOM_ENTITY_COUNT);
    if (count > v3_ents_cap) {
        v3_ent_t *b = realloc(v3_ents, count * sizeof *b);
        if (!b) abort();
        v3_ents = b;
        v3_ents_cap = count;
    }
    int n = 0;
    for (uint32_t k = 0; k < count; k++) {
        const void *e = ((const void *const *)list)[k];
        if (!e) continue;
        int t = G0(int, E_TYPE, e);
        if (t == 1 || !gbool(FN(E_VISIBLE), e)) continue;
        float cdmg = G0(float, E_CDMG, e);
        if (t == 1000 && v3_is_decorative((int)G0(uint32_t, E_VARIANT, e)) && cdmg == 0) continue;
        const void *npc;
        int kind = v3_kind(e, t, &npc);
        int f = kind == 6 ? v3_npc_flags(e) : 0;
        vec2f ep = G0(vec2f, E_POSITION, e);
        double best = 0;
        int have = 0;
        for (int b = 0; b < n_bodies; b++) {   // (bodies with a room position only: a sub-player's (0, 0) takes no part)
            if (!bodies[b].in_room) continue;
            vec2f bp = G0(vec2f, E_POSITION, bodies[b].p);
            double dx = (double)ep.x - (double)bp.x, dy = (double)ep.y - (double)bp.y, d = dx * dx + dy * dy;
            if (!have || d < best) best = d;
            have = 1;
        }
        v3_ents[n].e = e; v3_ents[n].prio = v3_prio(kind, t, f, cdmg); v3_ents[n].dist = best; v3_ents[n].idx = n;
        n++;
    }
    qsort(v3_ents, (size_t)n, sizeof *v3_ents, v3_ent_cmp);
    for (int c = 0; c < 5; c++) overflow[c] = 0;
    for (int k = ABP_V3_CAP_ENTITY; k < n; k++) overflow[v3_ents[k].prio]++;
    *n_out = n < ABP_V3_CAP_ENTITY ? n : ABP_V3_CAP_ENTITY;
    return NULL;
}

#define V3_LASER_REC 53
static const unsigned char *v3_lasers;
static size_t v3_lasers_len;

static const char *v3_entity_row(const v3_room *r, const void *e, uint32_t curses, float *E) {
    int t = G0(int, E_TYPE, e);
    const void *npc;
    int kind = v3_kind(e, t, &npc);
    uint32_t variant = G0(uint32_t, E_VARIANT, e), subtype = G0(uint32_t, E_SUBTYPE, e);
    vec2f pos = G0(vec2f, E_POSITION, e), vel = G0(vec2f, E_VELOCITY, e), sm = G0(vec2f, E_SIZEMULTI, e);
    const void *spr = G0(const void *, E_SPRITE, e);
    int identity_known = 1, flags_known = 4;
    uint64_t tear = 0, proj = 0;
    E[ABP_V3_E_TYPE] = (float)t;
    E[ABP_V3_E_VARIANT] = (float)variant;
    E[ABP_V3_E_X] = (float)((double)pos.x - r->ox);
    E[ABP_V3_E_Y] = (float)((double)pos.y - r->oy);
    E[ABP_V3_E_VX] = vel.x; E[ABP_V3_E_VY] = vel.y;
    E[ABP_V3_E_SIZE] = G0(float, E_SIZE, e);
    E[ABP_V3_E_SIZE_MX] = sm.x; E[ABP_V3_E_SIZE_MY] = sm.y;
    E[ABP_V3_E_COLL] = (float)G0(int, E_ECC, e);
    E[ABP_V3_E_GCOLL] = (float)G0(int, E_GCC, e);
    E[ABP_V3_E_CDMG] = G0(float, E_CDMG, e);
    E[ABP_V3_E_SPRITE_FRAME] = (float)G0(int, ANM2_FRAME, spr);
    E[ABP_V3_E_AGE] = (float)G0(int, E_FRAMECOUNT, e);
    E[ABP_V3_E_FLIP_X] = gbool(FN(E_FLIPX), e);
    E[ABP_V3_E_KIND] = (float)(kind + 1);
    if (t == 2) {
        const void *x = obs_dyncast(e, ti_entity, ti_tear, 0);
        if (!x) return "v3: ToTear";
        E[ABP_V3_E_HEIGHT] = G0(float, TEAR_HEIGHT, x);
        E[ABP_V3_E_FALL] = G0(float, TEAR_FALL, x);
        E[ABP_V3_E_SCALE] = G0(float, TEAR_SCALE, x);
        tear = G0(uint64_t, TEAR_FLAGS, x);
        flags_known = 4 | 1;
    } else if (t == 9) {
        const void *x = obs_dyncast(e, ti_entity, ti_projectile, 0);
        if (!x) return "v3: ToProjectile";
        E[ABP_V3_E_HEIGHT] = G0(float, PROJ_HEIGHT, x);
        E[ABP_V3_E_FALL] = G0(float, PROJ_FALL, x);
        E[ABP_V3_E_SCALE] = G0(float, PROJ_SCALE, x);
        proj = G0(uint64_t, PROJ_FLAGS, x);
        flags_known = 4 | 2;
    } else if (t == 4) {
        const void *x = obs_dyncast(e, ti_entity, ti_bomb, -1);
        // (2026-11-18, Codex round 2: the tear flags known only when read, as the Lua side's pcall: 0 and unknown
        // without a bomb)
        if (x) { tear = ((uint64_t (*)(const void *))V3FN(V3_BOMB_FLAGS))(x); flags_known = 4 | 1; }
    } else if (t == 7) {
        uint32_t seed = v3_seed(e);
        const unsigned char *rec = NULL;
        int found = 0;
        for (size_t at = 0; at + V3_LASER_REC <= v3_lasers_len; at += V3_LASER_REC) {
            uint32_t s_;
            memcpy(&s_, v3_lasers + at, 4);
            if (s_ == seed) { rec = v3_lasers + at; found++; }
        }
        if (!rec) return "v3: a laser not in the bridge's list";
        // (2026-11-18, Codex round 2: two lasers of one InitSeed cannot be told apart in the bridge's list -- the
        // Lua side reads each laser itself: handed back, the reason counted)
        if (found > 1) return "v3: two lasers with one InitSeed";
        double radius, angle, length, ex, ey;
        int64_t tf;
        memcpy(&radius, rec + 5, 8); memcpy(&angle, rec + 13, 8); memcpy(&length, rec + 21, 8);
        memcpy(&ex, rec + 29, 8); memcpy(&ey, rec + 37, 8); memcpy(&tf, rec + 45, 8);
        double rad = angle * (3.141592653589793238462643383279502884 / 180.0);   // (Lua's math.rad)
        E[ABP_V3_E_L_COS] = (float)cos(rad); E[ABP_V3_E_L_SIN] = (float)sin(rad);
        E[ABP_V3_E_L_LEN] = (float)length; E[ABP_V3_E_L_RADIUS] = (float)radius;
        if (rec[4]) {   // a circle laser: its end point is its own position
            E[ABP_V3_E_L_CIRCLE] = 1;
            E[ABP_V3_E_L_EX] = (float)((double)pos.x - r->ox); E[ABP_V3_E_L_EY] = (float)((double)pos.y - r->oy);
        } else {
            E[ABP_V3_E_L_EX] = (float)(ex - r->ox); E[ABP_V3_E_L_EY] = (float)(ey - r->oy);
        }
        tear = (uint64_t)tf;
        flags_known = 4 | 1;
    } else if (t == 5) {
        const void *x = obs_dyncast(e, ti_entity, ti_pickup, 0);
        if (!x) return "v3: ToPickup";
        E[ABP_V3_E_PRICE] = (float)G0(int32_t, PICKUP_PRICE, x);
        if ((curses & 64) != 0 && variant == 100) { subtype = 0; identity_known = 0; }   // Curse of the Blind
    } else if (t == 8) {
        const void *x = obs_dyncast(e, ti_entity, ti_knife, -1);
        if (x) { tear = ((uint64_t (*)(const void *))V3FN(V3_KNIFE_FLAGS))(x); flags_known = 4 | 1; }   // (as bombs)
    } else if (kind == 6) {
        int f = v3_npc_flags(e);
        E[ABP_V3_E_NPC_FLAGS] = (float)f;
        if (f & 4) {
            double maxhp = (double)G0(float, E_MAXHP, e);
            if (maxhp > 0) E[ABP_V3_E_BOSS_HP] = (float)((double)G0(float, E_HP, e) / maxhp);
        }
        E[ABP_V3_E_CHAMPION] = (float)(G0(int32_t, NPC_CHAMPION, npc) + 2);
    }
    E[ABP_V3_E_SUBTYPE] = (float)subtype;
    E[ABP_V3_E_IDENTITY_KNOWN] = (float)identity_known;
    E[ABP_V3_E_FLAGS_KNOWN] = (float)flags_known;
    const char *const *names = t == 20 ? V3_ANIM_MONSTRO : t == 4 ? V3_ANIM_BOMB : NULL;
    int n_names = t == 20 ? 6 : t == 4 ? 3 : 0;
    for (int k = 0; k < n_names; k++) {
        if (((char (*)(const void *, const char *))FN(ANM2_PLAYING))(spr, names[k]) ||
            ((char (*)(const void *, const char *))FN(ANM2_FINISHED))(spr, names[k])) {
            E[ABP_V3_E_ANIM] = (float)(k + 1);
            break;
        }
    }
    // the flag words: AB+ tear flags remapped to the canonical numbering; projectile / entity flags masked
    uint64_t canon_lo = 0, canon_hi = 0;
    for (int b = 0; b < ABP_V3_TEAR_AB_BITS; b++) {
        if (!((tear >> b) & 1)) continue;
        int c = abp_v3_tear_ab_to_canon[b];
        if (c < 64) canon_lo |= 1ULL << c; else canon_hi |= 1ULL << (c - 64);
    }
    v3_words16(E + ABP_V3_E_TEAR_W_FIRST, canon_lo, 4);
    v3_words16(E + ABP_V3_E_TEAR_W_FIRST + 4, canon_hi, 4);
    v3_words16(E + ABP_V3_E_PROJ_W_FIRST, proj & ABP_V3_PROJ_VISIBLE_MASK, ABP_V3_E_PROJ_W_N);
    uint64_t ef = ((uint64_t (*)(const void *))V3FN(V3_EFLAGS))(e);
    v3_words16(E + ABP_V3_E_EFLAG_W_FIRST, ef & ABP_V3_ENTITY_VISIBLE_MASK, ABP_V3_E_EFLAG_W_N);
    return NULL;
}

// Stage 4: inventory and slots (visible_v3_raw's v3_items; observation_frame_v3's N / S rows), from the native visible
// block's getters (VIS_FUNC_LIST). Rows into v3_inv / v3_slots first (their counts size the frame).
typedef struct { float v[ABP_V3_NF]; } v3_nrow;
typedef struct { float v[ABP_V3_SF]; } v3_srow;
static v3_nrow *v3_inv;
static v3_srow *v3_slots;
static size_t v3_inv_cap, v3_slots_cap;

static v3_nrow *v3_inv_add(size_t *n) {
    if (*n == v3_inv_cap) {
        size_t c = v3_inv_cap ? v3_inv_cap * 2 : 64;
        v3_nrow *b = realloc(v3_inv, c * sizeof *b);
        if (!b) abort();
        v3_inv = b;
        v3_inv_cap = c;
    }
    memset(&v3_inv[*n], 0, sizeof *v3_inv);
    return &v3_inv[(*n)++];
}

static v3_srow *v3_slot_add(size_t *n, int owner, int family, int slot) {
    if (*n == v3_slots_cap) {
        size_t c = v3_slots_cap ? v3_slots_cap * 2 : 32;
        v3_srow *b = realloc(v3_slots, c * sizeof *b);
        if (!b) abort();
        v3_slots = b;
        v3_slots_cap = c;
    }
    v3_srow *r = &v3_slots[(*n)++];
    memset(r, 0, sizeof *r);
    r->v[ABP_V3_S_OWNER] = (float)owner;
    r->v[ABP_V3_S_FAMILY] = (float)family;
    r->v[ABP_V3_S_SLOT] = (float)slot;
    r->v[ABP_V3_S_ID_KNOWN] = 1;
    r->v[ABP_V3_S_SELECTED_KNOWN] = 1;
    r->v[ABP_V3_S_CHARGE_KNOWN] = 1;
    return r;
}

static const char *v3_items(const v3_room *r, const v3_body_t *bodies, int n_bodies, size_t *n_inv, size_t *n_slots) {
    const void *config = ((const void *(*)(void))VFN(V_ITEMCONFIG))();
    const void *pool = ((const void *(*)(const void *))VFN(V_ITEMPOOL))((const void *)r->game);
    if (!config || !pool) return "v3: no item config or pool";
    const void *colls = ((const void *(*)(const void *))VFN(V_IC_COLLS))(config);
    long size = (long)((*(const char *const *)((const char *)colls + 8) - *(const char *const *)colls) / (ptrdiff_t)sizeof(void *));
    if (size > ABP_V3_ABP_COLLECTIBLES) size = ABP_V3_ABP_COLLECTIBLES;   // (GetCollectibleNum: no bounds check)
    *n_inv = *n_slots = 0;
    for (int b = 0; b < n_bodies; b++) {
        const void *p = bodies[b].p;
        for (long id = 1; id <= size - 1; id++) {
            int count = ((int (*)(const void *, int))VFN(V_NUM_HELD))(p, (int)id);
            if (count > 0) {
                v3_nrow *n = v3_inv_add(n_inv);
                n->v[ABP_V3_N_OWNER] = (float)b; n->v[ABP_V3_N_ID] = (float)id; n->v[ABP_V3_N_COUNT] = (float)count;
                n->v[ABP_V3_N_ID_KNOWN] = 1;
            }
        }
        int active = G0(int, P_ACTIVE, p), charge = G0(int, P_CHARGE, p);
        const void *ac = active != 0 ? ((const void *(*)(const void *, int))VFN(V_IC_COLL))(config, active) : NULL;
        v3_srow *a = v3_slot_add(n_slots, b, 1, 0);
        a->v[ABP_V3_S_ID] = (float)active; a->v[ABP_V3_S_OCCUPIED] = active != 0; a->v[ABP_V3_S_SELECTED] = 1;
        a->v[ABP_V3_S_CHARGE] = (float)charge; a->v[ABP_V3_S_MAX_CHARGE] = ac ? (float)FIELD(int32_t, ac, IC_MAX_CHARGES) : 0;
        a->v[ABP_V3_S_BATTERY] = (float)((int (*)(const void *))VFN(V_BATTERY))(p);
        const char *sec = (const char *)p + PL_SECONDARY;
        int sid = *(const int32_t *)sec;
        if (((char (*)(const void *, int))VFN(V_HAS_COLL))(p, 534) != 0 || sid != 0) {   // (Schoolbag)
            const void *desc = sid != 0 ? ((const void *(*)(const void *, int))VFN(V_IC_COLL))(config, sid) : NULL;
            v3_srow *c = v3_slot_add(n_slots, b, 1, 1);
            c->v[ABP_V3_S_ID] = (float)sid; c->v[ABP_V3_S_OCCUPIED] = sid != 0;
            c->v[ABP_V3_S_CHARGE] = (float)*(const int32_t *)(sec + 4);
            c->v[ABP_V3_S_MAX_CHARGE] = desc ? (float)FIELD(int32_t, desc, IC_MAX_CHARGES) : 0;
            c->v[ABP_V3_S_BATTERY] = (float)*(const int32_t *)(sec + 8);
        }
        int max_tr = ((int (*)(const void *))VFN(V_MAX_TRINKETS))(p);
        if (max_tr > ABP_V3_ABP_TRINKET_SLOTS) max_tr = ABP_V3_ABP_TRINKET_SLOTS;
        for (int slot = 0; slot < max_tr; slot++) {
            int id = ((int (*)(const void *, unsigned))VFN(V_TRINKET))(p, (unsigned)slot);
            v3_srow *t = v3_slot_add(n_slots, b, 2, slot);
            t->v[ABP_V3_S_ID] = (float)(id & 0x7fff); t->v[ABP_V3_S_OCCUPIED] = id != 0;
            t->v[ABP_V3_S_GOLDEN] = (id & 0x8000) != 0; t->v[ABP_V3_S_CHARGE_KNOWN] = 0;
        }
        int max_pk = ((int (*)(const void *))VFN(V_MAX_POCKET))(p);
        if (max_pk > ABP_V3_ABP_POCKET_SLOTS) max_pk = ABP_V3_ABP_POCKET_SLOTS;
        for (int slot = 0; slot < max_pk; slot++) {
            int pill = ((int (*)(const void *, int))VFN(V_PILL))(p, slot), card = ((int (*)(const void *, int))VFN(V_CARD))(p, slot);
            int identified = pill > 0 && pill < ABP_V3_ABP_PILLS && ((char (*)(const void *, int))VFN(V_PILL_ID))(pool, pill) != 0;
            int effect = identified ? ((int (*)(const void *, int))VFN(V_PILL_EFFECT))(pool, pill) + 1 : 0;
            v3_srow *k = v3_slot_add(n_slots, b, card != 0 ? 4 : 3, slot);
            k->v[ABP_V3_S_ID] = (float)(card != 0 ? card : effect);
            k->v[ABP_V3_S_OCCUPIED] = pill != 0 || card != 0;
            k->v[ABP_V3_S_SELECTED] = slot == 0;
            k->v[ABP_V3_S_ID_KNOWN] = pill == 0 || identified;
            k->v[ABP_V3_S_CHARGE_KNOWN] = 0;
            k->v[ABP_V3_S_PILL_COLOUR] = (float)(pill & 0x7ff);
            k->v[ABP_V3_S_GIANT] = (pill & 0x800) != 0;
        }
    }
    return NULL;
}

// Stage 5: the minimap from the bridge's room list (visible_v3_raw's v3_map: the Lua code packs it, 6 bytes a room:
// GridIndex, Shape, the icon code, visited, clear, current), as observation_frame_v3 writes it: the cells the room
// occupies (the header's ABP_V3_MAP_SHAPE_CELLS, from the engine; another shape: its cell 0).
static const char *v3_minimap(const unsigned char *map, size_t len, float *minimap) {
    if (len % 6) return "v3: map block";
    for (size_t at = 0; at < len; at += 6) {
        int gi = map[at], shape = map[at + 1];
        uint32_t w = 0;
        if (v3_bits(&w, 1, ABP_V3_MAP_SHOWN_SHIFT, ABP_V3_MAP_SHOWN_BITS) ||
            v3_bits(&w, map[at + 3], ABP_V3_MAP_VISITED_SHIFT, ABP_V3_MAP_VISITED_BITS) ||
            v3_bits(&w, map[at + 4], ABP_V3_MAP_CLEAR_SHIFT, ABP_V3_MAP_CLEAR_BITS) ||
            v3_bits(&w, map[at + 5], ABP_V3_MAP_CURRENT_SHIFT, ABP_V3_MAP_CURRENT_BITS) ||
            v3_bits(&w, map[at + 2], ABP_V3_MAP_ICON_SHIFT, ABP_V3_MAP_ICON_BITS))
            return "v3: a map bit does not fit";
        int cells[4] = {0, -1, -1, -1};
        if (shape >= 0 && shape < ABP_V3_MAP_SHAPES && ABP_V3_MAP_SHAPE_CELLS[shape][0] >= 0)
            for (int k = 0; k < 4; k++) cells[k] = ABP_V3_MAP_SHAPE_CELLS[shape][k];
        for (int k = 0; k < 4; k++) {
            if (cells[k] < 0) continue;
            int j = gi + cells[k];
            if (j >= 0 && j < ABP_V3_MAP_SIDE * ABP_V3_MAP_SIDE) minimap[j] = (float)w;
        }
    }
    return NULL;
}

// The frame into v3_buf; its size in floats in *n_floats. NULL, or why the Lua block is sent instead.
static const char *v3_build(const unsigned char *map, size_t map_len, size_t *n_floats) {
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
    int e = 0, overflow[5];
    const char *sel = v3_select(&r, bodies, p, &e, overflow);
    if (sel) return sel;
    size_t n_inv = 0, n_slots = 0;
    const char *it = v3_items(&r, bodies, p, &n_inv, &n_slots);
    if (it) return it;
    int n = (int)n_inv, s = (int)n_slots;
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
    h[ABP_V3_H_N] = (float)n; h[ABP_V3_H_S] = (float)s; h[ABP_V3_H_VERSION] = ABP_V3_VERSION_WORD;
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
    for (int i = 0; i < p; i++) {
        float *row = P + (size_t)i * ABP_V3_PF;
        row[ABP_V3_P_OVERFLOW_INTERACTABLE] = (float)overflow[0]; row[ABP_V3_P_OVERFLOW_MONSTER] = (float)overflow[1];
        row[ABP_V3_P_OVERFLOW_FAMILIAR] = (float)overflow[2]; row[ABP_V3_P_OVERFLOW_PROJECTILE] = (float)overflow[3];
        row[ABP_V3_P_OVERFLOW_OTHER] = (float)overflow[4];
    }
    float *E = local + (size_t)p * ABP_V3_LOCAL;
    for (int i = 0; i < e; i++) {
        why = v3_entity_row(&r, v3_ents[i].e, curses, E + (size_t)i * ABP_V3_EF);
        if (why) return why;
    }
    float *N = E + (size_t)e * ABP_V3_EF, *S = N + (size_t)n * ABP_V3_NF;
    for (int i = 0; i < n; i++) memcpy(N + (size_t)i * ABP_V3_NF, v3_inv[i].v, sizeof v3_inv[i].v);
    for (int i = 0; i < s; i++) memcpy(S + (size_t)i * ABP_V3_SF, v3_slots[i].v, sizeof v3_slots[i].v);
    why = v3_minimap(map, map_len, canvas + (size_t)ABP_V3_CANVAS_H * ABP_V3_CANVAS_W * ABP_V3_ROOM_WORDS);
    if (why) return why;
    *n_floats = total;
    return NULL;
}

static int v3_lua(lua_State *L) {
    v3_calls++;
    size_t n = 0, map_len = 0;
    const unsigned char *map = (const unsigned char *)((const char *(*)(lua_State *, int, size_t *))V3FN(V3_TOLSTRING))(L, 1, &map_len);
    if (!map) return obs_fallback(L, "v3: no map block");
    v3_lasers = (const unsigned char *)((const char *(*)(lua_State *, int, size_t *))V3FN(V3_TOLSTRING))(L, 2, &v3_lasers_len);
    if (!v3_lasers) v3_lasers_len = 0;
    const char *why = v3_build(map, map_len, &n);
    if (why) {
        v3_fallbacks++;
        snprintf(v3_last_fallback, sizeof v3_last_fallback, "%s", why);
        return obs_fallback(L, why);
    }
    LPUSHSTR(L, (const char *)v3_buf, n * sizeof *v3_buf);
    return 1;
}
