// abp_turbo: in-process control layer for Afterbirth+ v1.06.T1 Linux (isaac.x64,
// Steam build 22878971, original sha256 4cf33d83...1f0f15). Loaded with LD_PRELOAD;
// patches nothing on disk. Addresses come from the exported .dynsym of that build.
//
// Frame loop (main 0x4ef000): glfwGetTime -> IsaacUpdate -> IsaacRender -> limiter.
// The limiter nanosleeps (1/60 s - elapsed - 1 ms) at 0x4ef281 and then spins on
// glfwGetTime until 1/60 s has passed. Manager::Update runs game logic on every other
// iteration (counter g_Manager+0xf2d0), i.e. 30 logic frames per 60 loop iterations.
//
// ABP_TURBO=1       virtual clock: time reads made by game code (return address inside
//                   the executable's .text) see a frozen virtual CLOCK_MONOTONIC/REALTIME;
//                   it advances by exactly one loop period (16,666,667 ns) when main's
//                   limiter calls nanosleep, which then returns at once. Reads never advance
//                   it. Other threads/libraries (Steam, OpenAL, GL driver) keep real time.
//                   Safety valve: 200000 game reads without a tick (a spin outside the
//                   normal limiter, e.g. video playback) advance one period and are counted.
// ABP_RENDER_EVERY  N>0 renders every Nth loop iteration, 0 never, unset = every iteration.
//                   Implemented by redirecting main's `call IsaacRender` (0x4ef141).
// ABP_STATS=path    appends one counter line per ABP_STATS_MS real milliseconds (default 1000).
// ABP_PROF=path     SIGPROF sampler of the main thread (see prof_* below).
// ABP_FIXED_TIME=N  game-code calls of time() return N (InitRandomNumbers seeds the global
//                   MT19937 behind RandomU32 with it) and the virtual clock starts at N s, so two
//                   runs with the same inputs can be compared frame by frame.
// ABP_NULLGL=1      null drawing: the statically linked libepoxy dispatch pointers of calls that
//                   only produce pixels (draws, clears, sub-image/buffer updates, copies, blits,
//                   flush/finish, glXSwapBuffers) are replaced by a no-op before main(). The window,
//                   GL context, object creation, state, queries and read-backs stay real. Logic that
//                   runs inside render-to-texture passes (e.g. EntityList::Update removing floor
//                   decals after baking them) still runs; only the pixels are skipped.
// ABP_NULLGL_LIST   file of further GL names (no epoxy_ prefix) to replace by the no-op; only
//                   for functions that return void and write no output the game reads.
// ABP_GLCOUNT       GL call census per libepoxy entry, split outside / inside IsaacRender
//                   (see install_gl_hooks).
// getenv("ABP_RESEED:<n>") from game code (the Lua bridge's os.getenv) reseeds the global MT19937
//                   with init_genrand(n) and returns "1"; nothing is looked up. InitRandomNumbers
//                   seeds it only once, at startup, and Monstro's AI (Random/RandomU32/
//                   RandomUnitVector) and the render path draw from it, so an episode reset
//                   needs this to be reproducible. Always active in the game process.
// getenv("ABP_STARTSEED:<n>") from game code arms a start-seed override and returns "1" (nil if the
//                   call site below was not patched). A run started without a seed string (the console's
//                   `restart`) takes its start seed from RandomU32 in Seeds::SetStartSeed(0) (call at
//                   0x863680), and only on a later frame: Manager::StartDebugGame just flags the start.
//                   The rest of the frame of the bridge's ABP_RESEED draws from the MT in between, by an
//                   amount that depends on the previous room, so the start seed and the new player's seeds
//                   followed the instance's history (rl/bridge/abplus/EXPERIMENTS.md A7). While armed,
//                   that call reseeds the MT with init_genrand(n) right before drawing. The next
//                   ABP_RESEED disarms it.
// getenv("ABP_PLAYERTYPE:<n>") from game code (2026-10-08) arms PlayerType n for the next debug-game start and returns
//                   "1" (nil if the call site was not patched): Game::StartDebug's PlayerManager::Init call (0x6dff3f)
//                   passes the constant 0 (Isaac), and the console's `restart <id>` cannot change that in a debug
//                   game; armed, the call passes n instead (one shot). See start_debug_player_init.
// getenv("ABP_SOUND_RESET") from game code sets every sound's replay stamp to -1 and returns "1".
//                   SoundEffects::Play only plays a sound (and draws Random for its variant) if the stamp,
//                   "Manager logic frame of the last play + frame delay", is not in the future. The stamps
//                   outlive episodes, so a sound played shortly before a reset could skip that draw in the
//                   next episode and shift the MT for everything after it (A7: a Mulligan drew 15 vs 16).
// getenv("ABP_TURBO_COUNTERS") from game code returns "reseeds=.. start_patched=.. start_draws=..
//                   start_overrides=.. al_stopped=.. ptype_patched=.. ptype_overrides=..". getenv("ABP_MT") returns the MT19937 index and a hash of its state.
// ABP_AL_STOPPED=1  every OpenAL source reads as stopped to the game: alGetSourcei(AL_SOURCE_STATE) from game code
//                   answers AL_STOPPED. The game plays a sound only if one of its 16 sources is not playing
//                   (KAGE::Sound::Manager::get_sample_source_buffer, 0x482C90), and asks whether a sound still plays
//                   (SoundSourcePlatformBase::IsPlaying, 0x483010); the audio thread plays in real time while the
//                   virtual clock runs the game up to ~100x faster, so these answers followed the machine's timing and
//                   the same seed and actions could end in different states (rl/bridge/abplus/EXPERIMENTS.md A8). With
//                   it every sound counts as finished at once. Off by default.
// getenv("ABP_INPUT_ON:<mask>") from game code switches native input on for the ButtonActions in the bit mask and
//                   returns "1" (nil if the call sites below were not patched); getenv("ABP_INPUT:<held>:<triggered>")
//                   sets which of them are held and which were pressed this frame; getenv("ABP_INPUT_OFF") switches
//                   it off. The engine asks for a player's input through Manager::IsActionPressed / IsActionTriggered
//                   / GetActionValue, which first call LuaEngine::PreActionHookB / PreActionHookF: the MC_INPUT_ACTION
//                   callbacks of the Lua mods, here the bridge's. With native input on, the three calls (0x7d66b5,
//                   0x7d6728, 0x7d67b8) are answered here: for a player entity (Entity::GetType() == 1) and a masked
//                   action with the values the bridge's callback would return (held, pressed this frame, 1.0 / 0.0),
//                   for every other query with "no callback answered", which is what the bridge's callback, the only
//                   one there is (the instances load no mods), returns for them; the engine then reads its devices.
//                   No Lua runs for an input query. The bridge's lean mode uses it: the callback, a Lua call per
//                   query, was the largest part of the game's time left after the per-frame bookkeeping
//                   (rl/bridge/abplus/EXPERIMENTS.md B9).
// getenv("ABP_FORK") from game code (the Lua bridge's `fork` command) clones the running game with fork(): the child
//                   is a second process in exactly the parent's state (entities, projectiles, every RNG, the Lua state
//                   and the virtual clock), the one thing Game::SaveState cannot give inside a room. Returns the child's
//                   pid to the parent ("-1" on failure) and "0" to the child. Only the calling (main) thread exists in
//                   the child; game logic runs on that thread alone (rl/bridge/abplus/EXPERIMENTS.md A19). Everything
//                   the two processes would share is cut off in the child before it returns:
//                     - every socket, pipe, eventfd and epoll fd (the X connection, the Steam pipe, the bridge's sockets)
//                       becomes a private dummy; read-only files are reopened at the same offset (file offsets are
//                       shared across fork otherwise); files open for writing go to /dev/null (stdout/stderr stay);
//                     - OpenAL calls from the game are no-ops (the mixer thread is gone and may have held a lock):
//                       sources read as stopped, as under ABP_AL_STOPPED=1, which the parent should run with;
//                     - SteamAPI_RunCallbacks, XPending, XFlush, XSync are no-ops, window queries answer from the
//                       ABP_X11CACHE cache only, glXSwapBuffers is skipped; the engine's calls into Steam's
//                       interfaces that write or go to the network (KAGE::System::TrophyManager unlock / progress,
//                       LeaderboardManager and UgcManager uploads and downloads: CLONE_STUBS) return 0 at once: they
//                       would talk over the dummy pipe and wait for an answer that never comes;
//                     - exit() becomes _exit() (no atexit handlers: they would shut down Steam, GL and X for both);
//                     - the game's file writes go nowhere: fopen for writing opens /dev/null, remove, rename and
//                       mkdir do nothing (the clone shares the instance's save directory with its parent: the
//                       continue save written at every room change, persistent data at a run's end), system and
//                       popen fail;
//                     - the game's own mutexes (KAGE::System::Mutex and the statically linked libraries': every
//                       pthread mutex initialised from the executable is registered) are all taken by the main thread
//                       before the fork, so no other thread (the sound update thread, the gamepad thread) is inside
//                       a critical section at that moment, and set free in the clone, where those threads do not
//                       exist. Without this about 1 clone in 300 stopped for ever in
//                       KAGE::Sound::ManagerBase::Update on the sound manager's lock (A19). A game lock a clone still
//                       cannot get (one never initialised through pthread_mutex_init) is passed over, not waited for;
//                     - the child ends after ABP_FORK_ALARM real seconds (default 120, 0 = never), so a stuck clone
//                       cannot stay behind (exit status 14; with ABP_FORK_HANG_DIR its stack is written there first); getenv("ABP_ALARM:<s>") in a clone sets the time left anew. A clone is not
//                       tied to the process it was cloned from (a clone of a clone outlives the first): the bridge ends
//                       it when its connection closes, and it stays in the instance's process group.
//                   Run the instance with Mesa's software OpenGL and LP_NUM_THREADS=0: the GL context is then
//                   plain memory of the process and is copied with it. (With the display driver's OpenGL three
//                   seeds gave identical clones too, A19: the exact mode hardly touches the driver. A driver context
//                   used by two processes is still nothing to rely on.)
//                   ABP_FORK_LITE=1 (at launch, or getenv("ABP_FORK_LITE:<0|1>") at run time; off by default,
//                   2026-10-05): a clone with one thread whose registered mutexes are all free forks without the mutex
//                   protocol (fork_lite: same memory in both processes, without its writes; see there).
// getenv("ABP_EXIT") from game code ends a clone at once with _exit(0); it is refused (nil) outside a clone.
// getenv("ABP_FORK_TIMES") (diagnostic, 2026-10-05) returns where this process's fork time went and its own CPU, run-queue
//                   wait and minor faults (field list at the gate); getenv("ABP_MALLOC_INFO[:TRIM]") mallinfo2 (and
//                   malloc_trim first). ABP_FM_COLLECT:<n>:w waits for the clones to exit (their CPU, exit included,
//                   goes to ABP_FORK_TIMES).
// getenv("ABP_FORK_COUNTERS") returns "child=.. pid=.. forks=.. failed=.. live=.. isolated_fds=.. fork_us=.. exit0=..
//                   signalled=.. last_signal=.. other=.." (how the clones this process made and reaped have ended)
//                   and the mutex protocol's counters: registered now, initialised / destroyed so far, dropped (table
//                   full), missed (held by another thread after 400 passes) and self (already this thread's) summed
//                   over the forks, passes that had to wait, locks a clone passed over, file writes a clone sent to
//                   /dev/null.
// getenv("ABP_OBS_INIT") from game code (the Lua bridge, once when it loads) registers two Lua C functions, the native
//                   lean observation: abp_native_lean(logic_frames, extra_flags) returns the fixed part of the bridge's
//                   lean observation (pack_lean: header, room, totals, players, doors, entity records) byte for byte as
//                   the Lua code builds it, read from the engine through the same getters luabridge calls for the Lua
//                   code (or nil, reason when the Lua code must build it: a laser in the room);
//                   abp_native_terrain(logic_frames, force) is the bridge's lean_terrain_changed (same rules, own
//                   state) and also returns the room index and whether the room is clear. Every address is checked
//                   against its exported name first; nil (refused) otherwise. getenv("ABP_OBS_STATUS") returns
//                   "ready=.. calls=.. fallbacks=.. terrain_calls=.." and the reason of a refusal. The Lua code per
//                   decision was 37-42% of a lean decision (rl/bridge/abplus/EXPERIMENTS.md B9).
// Frame-cost work (2026-10-04, rl/bridge/abplus/EXPERIMENTS.md; every part off unless switched on, the default path is
// unchanged):
//   ABP_PROF_STACK=<depth> with ABP_PROF: stack samples (<prof>.stk; analysis/scripts/abplus/prof_stack_report.py);
//                   ABP_PROF_CLONES=<dir>: fork clones keep sampling into <dir>/clone-<pid>.*; getenv("ABP_PROF_FLUSH").
//   getenv("ABP_STUBS:<file>") applies a stub list (ABP_STUB_LIST format) to this process now (one clone of two in the
//                   exactness probe abplus_probe_frame_exact.py); stub_render_h.txt is the list of this round.
//   ABP_FAST=<mask> (start-up) / getenv("ABP_FAST:<mask>"): exact fast paths, see "ABP_FAST" below: 1 Camera::
//                   smooth_samples and smooth_focus_samples in C, 2 decoded-PNG cache (shared by the root and its
//                   clones), 16 access() memo for the game's relative resource paths; 4 / 8 / 32 their check modes (the
//                   engine's result is used and compared). getenv("ABP_FAST_STATUS") gives the counters.
//                   The shared PNG cache region (MAP_SHARED, made at start-up so that clones share it) exists only
//                   when ABP_FAST has bit 2 at start-up (512 MiB) or ABP_PNG_CACHE_MB=<MiB> is set (needed to switch
//                   the cache on at run time in a clone, as abplus_probe_frame_exact.py --env ABP_PNG_CACHE_MB=512).
//   ABP_PU_GATE (default 1): the engine's call of LuaEngine::PostUpdate goes through pu_gate, which answers the calls
//                   the bridge armed (abp_pu_arm / abp_pu_take Lua functions, registered with the native obs; bridge
//                   switch ISAAC_RL_PU_SKIP=1 or ABP_PU.enabled) and passes every other call through.
//                   getenv("ABP_PU_STATUS") gives the counters.
// getenv("ABP_MUTEX_PROBE:<n>") (diagnostic) try-locks every registered game mutex n times and returns "rounds
//                   busy_rounds busy_total sound_registered": how often another thread holds one at a random moment.
// getenv("ABP_GRAB_ON:<every>:<w>:<h>:<path>") / "ABP_GRAB_OFF" / "ABP_GRAB_STATUS": frame grab of the presented frames
//                   into a pipe (footage, abplus_render_replay.py; see grab_gate). Off by default.
//
// Build: gcc -shared -fPIC -O2 -Wall -o libabp_turbo.so abp_turbo.c -ldl

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define GAME_TEXT_LO 0x4ead80UL
#define GAME_TEXT_HI 0x94da04UL
#define GAME_MAIN 0x4ef000UL
#define MAIN_NANOSLEEP_RET 0x4ef286UL
#define MAIN_CALL_RENDER 0x4ef141UL
#define ISAAC_RENDER 0x71dcb0UL
#define G_GAME 0xdbafb0UL
#define G_MANAGER 0xdbc4b0UL
#define GAME_FRAMECOUNT 0x1f5db4UL
#define GAME_FRAMECOUNT2 0x1f5db8UL
#define MANAGER_LOOPCOUNT 0xf2d0UL
#define FRAME_NS 16666667ULL
#define VALVE_READS 200000UL
#define INIT_GENRAND 0x86aa90UL
#define RANDOM_U32 0x51d8e0UL        // RandomU32(): return genrand_int32()
#define START_SEED_CALL 0x863680UL   // Seeds::SetStartSeed(unsigned int), seed 0: call RandomU32
#define MT_STATE 0xdbc740UL          // genrand_int32's state: 624 x uint64
#define MT_INDEX 0xd691dcUL          // and its index (int)
#define MANAGER_SOUNDS 0xf020UL      // Manager: SoundEffects, a std::vector<SoundEffect> (begin, end at +8)
#define SOUND_SIZE 0x110UL           // sizeof(SoundEffect); +4 = frame until which it cannot play again

typedef int (*clock_gettime_fn)(clockid_t, struct timespec *);
typedef int (*gettimeofday_fn)(struct timeval *, void *);
typedef int (*nanosleep_fn)(const struct timespec *, struct timespec *);
typedef time_t (*time_fn)(time_t *);
typedef int (*start_main_fn)(int (*)(int, char **, char **), int, char **,
                             void (*)(void), void (*)(void), void (*)(void), void *);

static clock_gettime_fn real_clock_gettime;
static gettimeofday_fn real_gettimeofday;
static nanosleep_fn real_nanosleep;
static time_fn real_time;
static long long fixed_time = -1;

static int turbo;
static long render_every = 1;
static const char *stats_path;
static volatile uint64_t v_mono_ns;
static int64_t realtime_offset_ns;
static uint64_t ticks, iterations, renders, valve_ticks, reads_since_tick;
static uint64_t real_start_ns, real_last_stats_ns;
static int render_patched;
static volatile uint64_t swaps_skipped;
static volatile uint64_t gl_bank;   // 0 outside IsaacRender, 8 inside (byte offset into census pairs)
static int x11_cache;
static volatile uint64_t x_gwa, x_tc, x_pending, x_cached;
static __thread int is_main_thread;
static int is_child;                // this process is a clone made by ABP_FORK
static int is_child_done;           // child_isolate has run in this process or an ancestor clone
static uint64_t forks, fork_failures, isolated_fds, fork_last_us;
#define MAX_CHILDREN 256
static pid_t children[MAX_CHILDREN];
static int n_children;
static uint64_t clones_exit0, clones_signalled, clones_other;   // how reaped clones ended
static int clone_last_signal;
// getenv("ABP_FORK_TIMES") (diagnostic, 2026-10-05, the teacher's cost breakdown): where the time of this process's
// forks went (ft_*: summed over its own forks; a clone starts from zero) and, in a clone, the time since the fork call
// that made it began in its parent (ft_t0, inherited) and of its own isolation. Nothing the game reads.
static uint64_t ft_n, ft_total_ns, ft_hold_ns, ft_sys_ns, ft_collect_ns, ft_t0, ft_isolate_ns, ft_born_ns;
static uint64_t stats_interval_ns = 1000000000ULL;

// ABP_PROF=path: SIGPROF sampler (1 kHz of process CPU time). Main-thread RIPs are appended
// to path as raw little-endian uint64 on every stats write; path.maps is refreshed with
// /proc/self/maps at each write. The stats line's prof= field gives the sample index at that time.
#define PROF_CAP (1u << 20)
static const char *prof_path;
static uint64_t prof_samples[PROF_CAP];
static volatile uint32_t prof_count, prof_flushed;
static volatile uint64_t prof_other_threads;
// ABP_PROF_STACK=<depth> (with ABP_PROF, diagnostic, 2026-10-04): every main-thread sample also keeps a record of
// depth + 1 words in <path>.stk: the interrupted RIP and up to <depth> return addresses above it (glibc backtrace(),
// the unwinder walks the signal frame with the binaries' .eh_frame; zero-padded). The stats line's stk= field gives the
// record index. ABP_PROF_CLONES=<dir>: a clone made by ABP_FORK keeps sampling into <dir>/clone-<pid>.prof (+ .stk,
// .stats), fresh buffers; otherwise a clone stops sampling.
#define PROF_STACK_MAX 64
static int prof_stack_depth;
static uint64_t *prof_stacks;
static uint32_t prof_stack_cap;
static volatile uint32_t prof_stack_count, prof_stack_flushed;

static uint64_t real_ns(clockid_t id) {
    struct timespec ts;
    real_clock_gettime(id, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline int from_game(void *ret) {
    uintptr_t r = (uintptr_t)ret;
    return r >= GAME_TEXT_LO && r < GAME_TEXT_HI;
}

static void resolve(void) {
    if (!real_clock_gettime) {
        real_clock_gettime = (clock_gettime_fn)dlsym(RTLD_NEXT, "clock_gettime");
        real_gettimeofday = (gettimeofday_fn)dlsym(RTLD_NEXT, "gettimeofday");
        real_nanosleep = (nanosleep_fn)dlsym(RTLD_NEXT, "nanosleep");
        real_time = (time_fn)dlsym(RTLD_NEXT, "time");
    }
}

static uint32_t read_u32(uintptr_t global, uintptr_t field) {
    uintptr_t obj = *(volatile uintptr_t *)global;
    return obj ? *(volatile uint32_t *)(obj + field) : 0;
}

static void prof_flush(void);
static void prof_dump_maps(void);
static void glcount_flush(double real_s);
static int apply_stub_file(const char *path);
static char *fast_apply(unsigned mask);
static char *fast_status(void);

static void write_stats(int force) {
    if (!stats_path) {
        return;
    }
    uint64_t now = real_ns(CLOCK_MONOTONIC);
    if (!force && now - real_last_stats_ns < stats_interval_ns) {
        return;
    }
    real_last_stats_ns = now;
    prof_flush();
    glcount_flush((now - real_start_ns) / 1e9);
    char buf[512];
    int n = snprintf(buf, sizeof buf,
        "real_s=%.3f ticks=%llu iterations=%llu renders=%llu valve_ticks=%llu virtual_s=%.3f "
        "game_frames=%u game_frames2=%u manager_loops=%u turbo=%d render_every=%ld patched=%d "
        "prof=%u prof_other=%llu swaps_skipped=%llu x_gwa=%llu x_tc=%llu x_pending=%llu x_cached=%llu stk=%u\n",
        (now - real_start_ns) / 1e9, (unsigned long long)ticks, (unsigned long long)iterations,
        (unsigned long long)renders, (unsigned long long)valve_ticks,
        turbo ? (double)(int64_t)(v_mono_ns - (fixed_time >= 0 ? (uint64_t)fixed_time * 1000000000ULL : real_start_ns)) / 1e9 : 0.0,
        read_u32(G_GAME, GAME_FRAMECOUNT), read_u32(G_GAME, GAME_FRAMECOUNT2),
        read_u32(G_MANAGER, MANAGER_LOOPCOUNT), turbo, render_every, render_patched,
        (unsigned)prof_count, (unsigned long long)prof_other_threads, (unsigned long long)swaps_skipped,
        (unsigned long long)x_gwa, (unsigned long long)x_tc, (unsigned long long)x_pending,
        (unsigned long long)x_cached, (unsigned)prof_stack_count);
    int fd = open(stats_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        if (write(fd, buf, (size_t)n) != n) {
            perror("ABP_TURBO stats");
        }
        close(fd);
    }
}

static void prof_handler(int sig, siginfo_t *info, void *ctx) {
    (void)sig; (void)info;
    if (!is_main_thread) {
        prof_other_threads++;
        return;
    }
    uint32_t i = prof_count;
    uint64_t rip = (uint64_t)((ucontext_t *)ctx)->uc_mcontext.gregs[REG_RIP];
    if (i < PROF_CAP) {
        prof_samples[i] = rip;
        prof_count = i + 1;
    }
    uint32_t s = prof_stack_count;
    if (prof_stacks && s < prof_stack_cap) {
        void *bt[PROF_STACK_MAX + 8];
        int n = backtrace(bt, prof_stack_depth + 8);
        uint64_t *rec = prof_stacks + (size_t)s * (size_t)(prof_stack_depth + 1);
        int k = 1, found = 0;
        rec[0] = rip;
        for (int j = 0; j < n && k <= prof_stack_depth; j++) {
            if (!found) {
                found = (uint64_t)(uintptr_t)bt[j] == rip;   // the signal frame's IP: the frames above it follow
                continue;
            }
            rec[k++] = (uint64_t)(uintptr_t)bt[j];
        }
        while (k <= prof_stack_depth) rec[k++] = 0;
        prof_stack_count = s + 1;
    }
}

static void prof_flush(void) {
    if (!prof_path) {
        return;
    }
    uint32_t end = prof_count;
    if (end == prof_flushed) {
        return;
    }
    int fd = open(prof_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        size_t bytes = (size_t)(end - prof_flushed) * sizeof(uint64_t);
        if (write(fd, &prof_samples[prof_flushed], bytes) != (ssize_t)bytes) {
            perror("ABP_TURBO prof");
        }
        close(fd);
    }
    prof_flushed = end;
    uint32_t send = prof_stack_count;
    if (prof_stacks && send != prof_stack_flushed) {
        char stk[4096];
        snprintf(stk, sizeof stk, "%s.stk", prof_path);
        int sfd = open(stk, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (sfd >= 0) {
            size_t w = (size_t)(prof_stack_depth + 1) * sizeof(uint64_t);
            const char *p = (const char *)(prof_stacks + (size_t)prof_stack_flushed * (size_t)(prof_stack_depth + 1));
            size_t bytes = (size_t)(send - prof_stack_flushed) * w;
            while (bytes > 0) {
                ssize_t r = write(sfd, p, bytes);
                if (r <= 0) break;
                p += r;
                bytes -= (size_t)r;
            }
            close(sfd);
        }
        prof_stack_flushed = send;
    }
    prof_dump_maps();  // libraries dlopen'ed after start (GL driver) must resolve too
}

static void prof_dump_maps(void) {
    char maps[4096];
    snprintf(maps, sizeof maps, "%s.maps", prof_path);
    int in = open("/proc/self/maps", O_RDONLY), out = open(maps, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    char buf[65536];
    ssize_t n;
    while (in >= 0 && out >= 0 && (n = read(in, buf, sizeof buf)) > 0) {
        if (write(out, buf, (size_t)n) != n) break;
    }
    if (in >= 0) close(in);
    if (out >= 0) close(out);
}

static void prof_start(void) {
    prof_path = getenv("ABP_PROF");
    if (!prof_path) {
        return;
    }
    const char *sd = getenv("ABP_PROF_STACK");
    if (sd && atoi(sd) > 0 && !prof_stacks) {
        prof_stack_depth = atoi(sd) > PROF_STACK_MAX ? PROF_STACK_MAX : atoi(sd);
        prof_stack_cap = (uint32_t)((256u << 20) / ((unsigned)(prof_stack_depth + 1) * 8u));   // 256 MiB of records
        void *m = mmap(NULL, (size_t)prof_stack_cap * (size_t)(prof_stack_depth + 1) * 8, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        prof_stacks = m == MAP_FAILED ? NULL : (uint64_t *)m;
    }
    prof_dump_maps();
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = prof_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, NULL);
    struct itimerval it = {{0, 1000}, {0, 1000}};
    setitimer(ITIMER_PROF, &it, NULL);
}

static void reap_children(void);

static void tick(void) {
    v_mono_ns += FRAME_NS;
    reads_since_tick = 0;
    ticks++;
    if (n_children) {
        reap_children();
    }
    write_stats(0);
}

static void count_read(void) {
    if (!is_main_thread) {
        return;
    }
    if (++reads_since_tick >= VALVE_READS) {
        valve_ticks++;
        tick();
    }
}

int clock_gettime(clockid_t id, struct timespec *ts) {
    resolve();
    if (turbo && from_game(__builtin_return_address(0))
        && (id == CLOCK_MONOTONIC || id == CLOCK_MONOTONIC_RAW || id == CLOCK_REALTIME)) {
        count_read();
        uint64_t t = v_mono_ns + (id == CLOCK_REALTIME ? (uint64_t)realtime_offset_ns : 0);
        ts->tv_sec = (time_t)(t / 1000000000ULL);
        ts->tv_nsec = (long)(t % 1000000000ULL);
        return 0;
    }
    return real_clock_gettime(id, ts);
}

int gettimeofday(struct timeval *tv, void *tz) {
    resolve();
    if (turbo && from_game(__builtin_return_address(0))) {
        count_read();
        uint64_t t = v_mono_ns + (uint64_t)realtime_offset_ns;
        tv->tv_sec = (time_t)(t / 1000000000ULL);
        tv->tv_usec = (suseconds_t)((t % 1000000000ULL) / 1000);
        return 0;
    }
    return real_gettimeofday(tv, tz);
}

time_t time(time_t *t) {
    resolve();
    if (fixed_time >= 0 && from_game(__builtin_return_address(0))) {
        if (t) {
            *t = (time_t)fixed_time;
        }
        return (time_t)fixed_time;
    }
    return real_time(t);
}

// ABP_AL_STOPPED (A8): the game's view of OpenAL source states, see the header.
#define AL_SOURCE_STATE_PARAM 0x1010
#define AL_STOPPED_VALUE 0x1014
typedef void (*al_get_sourcei_fn)(unsigned int, int, int *);
static al_get_sourcei_fn real_al_get_sourcei;
static int al_stopped;
static uint64_t al_stopped_answers;

void alGetSourcei(unsigned int source, int param, int *value) {
    if (is_child) {
        // No audio in a clone: every source reads as stopped with nothing queued or processed.
        if (value) *value = param == AL_SOURCE_STATE_PARAM ? AL_STOPPED_VALUE : 0;
        return;
    }
    if (!real_al_get_sourcei) {
        real_al_get_sourcei = (al_get_sourcei_fn)dlsym(RTLD_NEXT, "alGetSourcei");
    }
    real_al_get_sourcei(source, param, value);
    if (al_stopped && param == AL_SOURCE_STATE_PARAM && value && from_game(__builtin_return_address(0))) {
        *value = AL_STOPPED_VALUE;
        al_stopped_answers++;
    }
}

// The other OpenAL entries the game imports: passed through in the game itself, no-ops in a clone (ABP_FORK), where the
// mixer thread does not exist and the library's locks may have been held by it at the fork.
#define AL_PASS(name, params, args) \
    void name params { \
        static void (*real) params; \
        if (is_child) return; \
        if (!real) real = (void (*) params)dlsym(RTLD_NEXT, #name); \
        real args; \
    }
AL_PASS(alBufferData, (unsigned int b, int f, const void *d, int n, int hz), (b, f, d, n, hz))
AL_PASS(alcProcessContext, (void *c), (c))
AL_PASS(alDeleteBuffers, (int n, const unsigned int *b), (n, b))
AL_PASS(alDeleteSources, (int n, const unsigned int *b), (n, b))
AL_PASS(alListener3f, (int p, float x, float y, float z), (p, x, y, z))
AL_PASS(alListenerfv, (int p, const float *v), (p, v))
AL_PASS(alSource3f, (unsigned int s, int p, float x, float y, float z), (s, p, x, y, z))
AL_PASS(alSourcef, (unsigned int s, int p, float v), (s, p, v))
AL_PASS(alSourcei, (unsigned int s, int p, int v), (s, p, v))
AL_PASS(alSourcePause, (unsigned int s), (s))
AL_PASS(alSourcePlay, (unsigned int s), (s))
AL_PASS(alSourceStop, (unsigned int s), (s))
AL_PASS(alSourceQueueBuffers, (unsigned int s, int n, const unsigned int *b), (s, n, b))

void alSourceUnqueueBuffers(unsigned int s, int n, unsigned int *b) {
    static void (*real)(unsigned int, int, unsigned int *);
    if (is_child) {
        for (int i = 0; b && i < n; i++) b[i] = 0;
        return;
    }
    if (!real) real = (void (*)(unsigned int, int, unsigned int *))dlsym(RTLD_NEXT, "alSourceUnqueueBuffers");
    real(s, n, b);
}

// Names made in a clone are never given to the library; they only have to be distinct and non-zero.
static unsigned int clone_al_name = 0x7f000000u;

void alGenBuffers(int n, unsigned int *b) {
    static void (*real)(int, unsigned int *);
    if (is_child) {
        for (int i = 0; b && i < n; i++) b[i] = clone_al_name++;
        return;
    }
    if (!real) real = (void (*)(int, unsigned int *))dlsym(RTLD_NEXT, "alGenBuffers");
    real(n, b);
}

void alGenSources(int n, unsigned int *b) {
    static void (*real)(int, unsigned int *);
    if (is_child) {
        for (int i = 0; b && i < n; i++) b[i] = clone_al_name++;
        return;
    }
    if (!real) real = (void (*)(int, unsigned int *))dlsym(RTLD_NEXT, "alGenSources");
    real(n, b);
}

int alGetError(void) {
    static int (*real)(void);
    if (is_child) return 0;
    if (!real) real = (int (*)(void))dlsym(RTLD_NEXT, "alGetError");
    return real();
}

void alGetSourcef(unsigned int s, int p, float *v) {
    static void (*real)(unsigned int, int, float *);
    if (is_child) {
        if (v) *v = 0.0f;
        return;
    }
    if (!real) real = (void (*)(unsigned int, int, float *))dlsym(RTLD_NEXT, "alGetSourcef");
    real(s, p, v);
}

// Steam's callback pump talks to the Steam client over a pipe both processes would share.
void SteamAPI_RunCallbacks(void) {
    static void (*real)(void);
    if (is_child) return;
    if (!real) real = (void (*)(void))dlsym(RTLD_NEXT, "SteamAPI_RunCallbacks");
    if (real) real();
}

// A clone leaves the instance's files alone (header): the save directory belongs to the game it was cloned from.
static uint64_t clone_writes;

static int write_mode(const char *mode) {
    return mode && (strchr(mode, 'w') || strchr(mode, 'a') || strchr(mode, '+'));
}

FILE *fopen(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *);
    if (!real) real = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "fopen");
    if (is_child && write_mode(mode) && from_game(__builtin_return_address(0))) {
        clone_writes++;
        path = "/dev/null";
    }
    return real(path, mode);
}

FILE *fopen64(const char *path, const char *mode) {
    static FILE *(*real)(const char *, const char *);
    if (!real) real = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "fopen64");
    if (is_child && write_mode(mode) && from_game(__builtin_return_address(0))) {
        clone_writes++;
        path = "/dev/null";
    }
    return real(path, mode);
}

int remove(const char *path) {
    static int (*real)(const char *);
    if (is_child && from_game(__builtin_return_address(0))) return 0;
    if (!real) real = (int (*)(const char *))dlsym(RTLD_NEXT, "remove");
    return real(path);
}

int rename(const char *from, const char *to) {
    static int (*real)(const char *, const char *);
    if (is_child && from_game(__builtin_return_address(0))) return 0;
    if (!real) real = (int (*)(const char *, const char *))dlsym(RTLD_NEXT, "rename");
    return real(from, to);
}

int mkdir(const char *path, mode_t mode) {
    static int (*real)(const char *, mode_t);
    if (is_child && from_game(__builtin_return_address(0))) return 0;
    if (!real) real = (int (*)(const char *, mode_t))dlsym(RTLD_NEXT, "mkdir");
    return real(path, mode);
}

int system(const char *command) {
    static int (*real)(const char *);
    if (is_child && from_game(__builtin_return_address(0))) return -1;
    if (!real) real = (int (*)(const char *))dlsym(RTLD_NEXT, "system");
    return real(command);
}

FILE *popen(const char *command, const char *type) {
    static FILE *(*real)(const char *, const char *);
    if (is_child && from_game(__builtin_return_address(0))) return NULL;
    if (!real) real = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "popen");
    return real(command, type);
}

// A clone never runs the atexit handlers: they would shut down Steam, the GL context and the X connection.
void exit(int status) {
    static void (*real)(int);
    if (is_child) _exit(status);
    if (!real) real = (void (*)(int))dlsym(RTLD_NEXT, "exit");
    real(status);
    __builtin_unreachable();
}

int nanosleep(const struct timespec *req, struct timespec *rem) {
    resolve();
    if (turbo && (uintptr_t)__builtin_return_address(0) == MAIN_NANOSLEEP_RET) {
        tick();
        return 0;
    }
    return real_nanosleep(req, rem);
}

typedef char *(*getenv_fn)(const char *);
typedef void (*init_genrand_fn)(unsigned long);
typedef unsigned long (*random_u32_fn)(void);
static getenv_fn real_getenv;
static init_genrand_fn game_init_genrand;  // set in __libc_start_main after checking the export
static uint64_t reseeds;
static int start_patched;                  // Seeds::SetStartSeed's RandomU32 call goes to start_seed_draw
static volatile int start_armed;           // ABP_STARTSEED:<n> until the next ABP_RESEED
static unsigned long start_value;
static uint64_t start_draws, start_overrides;
static char getenv_ok[] = "1";
extern char **environ;

// Seeds::SetStartSeed(0)'s draw. Armed, it reseeds the MT with the ABP_STARTSEED value first, so the start seed
// and all the new run then draws from the MT follow from that value alone.
static unsigned long start_seed_draw(void) {
    start_draws++;
    if (start_armed && game_init_genrand) {
        game_init_genrand(start_value);
        start_overrides++;
    }
    return ((random_u32_fn)RANDOM_U32)();
}

// 2026-10-08 (character randomisation): the player type of a debug game's start. The instances run a debug game
// (--set-stage=1: Manager::StartDebugGame), so every `restart` starts through Game::StartDebug, whose
// PlayerManager::Init(players, ePlayerType, ...) call (at 0x6dff3f) passes the constant 0 (`xor esi, esi`): Isaac,
// whatever the console's `restart <id>` says (that argument only reaches Manager+0x8f95c, which Game::Start reads, not
// Game::StartDebug). getenv("ABP_PLAYERTYPE:<n>") from game code arms n for the next such call (one shot: the call
// disarms it); unarmed, the call passes 0 as before.
#define START_DEBUG_PM_INIT_CALL 0x6dff3fUL   // in Game::StartDebug(eLevelStage, eStageType, std::string)
#define PLAYER_MANAGER_INIT 0x8271d0UL        // PlayerManager::Init(ePlayerType, int)
typedef uint64_t (*pm_init_fn)(void *, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static int ptype_patched;
static volatile int ptype_armed = -1;
static uint64_t ptype_overrides;

// The call's arguments forwarded as they are (rdi players, esi player type, edx, and whatever the other argument
// registers hold), the player type replaced when armed.
static uint64_t start_debug_player_init(void *players, uint64_t ptype, uint64_t a3, uint64_t a4, uint64_t a5,
                                        uint64_t a6) {
    int armed = ptype_armed;
    ptype_armed = -1;
    if (armed >= 0) {
        ptype = (ptype & ~0xffffffffULL) | (uint32_t)armed;
        ptype_overrides++;
    }
    return ((pm_init_fn)PLAYER_MANAGER_INIT)(players, ptype, a3, a4, a5, a6);
}

// ---- ABP_INPUT: see the header ----
#define PRE_ACTION_HOOK_B 0x756950UL   // LuaEngine::PreActionHookB(Entity*, eInputHook, unsigned, bool&)
#define PRE_ACTION_HOOK_F 0x756e60UL   // LuaEngine::PreActionHookF(Entity*, eInputHook, unsigned, float&)
#define CALL_HOOK_PRESSED 0x7d66b5UL   // in Manager::IsActionPressed
#define CALL_HOOK_TRIGGERED 0x7d6728UL // in Manager::IsActionTriggered
#define CALL_HOOK_VALUE 0x7d67b8UL     // in Manager::GetActionValue
#define ENTITY_GET_TYPE 0x780600UL     // Entity::GetType() const
static int input_patched;              // how many of the three call sites go through the functions below
static volatile int native_input;
static uint32_t input_mask, input_held, input_triggered;
static uint64_t input_answers, input_passed;

static inline int native_action(void *entity, unsigned action) {
    return entity && action < 32 && ((input_mask >> action) & 1) && ((int (*)(void *))ENTITY_GET_TYPE)(entity) == 1;
}

static char input_hook_b(void *engine, void *entity, int hook, unsigned action, unsigned char *out) {
    if (!native_input) {
        return ((char (*)(void *, void *, int, unsigned, unsigned char *))PRE_ACTION_HOOK_B)(engine, entity, hook,
                                                                                             action, out);
    }
    if (native_action(entity, action)) {
        *out = ((hook == 0 ? input_held : input_triggered) >> action) & 1;
        input_answers++;
        return 1;
    }
    input_passed++;
    return 0;   // "no callback answered": the engine reads its devices, as after the bridge's callback returns nil
}

static char input_hook_f(void *engine, void *entity, int hook, unsigned action, float *out) {
    if (!native_input) {
        return ((char (*)(void *, void *, int, unsigned, float *))PRE_ACTION_HOOK_F)(engine, entity, hook, action, out);
    }
    if (native_action(entity, action)) {
        *out = ((input_held >> action) & 1) ? 1.0f : 0.0f;
        input_answers++;
        return 1;
    }
    input_passed++;
    return 0;
}

// ---- post-update gate (2026-10-04, frame-cost work): see the header (ABP_PU) ----
#define LUA_POST_UPDATE 0x749760UL      // LuaEngine::PostUpdate(): every mod's MC_POST_UPDATE callback, here the bridge's
#define CALL_POST_UPDATE 0x6dd0faUL     // its one call, in Game::Update
static int pu_patched;
static volatile long pu_skip;           // calls still to answer here (armed by the bridge for a step's inner frames)
static long pu_skipped;                 // answered here since the bridge last took the count
static uint64_t pu_calls, pu_total_skipped, pu_arms;

static void pu_gate(void) {
    pu_calls++;
    if (pu_skip > 0) {
        // What the bridge's callback does on such a frame: count it, count the step's frames down, and on the step's
        // first frame clear the pressed-this-frame actions (state.triggered = {}; push_input -> input_triggered = 0).
        pu_skip--;
        pu_skipped++;
        pu_total_skipped++;
        input_triggered = 0;
        return;
    }
    ((void (*)(void))LUA_POST_UPDATE)();
}

// ---- ABP_FORK: see the header ----

// Every pthread mutex the executable initialises, in creation order. The spin lock guards the table (mutexes are made
// and destroyed by several game threads) and is held across the fork.
#define MAX_GAME_MUTEXES 16384
static pthread_mutex_t *game_mutexes[MAX_GAME_MUTEXES];
static unsigned char game_mutex_got[MAX_GAME_MUTEXES];
static int n_game_mutexes;
static volatile int registry_lock;
static uint64_t mutex_dropped, mutex_inits, mutex_destroys;
static uint64_t fork_mutex_missed, fork_mutex_self, fork_mutex_passes, clone_lock_passed;
static int (*real_mutex_init)(pthread_mutex_t *, const pthread_mutexattr_t *);
static int (*real_mutex_destroy)(pthread_mutex_t *);
static int (*real_mutex_lock)(pthread_mutex_t *);
static int (*real_mutex_trylock)(pthread_mutex_t *);
static int (*real_mutex_unlock)(pthread_mutex_t *);

static void resolve_mutex(void) {
    real_mutex_init = (int (*)(pthread_mutex_t *, const pthread_mutexattr_t *))dlsym(RTLD_NEXT, "pthread_mutex_init");
    real_mutex_destroy = (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_destroy");
    real_mutex_trylock = (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_trylock");
    real_mutex_unlock = (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_unlock");
    real_mutex_lock = (int (*)(pthread_mutex_t *))dlsym(RTLD_NEXT, "pthread_mutex_lock");
}

static void registry_acquire(void) {
    while (__sync_lock_test_and_set(&registry_lock, 1)) {
        sched_yield();
    }
}

static void registry_release(void) {
    __sync_lock_release(&registry_lock);
}

int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *attr) {
    if (!real_mutex_lock) resolve_mutex();
    int r = real_mutex_init(m, attr);
    if (r == 0 && from_game(__builtin_return_address(0))) {
        registry_acquire();
        mutex_inits++;
        if (n_game_mutexes < MAX_GAME_MUTEXES) {
            game_mutexes[n_game_mutexes++] = m;
        } else {
            mutex_dropped++;
        }
        registry_release();
    }
    return r;
}

int pthread_mutex_destroy(pthread_mutex_t *m) {
    if (!real_mutex_lock) resolve_mutex();
    if (from_game(__builtin_return_address(0))) {
        registry_acquire();
        mutex_destroys++;
        for (int i = n_game_mutexes - 1; i >= 0; i--) {
            if (game_mutexes[i] == m) {
                game_mutexes[i] = game_mutexes[--n_game_mutexes];
                break;
            }
        }
        registry_release();
    }
    return real_mutex_destroy(m);
}

int pthread_mutex_lock(pthread_mutex_t *m) {
    if (__builtin_expect(!real_mutex_lock, 0)) resolve_mutex();
    if (__builtin_expect(is_child, 0) && from_game(__builtin_return_address(0))) {
        // One thread: nobody to wait for. A lock that cannot be had was left by a thread that does not exist here.
        if (real_mutex_trylock(m) != 0) clone_lock_passed++;
        return 0;
    }
    return real_mutex_lock(m);
}

// Before the fork (main thread): take every registered mutex, without ever waiting while holding others for long: a
// thread that holds one we want and waits for one we have gets it back when we let go of all of them every 16 passes.
// A mutex this thread already holds (the code above us in the stack, not recursive) is left alone.
static pid_t fork_parent_tid;

static void game_mutexes_hold(void) {
    if (!real_mutex_lock) resolve_mutex();
    registry_acquire();
    fork_parent_tid = (pid_t)syscall(SYS_gettid);
    int n = n_game_mutexes;
    memset(game_mutex_got, 0, (size_t)n);
    for (int pass = 0; pass < 400; pass++) {
        int missing = 0;
        for (int i = 0; i < n; i++) {
            if (game_mutex_got[i]) continue;
            if (real_mutex_trylock(game_mutexes[i]) == 0) {
                game_mutex_got[i] = 1;
            } else if (game_mutexes[i]->__data.__owner == fork_parent_tid) {
                game_mutex_got[i] = 2;   // ours already
            } else {
                missing++;
            }
        }
        if (!missing) break;
        fork_mutex_passes++;
        if (pass % 16 == 15) {
            for (int i = 0; i < n; i++) {
                if (game_mutex_got[i] == 1) {
                    real_mutex_unlock(game_mutexes[i]);
                    game_mutex_got[i] = 0;
                }
            }
        }
        struct timespec ts = {0, 50000};
        real_nanosleep(&ts, NULL);
    }
    for (int i = 0; i < n; i++) {
        if (!game_mutex_got[i]) fork_mutex_missed++;
        else if (game_mutex_got[i] == 2) fork_mutex_self++;
    }
}

static void game_mutexes_release_parent(void) {
    for (int i = 0; i < n_game_mutexes; i++) {
        if (game_mutex_got[i] == 1) real_mutex_unlock(game_mutexes[i]);
    }
    registry_release();
}

// In the clone the thread has a new id, which glibc's recursive and error-checking unlocks compare with the owner
// field, so the mutexes are not unlocked but set to their free state (glibc's layout: lock word, count, owner, users;
// the kind stays). One this thread held before the protocol keeps its hold, under the new id.
static void game_mutexes_release_child(void) {
    pid_t me = (pid_t)syscall(SYS_gettid);
    for (int i = 0; i < n_game_mutexes; i++) {
        pthread_mutex_t *m = game_mutexes[i];
        int recursive_hold = game_mutex_got[i] == 1 && m->__data.__owner == fork_parent_tid && m->__data.__count > 1;
        if (game_mutex_got[i] == 2) {
            m->__data.__owner = me;
        } else if (recursive_hold) {
            m->__data.__count--;
            m->__data.__owner = me;
        } else {
            m->__data.__lock = 0;
            m->__data.__count = 0;
            m->__data.__owner = 0;
            m->__data.__nusers = 0;
        }
    }
    registry_release();
}

static uint64_t ft_reaped_ns;   // CPU of the reaped clones, exit included (getrusage of wait4; ABP_FORK_TIMES)

static void reap_children_wait(int options) {
    int kept = 0;
    for (int i = 0; i < n_children; i++) {
        int status = 0;
        struct rusage ru;
        pid_t r = wait4(children[i], &status, options, &ru);
        if (r > 0) {
            ft_reaped_ns += (uint64_t)ru.ru_utime.tv_sec * 1000000000ULL + (uint64_t)ru.ru_utime.tv_usec * 1000ULL +
                            (uint64_t)ru.ru_stime.tv_sec * 1000000000ULL + (uint64_t)ru.ru_stime.tv_usec * 1000ULL;
        }
        if (r == 0) {
            children[kept++] = children[i];
        } else if (r > 0 && WIFSIGNALED(status)) {
            clones_signalled++;
            clone_last_signal = WTERMSIG(status);
        } else if (r > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            clones_exit0++;
        } else {
            clones_other++;
        }
    }
    n_children = kept;
}

static void reap_children(void) {
    reap_children_wait(WNOHANG);
}

// The clone's watchdog fired: it is stuck (or simply outlived ABP_FORK_ALARM). With ABP_FORK_HANG_DIR set, the main
// thread's stack goes to <dir>/hang-<pid>.txt first, so a deadlock on a lock another thread held at the fork shows up.
static const char *hang_dir;

static void clone_alarm(int sig) {
    (void)sig;
    if (hang_dir) {
        char path[4096];
        snprintf(path, sizeof path, "%s/hang-%d.txt", hang_dir, (int)getpid());
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            void *frames[64];
            int n = backtrace(frames, 64);
            backtrace_symbols_fd(frames, n, fd);
            close(fd);
        }
    }
    _exit(14);
}

// ABP_STACK_DUMP_DIR=<dir> (diagnostic, 2026-10-06, read at start-up; the root's first-build hang): SIGUSR2 makes the
// main thread write the interrupted RIP and its stack (glibc backtrace_symbols_fd: exported names resolve) to
// <dir>/stack-<pid>-<n>.txt. Another thread that receives the signal hands it on to the main thread. The tok workers
// send it to a root whose start-state build failed or stalled, before they kill it (tok_sampler.stall_evidence).
// Unset: no handler is installed. Nothing the game reads.
static const char *stack_dump_dir;
static volatile pid_t stack_main_tid;
static volatile int stack_dump_n;

static void stack_dump_handler(int sig, siginfo_t *info, void *uctx) {
    (void)info;
    if (!is_main_thread) {
        if (stack_main_tid > 0) syscall(SYS_tgkill, getpid(), stack_main_tid, sig);
        return;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/stack-%d-%d.txt", stack_dump_dir, (int)getpid(), stack_dump_n++);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    char head[96];
    int len = snprintf(head, sizeof head, "rip=0x%llx\n",
                       (unsigned long long)((ucontext_t *)uctx)->uc_mcontext.gregs[REG_RIP]);
    if (len > 0 && write(fd, head, (size_t)len) != len) {
        close(fd);
        return;
    }
    void *frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, fd);
    close(fd);
}

static void stack_dump_install(void) {
    const char *d = getenv("ABP_STACK_DUMP_DIR");
    if (!d || !*d) return;
    stack_dump_dir = d;
    stack_main_tid = (pid_t)syscall(SYS_gettid);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = stack_dump_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR2, &sa, NULL);
}

// Game functions that return 0 at once in a clone (header). A line is applied only if the exported symbol at the
// address has exactly this name, so another build is left alone. The patch touches the clone's private copy of the
// code pages.
static const struct { uintptr_t va; const char *name; } CLONE_STUBS[] = {
    {0x887430UL, "_ZN4KAGE6System13TrophyManager16UnlockTrophyByIdEj"},
    {0x8873E0UL, "_ZN4KAGE6System13TrophyManager18UnlockTrophyByNameEPKc"},
    {0x887520UL, "_ZN4KAGE6System13TrophyManager21SetTrophyProgressByIDEjii"},
    {0x8874C0UL, "_ZN4KAGE6System13TrophyManager23SetTrophyProgressByNameEPKcii"},
    {0x885CE0UL, "_ZN4KAGE6System18LeaderboardManager11UploadScoreEPNS0_15LeaderboardDataEiPhPFvbbiiPvES5_"},
    {0x885B70UL, "_ZN4KAGE6System18LeaderboardManager11UploadScoreEPNS0_15LeaderboardDataEiPhmPFvbbiiPvEPFvbS5_ES5_"},
    {0x8860B0UL, "_ZN4KAGE6System18LeaderboardManager20DownloadFriendScoresEPNS0_15LeaderboardDataEPFvPNS0_16LeaderboardEntryEiPvES6_"},
    {0x885F60UL, "_ZN4KAGE6System18LeaderboardManager25DownloadSurroundingScoresEPNS0_15LeaderboardDataEiPFvPNS0_16LeaderboardEntryEiPvES6_"},
    {0x885E20UL, "_ZN4KAGE6System18LeaderboardManager17DownloadRankRangeEPNS0_15LeaderboardDataEiiPFvPNS0_16LeaderboardEntryEiPvES6_"},
    {0x885A40UL, "_ZN4KAGE6System18LeaderboardManager14GetLeaderboardEPKcNS0_22LeaderboardManagerBase9eSortModeEPFvPNS0_15LeaderboardDataEPvES8_"},
    {0x886300UL, "_ZN4KAGE6System18LeaderboardManager9AttachUGCEPNS0_15LeaderboardDataEmPFvbPvES4_"},
    {0x8861E0UL, "_ZN4KAGE6System18LeaderboardManager9AttachUGCEymPFvbPvES2_"},
    {0x887DF0UL, "_ZN4KAGE6System10UgcManager9UploadUGCEPcPvjPFvmbS3_ES3_"},
    {0x887F20UL, "_ZN4KAGE6System10UgcManager11DownloadUGCEmPFvPvjbS2_ES2_"},
    {0x887CD0UL, "_ZN4KAGE6System10UgcManager9AttachUGCEmPNS0_15LeaderboardDataEPFvbPvES4_"},
};
static int clone_stubbed;

static void clone_stub_steam(void) {
    static const unsigned char ret0[3] = {0x31, 0xC0, 0xC3};   // xor eax, eax; ret
    for (size_t i = 0; i < sizeof CLONE_STUBS / sizeof CLONE_STUBS[0]; i++) {
        uintptr_t va = CLONE_STUBS[i].va, page = va & ~0xFFFUL;
        Dl_info info;
        if (!dladdr((void *)va, &info) || (uintptr_t)info.dli_saddr != va || !info.dli_sname ||
            strcmp(info.dli_sname, CLONE_STUBS[i].name) != 0) {
            continue;
        }
        if (mprotect((void *)page, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) continue;
        memcpy((void *)va, ret0, sizeof ret0);
        mprotect((void *)page, 0x2000, PROT_READ | PROT_EXEC);
        clone_stubbed++;
    }
}

// ABP_FM_* (the bridge's fork_many, 2026-10-04): getenv("ABP_FM_OPEN") makes a pipe whose write end the clones made by
// the following ABP_FORKs keep (child_isolate leaves it alone); a clone reports with getenv("ABP_FM_RESULT:<text>")
// (one line, written at once); getenv("ABP_FM_COLLECT:<n>") in the parent closes its write end and reads until n
// lines or until every clone holding the pipe has ended, and returns the lines joined by ';'. getenv("ABP_FM_PROBE")
// answers "1".
static int fm_pipe[2] = {-1, -1};
static int fm_keep_fd = -1;

static char *fm_open(void) {
    if (fm_pipe[0] >= 0) close(fm_pipe[0]);
    if (fm_pipe[1] >= 0) close(fm_pipe[1]);
    fm_pipe[0] = fm_pipe[1] = fm_keep_fd = -1;
    if (pipe(fm_pipe) != 0) return NULL;
    fm_keep_fd = fm_pipe[1];
    return "1";
}

static char *fm_result(const char *text) {
    if (!is_child || fm_keep_fd < 0) return NULL;
    char line[512];
    int n = snprintf(line, sizeof line, "%s\n", text);
    if (n <= 0 || n >= (int)sizeof line) return NULL;
    ssize_t w;
    do {
        w = write(fm_keep_fd, line, (size_t)n);
    } while (w < 0 && errno == EINTR);
    return w == n ? "1" : NULL;
}

static char *fm_collect(long want, int wait_all) {
    static char out[16384];
    size_t have = 0;
    long lines = 0;
    if (fm_pipe[0] < 0) return NULL;
    uint64_t t0 = real_ns(CLOCK_MONOTONIC);
    if (fm_pipe[1] >= 0) close(fm_pipe[1]);
    fm_pipe[1] = fm_keep_fd = -1;
    while (lines < want && have < sizeof out - 1) {
        ssize_t r = read(fm_pipe[0], out + have, sizeof out - 1 - have);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        for (ssize_t i = 0; i < r; i++) {
            if (out[have + i] == '\n') {
                out[have + i] = ';';
                lines++;
            }
        }
        have += (size_t)r;
    }
    out[have] = 0;
    close(fm_pipe[0]);
    fm_pipe[0] = -1;
    reap_children_wait(wait_all ? 0 : WNOHANG);   // wait_all (diagnostic): until every clone has exited
    ft_collect_ns += real_ns(CLOCK_MONOTONIC) - t0;
    return out;
}

// In the clone: every channel shared with the parent becomes private (header list).
static void child_isolate(void) {
    int first_generation = !is_child_done;
    is_child_done = 1;
    is_child = 1;
    n_children = 0;
    if (stack_dump_dir) stack_main_tid = (pid_t)syscall(SYS_gettid);   // diagnostic SIGUSR2 dump: this clone's thread
    if (first_generation) clone_stub_steam();   // a clone of a clone inherits the patched pages
    stats_path = NULL;
    const char *prof_clones = real_getenv ? real_getenv("ABP_PROF_CLONES") : NULL;
    if (prof_path && prof_clones) {
        // diagnostic (header, ABP_PROF_STACK): this clone samples into files of its own; interval timers are not
        // inherited by fork(), so the timer is set again
        static char clone_prof[4096], clone_stats[4096];
        snprintf(clone_prof, sizeof clone_prof, "%s/clone-%d.prof", prof_clones, (int)getpid());
        snprintf(clone_stats, sizeof clone_stats, "%s/clone-%d.stats", prof_clones, (int)getpid());
        prof_path = clone_prof;
        stats_path = clone_stats;
        prof_count = prof_flushed = 0;
        prof_stack_count = prof_stack_flushed = 0;
        struct itimerval it = {{0, 1000}, {0, 1000}};
        setitimer(ITIMER_PROF, &it, NULL);
    } else if (prof_path) {
        struct itimerval off = {{0, 0}, {0, 0}};
        setitimer(ITIMER_PROF, &off, NULL);
        prof_path = NULL;
    }
    const char *a = real_getenv ? real_getenv("ABP_FORK_ALARM") : NULL;
    long alarm_s = a ? atol(a) : 120;
    hang_dir = real_getenv ? real_getenv("ABP_FORK_HANG_DIR") : NULL;
    signal(SIGALRM, clone_alarm);
    alarm(alarm_s > 0 ? (unsigned)alarm_s : 0);
    int fds[1024], n = 0;
    DIR *d = opendir("/proc/self/fd");
    if (!d) return;
    int dfd = dirfd(d);
    struct dirent *e;
    while ((e = readdir(d)) && n < 1024) {
        int fd = atoi(e->d_name);
        if (e->d_name[0] >= '0' && e->d_name[0] <= '9' && fd > 2 && fd != dfd) fds[n++] = fd;
    }
    closedir(d);
    int sv[2] = {-1, -1};
    int devnull = open("/dev/null", O_RDWR);
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) sv[0] = sv[1] = -1;   // sv[0] stays open and unread
    for (int i = 0; i < n; i++) {
        int fd = fds[i];
        struct stat st;
        char link[64], target[4096];
        if (fd == devnull || fd == sv[0] || fd == sv[1] || fd == fm_keep_fd || fstat(fd, &st) != 0) continue;
        snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
        ssize_t len = readlink(link, target, sizeof target - 1);
        target[len > 0 ? len : 0] = 0;
        int flags = fcntl(fd, F_GETFL), fdflags = fcntl(fd, F_GETFD);
        int replaced = -1;
        if (S_ISSOCK(st.st_mode) || S_ISFIFO(st.st_mode)) {
            replaced = sv[1];
        } else if (strncmp(target, "anon_inode:", 11) == 0) {
            replaced = devnull;
        } else if (S_ISREG(st.st_mode)) {
            if ((flags & O_ACCMODE) == O_RDONLY) {
                off_t at = lseek(fd, 0, SEEK_CUR);
                int own = open(link, O_RDONLY);
                if (own >= 0) {
                    if (at > 0) lseek(own, at, SEEK_SET);
                    if (dup2(own, fd) >= 0) isolated_fds++;
                    close(own);
                    if (fdflags >= 0) fcntl(fd, F_SETFD, fdflags);
                }
                continue;
            }
            replaced = devnull;
        }
        if (replaced >= 0 && dup2(replaced, fd) >= 0) {
            isolated_fds++;
            if (fdflags >= 0) fcntl(fd, F_SETFD, fdflags);
        }
    }
    if (sv[1] >= 0) close(sv[1]);
    if (devnull >= 0) close(devnull);
}

// Fork without the mutex protocol (2026-10-05, the teacher's cost; ABP_FORK_LITE=1 at launch or getenv("ABP_FORK_LITE:1")
// at run time, inherited by clones; off by default). The protocol exists for the threads of the instance: in a clone
// (made by ABP_FORK) only the main thread exists, which is here, inside getenv, so no game mutex can be held by
// anyone else. When this process is a clone, has exactly one thread (/proc/self/stat) and every registered mutex is
// in the free state (lock word, count, owner and users all 0, checked here by reading them), the protocol's net
// effect on the clone's mutex memory is nothing: game_mutexes_hold takes them and the parent's release gives them back
// as they were, and the child's release writes exactly these free values. Skipping it saves the writes to some
// thousand mutexes in the parent and in the child (copy-on-write faults on their pages: the parent's pages are shared
// with the clones it made before). Otherwise the full protocol runs (fork_lite_refused counts those forks).
static int fork_lite = -1;   // -1: not read from the environment yet
static uint64_t fork_lite_n, fork_lite_refused;

static int one_thread(void) {
    char buf[1024];
    int fd = open("/proc/self/stat", O_RDONLY);
    if (fd < 0) return 0;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = 0;
    char *p = strrchr(buf, ')');   // after the command name: state is field 3, num_threads field 20
    if (!p) return 0;
    int field = 2;
    for (p++; *p; p++) {
        if (*p == ' ' && ++field == 20) return atol(p + 1) == 1;
    }
    return 0;
}

static int game_mutexes_free(void) {
    for (int i = 0; i < n_game_mutexes; i++) {
        pthread_mutex_t *m = game_mutexes[i];
        if (m->__data.__lock || m->__data.__count || m->__data.__owner || m->__data.__nusers) return 0;
    }
    return 1;
}

static char *fork_game(void) {
    static char answer[24];
    reap_children();
    if (n_children >= MAX_CHILDREN) {
        fork_failures++;
        return "-1";
    }
    uint64_t t0 = real_ns(CLOCK_MONOTONIC);
    ft_t0 = t0;
    fflush(NULL);   // buffered output written once, by the parent
    int lite = 0;
    if (fork_lite < 0) {
        const char *v = real_getenv ? real_getenv("ABP_FORK_LITE") : NULL;
        fork_lite = v && atoi(v) > 0;
    }
    if (fork_lite && is_child) {
        lite = one_thread() && game_mutexes_free();
        if (lite) fork_lite_n++;
        else fork_lite_refused++;
    }
    if (!lite) game_mutexes_hold();
    uint64_t t1 = real_ns(CLOCK_MONOTONIC);
    pid_t pid = fork();
    if (pid == 0) {
        uint64_t c0 = real_ns(CLOCK_MONOTONIC);
        is_child = 1;
        if (!lite) game_mutexes_release_child();
        uint64_t c1 = real_ns(CLOCK_MONOTONIC);
        child_isolate();
        ft_born_ns = real_ns(CLOCK_MONOTONIC);
        ft_isolate_ns = ft_born_ns - c0;
        ft_n = ft_total_ns = ft_hold_ns = ft_sys_ns = ft_collect_ns = ft_reaped_ns = 0;
        ft_hold_ns = c1 - c0;   // in a clone: the time it took to set the held mutexes free
        return "0";
    }
    uint64_t t2 = real_ns(CLOCK_MONOTONIC);
    if (!lite) game_mutexes_release_parent();
    uint64_t t3 = real_ns(CLOCK_MONOTONIC);
    fork_last_us = (t3 - t0) / 1000;
    ft_n++;
    ft_total_ns += t3 - t0;
    ft_hold_ns += t1 - t0;
    ft_sys_ns += t2 - t1;
    if (pid < 0) {
        fork_failures++;
        return "-1";
    }
    forks++;
    children[n_children++] = pid;
    snprintf(answer, sizeof answer, "%d", (int)pid);
    return answer;
}

// ---- ABP_OBS_INIT: native lean observation, see the header ----
// Lua 5.3 C API and game functions statically linked in isaac.x64. Every address is checked against its exported name
// (dladdr) before anything is registered; the getters are the very functions luabridge calls for the bridge's Lua code
// (found in register_classes, RVA 0x35C250), called here directly.
typedef struct lua_State lua_State;
typedef int (*lua_cfn)(lua_State *);
#define G_LUAENGINE 0xdbc380UL        // g_LuaEngine: LuaEngine*, whose first field is the lua_State*
#define GAME_ROOM 0x99e8UL            // Game: Room* (Game::GetRoom)
#define ROOM_ENTITIES 0x1640UL        // Room: Entity** of the room's entity list (what Isaac.GetRoomEntities returns,
#define ROOM_ENTITY_COUNT 0x164cUL    //   LuaEngine::lua_get_entities) and its count (uint32)
#define VT_CAN_SHUT_DOORS 0x78UL      // Entity vtable slots (luabridge registers e:CanShutDoors() / e:IsBoss() as the
#define VT_IS_BOSS 0x80UL             //   virtual member pointers 0x79 / 0x81)
// luabridge data members (addData offsets in register_classes)
#define PL_FIRE_DELAY 0x26dc          // Entity_Player: int FireDelay
#define PL_MAX_FIRE_DELAY 0x26e0      //   int MaxFireDelay
#define PL_SHOT_SPEED 0x26e4          //   float ShotSpeed
#define PL_DAMAGE 0x26f0              //   float Damage
#define PL_TEAR_HEIGHT 0x26f4         //   float TearHeight
#define PL_MOVE_SPEED 0x27e4          //   float MoveSpeed
#define PL_LUCK 0x27e8                //   float Luck
#define PL_CAN_FLY 0x27ec             //   bool CanFly
// 2026-10-07 (charge): not luabridge members (no Lua API reads them in AB+); found in Entity_Player::render_chargebar
// (RVA 0x2564B0), EvaluateItems (0x25D2A0) and HasWeaponType (0x3819A0)
#define PL_CHARGE 0x2634              //   int: the charge counter the charge bar shows (Brimstone, Lung, Knife, ...)
#define PL_WEAPON_TYPES 0x2628        //   bool[11]: HasWeaponType(w) is the byte at +0x2628 + w
#define DOOR_TARGET_TYPE 0x538        // GridEntity_Door: int TargetRoomType
#define DOOR_TARGET_INDEX 0xa2c       //   int TargetRoomIndex
#define DESC_SAFE_GRID_INDEX 0x4      // RoomDescriptor: int SafeGridIndex
#define DESC_VISITED 0x4c             //   unsigned VisitedCount
#define LEAN_MAGIC 0x33504241u
#define LEAN_FULL_EVERY 30

#define OBS_FUNC_LIST(X) \
    X(PUSHLSTRING, 0x92e080, "lua_pushlstring") X(PUSHNUMBER, 0x92e040, "lua_pushnumber") \
    X(PUSHINTEGER, 0x92e060, "lua_pushinteger") X(PUSHNIL, 0x92e020, "lua_pushnil") \
    X(PUSHBOOLEAN, 0x92e320, "lua_pushboolean") X(PUSHCCLOSURE, 0x92e250, "lua_pushcclosure") \
    X(SETGLOBAL, 0x92e7a0, "lua_setglobal") X(TOINTEGERX, 0x92ddc0, "lua_tointegerx") \
    X(TOBOOLEAN, 0x92de10, "lua_toboolean") X(LUAENGINE, 0xdbc380, "g_LuaEngine") \
    X(E_TYPE, 0x780600, "_ZNK6Entity7GetTypeEv") X(E_VARIANT, 0x780610, "_ZNK6Entity10GetVariantEv") \
    X(E_SUBTYPE, 0x780630, "_ZNK6Entity10GetSubTypeEv") X(E_INDEX, 0x780810, "_ZNK6Entity8GetIndexEv") \
    X(E_POSITION, 0x780690, "_ZNK6Entity11GetPositionEv") X(E_VELOCITY, 0x7806a0, "_ZNK6Entity11GetVelocityEv") \
    X(E_SIZE, 0x746710, "_ZN15LuaProxyHelpers7GetSizeEPK6Entity") \
    X(E_SIZEMULTI, 0x7807a0, "_ZNK6Entity12GetSizeMultiEv") \
    X(E_ECC, 0x780840, "_ZNK6Entity23GetEntityCollisionClassEv") \
    X(E_GCC, 0x780830, "_ZNK6Entity21GetGridCollisionClassEv") \
    X(E_CDMG, 0x780870, "_ZNK6Entity18GetCollisionDamageEv") \
    X(E_FRAMECOUNT, 0x53a500, "_ZNK6Entity13GetFrameCountEv") X(E_FLIPX, 0x780750, "_ZNK6Entity8GetFlipXEv") \
    X(E_VISIBLE, 0x7806f0, "_ZNK6Entity10GetVisibleEv") X(E_HP, 0x7807e0, "_ZNK6Entity12GetHitPointsEv") \
    X(E_MAXHP, 0x7807d0, "_ZNK6Entity15GetMaxHitPointsEv") X(E_IS_DEAD, 0x780910, "_ZNK6Entity6IsDeadEv") \
    X(E_IS_ENEMY, 0x53b610, "_ZNK6Entity7IsEnemyEv") X(E_IS_VULNERABLE, 0x53b620, "_ZNK6Entity17IsVulnerableEnemyEv") \
    X(E_SPRITE, 0x7809c0, "_ZN6Entity9GetSpriteEv") X(E_CANSHUT_BASE, 0x541a80, "_ZNK6Entity12CanShutDoorsEv") \
    X(E_ISBOSS_BASE, 0x541a90, "_ZNK6Entity6IsBossEv") \
    X(ANM2_FRAME, 0x4f2ec0, "_ZNK4ANM28GetFrameEv") X(ANM2_PLAYING, 0x4f2ad0, "_ZNK4ANM29IsPlayingEPKc") \
    X(ANM2_FINISHED, 0x4f2b20, "_ZNK4ANM210IsFinishedEPKc") \
    X(NPC_CHAMPION, 0x781e60, "_ZNK10Entity_NPC19GetChampionColorIdxEv") \
    X(TEAR_HEIGHT, 0x7809d0, "_ZNK11Entity_Tear9GetHeightEv") \
    X(TEAR_FALL, 0x7809e0, "_ZNK11Entity_Tear15GetFallingSpeedEv") \
    X(TEAR_SCALE, 0x780a00, "_ZNK11Entity_Tear8GetScaleEv") \
    X(PROJ_HEIGHT, 0x781bb0, "_ZNK17Entity_Projectile9GetHeightEv") \
    X(PROJ_FALL, 0x781bf0, "_ZNK17Entity_Projectile15GetFallingSpeedEv") \
    X(PROJ_SCALE, 0x781d60, "_ZNK17Entity_Projectile8GetScaleEv") \
    X(TEAR_FLAGS, 0x780a10, "_ZNK11Entity_Tear12GetTearFlagsEv") \
    X(PROJ_FLAGS, 0x781da0, "_ZNK17Entity_Projectile18GetProjectileFlagsEv") \
    X(P_GET, 0x747190, "_ZN9LuaEngine14lua_get_playerEi") X(G_NUM_PLAYERS, 0x781ed0, "_ZN4Game13GetNumPlayersEv") \
    X(G_FRAME, 0x781f30, "_ZN4Game13GetFrameCountEv") X(G_PAUSED, 0x6dd5b0, "_ZNK4Game8IsPausedEv") \
    X(G_ROOM, 0x781ec0, "_ZN4Game7GetRoomEv") X(G_LEVEL, 0x781eb0, "_ZN4Game8GetLevelEv") \
    X(P_HEARTS, 0x781880, "_ZNK13Entity_Player9GetHeartsEv") X(P_MAXHEARTS, 0x781890, "_ZNK13Entity_Player12GetMaxHeartsEv") \
    X(P_SOUL, 0x7818a0, "_ZNK13Entity_Player13GetSoulHeartsEv") \
    X(P_BLACK, 0x7818b0, "_ZNK13Entity_Player14GetBlackHeartsEv") \
    X(P_BONE, 0x781b80, "_ZNK13Entity_Player13GetBoneHeartsEv") \
    X(P_ETERNAL, 0x7818c0, "_ZNK13Entity_Player16GetEternalHeartsEv") \
    X(P_GOLDEN, 0x781910, "_ZNK13Entity_Player15GetGoldenHeartsEv") \
    X(P_LIVES, 0x65a270, "_ZNK13Entity_Player13GetExtraLivesEv") X(P_COINS, 0x781920, "_ZNK13Entity_Player11GetNumCoinsEv") \
    X(P_BOMBS, 0x7818d0, "_ZNK13Entity_Player11GetNumBombsEv") X(P_KEYS, 0x7818e0, "_ZNK13Entity_Player10GetNumKeysEv") \
    X(P_ACTIVE, 0x781750, "_ZNK13Entity_Player13GetActiveItemEv") \
    X(P_CHARGE, 0x781760, "_ZNK13Entity_Player15GetActiveChargeEv") \
    X(P_NEEDS_CHARGE, 0x65a5e0, "_ZNK13Entity_Player11NeedsChargeEv") \
    X(P_DMG_COOLDOWN, 0x781a10, "_ZNK13Entity_Player17GetDamageCooldownEv") \
    X(P_CONTROLS, 0x7819f0, "_ZNK13Entity_Player18GetControlsEnabledEv") \
    X(P_HEAD_DIR, 0x7817f0, "_ZNK13Entity_Player16GetHeadDirectionEv") \
    X(P_FIRE_DIR, 0x7817e0, "_ZNK13Entity_Player16GetFireDirectionEv") \
    X(P_MOVE_DIR, 0x7817d0, "_ZNK13Entity_Player20GetMovementDirectionEv") \
    X(P_DAMAGE_TAKEN, 0x781ad0, "_ZNK13Entity_Player19GetTotalDamageTakenEv") \
    X(P_PLAYER_TYPE, 0x781930, "_ZNK13Entity_Player13GetPlayerTypeEv") \
    X(R_TYPE, 0x781060, "_ZNK4Room7GetTypeEv") X(R_SHAPE, 0x7810a0, "_ZNK4Room12GetRoomShapeEv") \
    X(R_WIDTH, 0x781120, "_ZNK4Room12GetGridWidthEv") X(R_HEIGHT, 0x781130, "_ZNK4Room13GetGridHeightEv") \
    X(R_SIZE, 0x781140, "_ZNK4Room11GetGridSizeEv") X(R_ALIVE, 0x7811b0, "_ZN4Room20GetAliveEnemiesCountEv") \
    X(R_FRAME, 0x82fb00, "_ZN4Room13GetFrameCountEv") X(R_TOPLEFT, 0x781180, "_ZNK4Room13GetTopLeftPosEv") \
    X(R_BOTTOMRIGHT, 0x781190, "_ZNK4Room17GetBottomRightPosEv") X(R_CLEAR, 0x781150, "_ZNK4Room7IsClearEv") \
    X(R_DOOR, 0x781030, "_ZN4Room7GetDoorE9eDoorSlot") X(R_COLLISION, 0x82d0a0, "_ZNK4Room16GetGridCollisionEi") \
    X(R_GRID_ENTITY, 0x781100, "_ZNK4Room13GetGridEntityEi") \
    X(L_STAGE, 0x781490, "_ZNK5Level8GetStageEv") X(L_ROOM_INDEX, 0x7813e0, "_ZNK5Level19GetCurrentRoomIndexEv") \
    X(L_ROOM_DESC, 0x734400, "_ZN5Level18GetCurrentRoomDescEv") X(L_ROOM_BY_IDX, 0x733cd0, "_ZN5Level12GetRoomByIdxEi") \
    X(D_CLEAR, 0x780ec0, "_ZNK14RoomDescriptor7IsClearEv") \
    X(GE_TYPE, 0x780d90, "_ZN10GridEntity7GetTypeEv") X(GE_VARIANT, 0x780da0, "_ZN10GridEntity10GetVariantEv") \
    X(GE_POSITION, 0x6f96f0, "_ZNK10GridEntity11GetPositionEv") \
    X(DOOR_OPEN, 0x780e40, "_ZNK15GridEntity_Door6IsOpenEv") X(DOOR_LOCKED, 0x7833b0, "_ZNK15GridEntity_Door8IsLockedEv")

#define OBS_ENUM(id, va, name) F_##id,
enum { OBS_FUNC_LIST(OBS_ENUM) F_COUNT };
#define OBS_ENTRY(id, va, name) [F_##id] = {(uintptr_t)va, name},
static const struct { uintptr_t va; const char *name; } OBS_FUNCS[F_COUNT] = { OBS_FUNC_LIST(OBS_ENTRY) };

typedef struct { float x, y; } vec2f;
typedef void *(*dyncast_fn)(const void *, const void *, const void *, ptrdiff_t);
static dyncast_fn obs_dyncast;
static const void *ti_entity, *ti_npc, *ti_tear, *ti_projectile;
static int obs_ready = -1;            // -1 not checked, 0 refused, 1 registered
static char obs_refusal[160];
static unsigned char *obs_buf;
static size_t obs_cap;
static uint64_t obs_calls, obs_fallbacks, obs_terrain_calls;

#define FN(id) ((void *)OBS_FUNCS[F_##id].va)
#define G0(T, id, obj) (((T (*)(const void *))FN(id))(obj))                       // T getter(this)
#define G1(T, id, obj, a) (((T (*)(const void *, int))FN(id))((obj), (a)))        // T getter(this, int)
#define FIELD(T, obj, off) (*(const T *)((const char *)(obj) + (off)))
#define LPUSHNIL(L) ((void (*)(lua_State *))FN(PUSHNIL))(L)
#define LPUSHSTR(L, s, n) ((const char *(*)(lua_State *, const void *, size_t))FN(PUSHLSTRING))((L), (s), (n))
#define LPUSHBOOL(L, b) ((void (*)(lua_State *, int))FN(PUSHBOOLEAN))((L), (b))
#define LPUSHINT(L, v) ((void (*)(lua_State *, long long))FN(PUSHINTEGER))((L), (v))
#define LTOINT(L, i) ((long long (*)(lua_State *, int, int *))FN(TOINTEGERX))((L), (i), NULL)
#define LTOBOOL(L, i) ((int (*)(lua_State *, int))FN(TOBOOLEAN))((L), (i))

static int obs_check(void) {
    for (int i = 0; i < F_COUNT; i++) {
        Dl_info info;
        if (!dladdr((void *)OBS_FUNCS[i].va, &info) || (uintptr_t)info.dli_saddr != OBS_FUNCS[i].va || !info.dli_sname ||
            strcmp(info.dli_sname, OBS_FUNCS[i].name) != 0) {
            snprintf(obs_refusal, sizeof obs_refusal, "%s not at 0x%lx", OBS_FUNCS[i].name,
                     (unsigned long)OBS_FUNCS[i].va);
            return 0;
        }
    }
    obs_dyncast = (dyncast_fn)dlsym(RTLD_DEFAULT, "__dynamic_cast");
    ti_entity = dlsym(RTLD_DEFAULT, "_ZTI6Entity");
    ti_npc = dlsym(RTLD_DEFAULT, "_ZTI10Entity_NPC");
    ti_tear = dlsym(RTLD_DEFAULT, "_ZTI11Entity_Tear");
    ti_projectile = dlsym(RTLD_DEFAULT, "_ZTI17Entity_Projectile");
    const uintptr_t *vt = (const uintptr_t *)dlsym(RTLD_DEFAULT, "_ZTV6Entity");
    if (!obs_dyncast || !ti_entity || !ti_npc || !ti_tear || !ti_projectile || !vt) {
        snprintf(obs_refusal, sizeof obs_refusal, "dynamic_cast / typeinfo / vtable not found");
        return 0;
    }
    // the base class's own slots are Entity::CanShutDoors / Entity::IsBoss (vtable entries start 16 bytes in)
    if (vt[2 + VT_CAN_SHUT_DOORS / 8] != OBS_FUNCS[F_E_CANSHUT_BASE].va ||
        vt[2 + VT_IS_BOSS / 8] != OBS_FUNCS[F_E_ISBOSS_BASE].va) {
        snprintf(obs_refusal, sizeof obs_refusal, "Entity vtable slots differ");
        return 0;
    }
    return 1;
}

static inline int vcall_bool(const void *e, uintptr_t slot) {
    const uintptr_t *vt = *(const uintptr_t *const *)e;
    return ((char (*)(const void *))vt[slot / 8])(e) != 0;
}

static inline int gbool(void *fn, const void *obj) {   // a bool getter (only AL is defined)
    return ((char (*)(const void *))fn)(obj) != 0;
}

static void obs_reserve(size_t need) {
    if (need <= obs_cap) return;
    size_t cap = obs_cap ? obs_cap : 8192;
    while (cap < need) cap *= 2;
    unsigned char *b = realloc(obs_buf, cap);
    if (!b) abort();
    obs_buf = b;
    obs_cap = cap;
}

#define PUT(off, v) do { __typeof__(v) put_tmp_ = (v); memcpy(obs_buf + (off), &put_tmp_, sizeof put_tmp_); } while (0)
// Lua numbers: luabridge hands an engine float to Lua as a double; string.pack writes a "d" field as that double and an
// "f" field as that double rounded to float. F2D / F2F do the same conversions (bit for bit, NaNs included).
#define F2D(f) ((double)(float)(f))
#define F2F(f) ((float)(double)(float)(f))

static int obs_fallback(lua_State *L, const char *why) {
    LPUSHNIL(L);
    LPUSHSTR(L, why, strlen(why));
    return 2;
}

static inline uintptr_t obs_game(void) { return *(volatile uintptr_t *)G_GAME; }

// lean_room_index: the current room's SafeGridIndex (as Level:GetCurrentRoomIndex() when that is negative).
static int obs_room_index(uintptr_t game) {
    const void *level = G0(const void *, G_LEVEL, (const void *)game);
    int idx = G0(int, L_ROOM_INDEX, level);
    if (idx >= 0) {
        const void *desc = G0(const void *, L_ROOM_DESC, level);
        if (desc) idx = FIELD(int32_t, desc, DESC_SAFE_GRID_INDEX);
    }
    return idx;
}

// ---- terrain change check (the bridge's lean_terrain_changed, same rules, its own state) ----
// The signatures are kept in binary, not as the Lua strings: two signatures are equal exactly when the Lua strings
// would be (the player's CanFly and Size, every cell's collision class, the eight door variants or -1, the cells holding
// a trapdoor or stairs). Each path keeps its own state; the bridge forces a full pass whenever it switches paths.
typedef struct { int32_t *v; size_t n, cap; int valid; } i32vec;
static i32vec tr_full_sig, tr_part_sig, tr_scratch, tr_cells;
static int tr_have_cells;
static long long tr_full_at = -100000;

static void iv_push(i32vec *a, int32_t x) {
    if (a->n == a->cap) {
        size_t cap = a->cap ? a->cap * 2 : 512;
        int32_t *v = realloc(a->v, cap * sizeof *v);
        if (!v) abort();
        a->v = v;
        a->cap = cap;
    }
    a->v[a->n++] = x;
}

static int iv_equal(const i32vec *a, const i32vec *b) {
    return a->valid && b->valid && a->n == b->n && memcmp(a->v, b->v, a->n * sizeof *a->v) == 0;
}

static void iv_copy(i32vec *dst, const i32vec *src) {
    dst->n = 0;
    for (size_t i = 0; i < src->n; i++) iv_push(dst, src->v[i]);
    dst->valid = 1;
}

static void door_variants(const void *room, i32vec *sig) {
    for (int slot = 0; slot < 8; slot++) {
        const void *d = G1(const void *, R_DOOR, room, slot);
        iv_push(sig, d ? (int32_t)G0(uint32_t, GE_VARIANT, d) : -1);
    }
}

// The content signature (2026-10-07; the bridge's terrain_content_sig, same values): per cell the room's collision class
// and, with a grid entity, its type, variant, state (+0x10) and collision class (+0x40), else -1. Returns 1 when it
// differs from the previous call's (none before the first call), and keeps it.
#define GE_STATE 0x10
#define GE_COLLISION_CLASS 0x40
static i32vec tr_content_sig, tr_content_now;
static uint64_t tr_content_changes;

static int obs_terrain_content(const void *room) {
    i32vec *k = &tr_content_now;
    k->n = 0;
    k->valid = 1;
    int n = G0(int, R_SIZE, room);
    for (int i = 0; i < n; i++) {
        iv_push(k, G1(int, R_COLLISION, room, i));
        const void *g = G1(const void *, R_GRID_ENTITY, room, i);
        if (!g) {
            iv_push(k, -1);
            continue;
        }
        iv_push(k, G0(int, GE_TYPE, g));
        iv_push(k, G0(int, GE_VARIANT, g));
        iv_push(k, FIELD(int32_t, g, GE_STATE));
        iv_push(k, FIELD(int32_t, g, GE_COLLISION_CLASS));
    }
    int changed = tr_content_sig.valid && !iv_equal(k, &tr_content_sig);
    tr_content_changes += changed;
    iv_copy(&tr_content_sig, k);
    return changed;
}

static int obs_terrain_changed(const void *room, const void *player, long long lf, int force) {
    int changed = 0;
    if (force || !tr_have_cells || lf - tr_full_at >= LEAN_FULL_EVERY || lf < tr_full_at) {
        i32vec *sig = &tr_scratch;
        sig->n = 0;
        sig->valid = 1;
        double size = F2D(G0(float, E_SIZE, player));
        int32_t sz[2];
        memcpy(sz, &size, sizeof size);
        iv_push(sig, FIELD(unsigned char, player, PL_CAN_FLY) != 0);
        iv_push(sig, sz[0]);
        iv_push(sig, sz[1]);
        int n = G0(int, R_SIZE, room);
        for (int i = 0; i < n; i++) iv_push(sig, G1(int, R_COLLISION, room, i));
        door_variants(room, sig);
        for (int i = 0; i < n; i++) {
            const void *g = G1(const void *, R_GRID_ENTITY, room, i);
            if (g) {
                int gt = G0(int, GE_TYPE, g);
                if (gt == 17 || gt == 18) {
                    iv_push(sig, gt);
                    iv_push(sig, i);
                }
            }
        }
        tr_full_at = lf;
        if (force || !iv_equal(sig, &tr_full_sig)) {
            iv_copy(&tr_full_sig, sig);
            tr_cells.n = 0;
            for (int i = 0; i < n; i++) {
                int c = G1(int, R_COLLISION, room, i);
                if (c != 0 && c != 4) iv_push(&tr_cells, i);
            }
            tr_have_cells = 1;
            tr_part_sig.valid = 0;
            changed = 1;
        }
    }
    i32vec *part = &tr_scratch;
    part->n = 0;
    part->valid = 1;
    for (size_t k = 0; k < tr_cells.n; k++) iv_push(part, G1(int, R_COLLISION, room, tr_cells.v[k]));
    door_variants(room, part);
    if (tr_part_sig.valid && !iv_equal(part, &tr_part_sig)) changed = 1;
    iv_copy(&tr_part_sig, part);
    // 2026-10-07: every decision, what the terrain block's grid records hold (bridge terrain_content_sig): a grid
    // entity's state or variant changing without a collision change (a poop or a cobweb hit by a bomb or a tear, a
    // trapdoor appearing) is resent at once, not at the next full pass, so a step's terrain is always the current one
    if (obs_terrain_content(room)) changed = 1;
    if (changed) tr_full_sig.valid = 0;
    return changed;
}

// Lua: resend, here, clear = abp_native_terrain(logic_frames, force)
static int obs_lua_terrain(lua_State *L) {
    obs_terrain_calls++;
    uintptr_t game = obs_game();
    const void *room = game ? G0(const void *, G_ROOM, (const void *)game) : NULL;
    const void *player0 = ((const void *(*)(int))FN(P_GET))(0);
    if (!room || !player0) {
        obs_fallbacks++;
        return obs_fallback(L, "no room or player");
    }
    int resend = obs_terrain_changed(room, player0, LTOINT(L, 1), LTOBOOL(L, 2));
    LPUSHBOOL(L, resend);
    LPUSHINT(L, obs_room_index(game));
    LPUSHBOOL(L, G0(char, R_CLEAR, room) != 0);
    return 3;
}

// Lua: same = abp_native_terrain_key(mark) -- the bridge's terrain block cache (abp_bridge.lua ABP_TERRAIN_CACHE): is
// everything the terrain block's two JSON strings are built from (terrain_record(room, player0, true), grid_records)
// the same as at the last marked call? mark: this call's key becomes the reference. The key: room shape, grid width,
// height and size, room index, stage, player 0's Size (as the double Lua sees) and CanFly; per cell the room's
// collision class and, with a grid entity, its type, variant, state (+0x10, GetState), collision class (+0x40, the
// CollisionClass property: luabridge data member, register_classes) and position (GE_STATE, GE_COLLISION_CLASS above).
static i32vec tk_ref, tk_now;
static uint64_t tk_calls, tk_same;

static int obs_lua_tkey(lua_State *L) {
    tk_calls++;
    int mark = LTOBOOL(L, 1);
    uintptr_t game = obs_game();
    const void *room = game ? G0(const void *, G_ROOM, (const void *)game) : NULL;
    const void *level = game ? G0(const void *, G_LEVEL, (const void *)game) : NULL;
    const void *player0 = ((const void *(*)(int))FN(P_GET))(0);
    if (!room || !level || !player0) {
        tk_ref.valid = 0;
        LPUSHBOOL(L, 0);
        return 1;
    }
    i32vec *k = &tk_now;
    k->n = 0;
    k->valid = 1;
    double size = F2D(G0(float, E_SIZE, player0));
    int32_t sz[2];
    memcpy(sz, &size, sizeof size);
    int n = G0(int, R_SIZE, room);
    iv_push(k, G0(int, R_SHAPE, room));
    iv_push(k, G0(int, R_WIDTH, room));
    iv_push(k, G0(int, R_HEIGHT, room));
    iv_push(k, n);
    iv_push(k, obs_room_index(game));
    iv_push(k, G0(int, L_STAGE, level));
    iv_push(k, sz[0]);
    iv_push(k, sz[1]);
    iv_push(k, FIELD(unsigned char, player0, PL_CAN_FLY) != 0);
    for (int i = 0; i < n; i++) {
        iv_push(k, G1(int, R_COLLISION, room, i));
        const void *g = G1(const void *, R_GRID_ENTITY, room, i);
        if (!g) {
            iv_push(k, -1);
            continue;
        }
        vec2f pos = G0(vec2f, GE_POSITION, g);
        int32_t px, py;
        memcpy(&px, &pos.x, 4);
        memcpy(&py, &pos.y, 4);
        iv_push(k, G0(int, GE_TYPE, g));
        iv_push(k, G0(int, GE_VARIANT, g));
        iv_push(k, FIELD(int32_t, g, GE_STATE));
        iv_push(k, FIELD(int32_t, g, GE_COLLISION_CLASS));
        iv_push(k, px);
        iv_push(k, py);
    }
    int same = iv_equal(k, &tk_ref);
    tk_same += same;
    if (mark) iv_copy(&tk_ref, k);
    LPUSHBOOL(L, same);
    return 1;
}

// ---- the fixed part of the lean observation (pack_lean up to the lasers) ----
static const char *const ANIM_MONSTRO[] = {"Walk", "JumpUp", "JumpDown", "Taunt", "Appear", "Death"};
static const char *const ANIM_BOMB[] = {"Pulse", "Idle", "Explode"};

// The entity records (no player, whitelisted effects only, visible only) and the totals of the room's NPCs, appended
// at *off. Returns NULL, or why the Lua code has to do it (a laser carries Lua-built samples; a value the Lua code
// would refuse).
// abp-0.2.16 (2026-10-08): LEAN_ENTITY_BYTES per record: the 108 of before, then the flag word (Entity_Tear::
// GetTearFlags / Entity_Projectile::GetProjectileFlags, the uint64 at +0x758 / +0x760 that luabridge pushes with
// lua_pushinteger as the Lua properties TearFlags / ProjectileFlags; 0 for other entities) and the laser geometry
// (zero here: a laser in the room sends the observation to the Lua code).
#define LEAN_ENTITY_BYTES 144
static const char *obs_entities(uintptr_t room, size_t *off, double *monsters_hp, double *blocking_hp,
                                long long *blocking_count) {
    void *const *list = *(void *const *const *)(room + ROOM_ENTITIES);
    uint32_t n = *(const uint32_t *)(room + ROOM_ENTITY_COUNT);
    obs_reserve(*off + 2 + (size_t)n * LEAN_ENTITY_BYTES);
    size_t at = *off + 2;
    unsigned count = 0;
    for (uint32_t i = 0; i < n; i++) {
        const void *e = list[i];
        if (!e) continue;   // lua_get_entities pushes nil for it, which luaL_ref does not store
        int t = G0(int, E_TYPE, e);
        const void *npc = (t >= 10 && t < 1000) ? obs_dyncast(e, ti_entity, ti_npc, 0) : NULL;   // e:ToNPC()
        if (npc) {
            double hp = F2D(G0(float, E_HP, e));
            int blocks = vcall_bool(e, VT_CAN_SHUT_DOORS) && !gbool(FN(E_IS_DEAD), e);
            if (hp > 0) {
                if (t != 17 && t != 33 && t != 245 && t != 292) *monsters_hp = *monsters_hp + hp;   // is_monster
                if (blocks) {
                    *blocking_hp = *blocking_hp + hp;
                    (*blocking_count)++;
                }
            } else if (blocks) {
                (*blocking_count)++;
            }
        }
        if (t == 1) continue;
        uint32_t variant = G0(uint32_t, E_VARIANT, e);
        if (t == 1000) {
            switch (variant) {   // cfg.effect_whitelist
            case 1: case 22: case 23: case 26: case 30: case 45: case 46: case 51: case 53: case 55: break;
            default: continue;
            }
        }
        if (!gbool(FN(E_VISIBLE), e)) continue;
        int kind = 0, flags = 0, anim = 0;
        int32_t champion = -1;
        float hpf = 0.0f, a = 0.0f, b = 0.0f, c = 0.0f;
        uint64_t word = 0;
        if (t == 2) {
            const void *x = obs_dyncast(e, ti_entity, ti_tear, 0);
            if (!x) return "ToTear";
            kind = 1;
            a = F2F(G0(float, TEAR_HEIGHT, x));
            b = F2F(G0(float, TEAR_FALL, x));
            c = F2F(G0(float, TEAR_SCALE, x));
            word = G0(uint64_t, TEAR_FLAGS, x);
        } else if (t == 9) {
            const void *x = obs_dyncast(e, ti_entity, ti_projectile, 0);
            if (!x) return "ToProjectile";
            kind = 2;
            a = F2F(G0(float, PROJ_HEIGHT, x));
            b = F2F(G0(float, PROJ_FALL, x));
            c = F2F(G0(float, PROJ_SCALE, x));
            word = G0(uint64_t, PROJ_FLAGS, x);
        } else if (t == 7) {
            return "laser";
        } else if (t == 4) {
            kind = 4;
        } else if (t == 5) {
            kind = 5;
        } else if (npc) {
            kind = 6;
            int boss = vcall_bool(e, VT_IS_BOSS);
            if (gbool(FN(E_IS_ENEMY), e)) flags += 1;
            if (gbool(FN(E_IS_VULNERABLE), e)) flags += 2;
            if (boss) {
                flags += 4;
                double maxhp = F2D(G0(float, E_MAXHP, e));
                if (maxhp > 0) hpf = (float)(F2D(G0(float, E_HP, e)) / maxhp);
            }
            if (vcall_bool(e, VT_CAN_SHUT_DOORS) && !gbool(FN(E_IS_DEAD), e)) flags += 8;
            champion = G0(int32_t, NPC_CHAMPION, npc);
        }
        const void *spr = G0(const void *, E_SPRITE, e);
        const char *const *names = t == 20 ? ANIM_MONSTRO : t == 4 ? ANIM_BOMB : NULL;   // ANIMATIONS
        int n_names = t == 20 ? 6 : t == 4 ? 3 : 0;
        for (int k = 0; k < n_names; k++) {
            if (((char (*)(const void *, const char *))FN(ANM2_PLAYING))(spr, names[k]) ||
                ((char (*)(const void *, const char *))FN(ANM2_FINISHED))(spr, names[k])) {
                anim = k + 1;
                break;
            }
        }
        uint32_t subtype = G0(uint32_t, E_SUBTYPE, e);
        if (variant > 0x7fffffffu || subtype > 0x7fffffffu) return "int32 range";   // string.pack "i4" refuses it
        vec2f pos = G0(vec2f, E_POSITION, e), vel = G0(vec2f, E_VELOCITY, e), sm = G0(vec2f, E_SIZEMULTI, e);
        PUT(at + 0, (int64_t)G0(uint32_t, E_INDEX, e));
        PUT(at + 8, (int32_t)t);
        PUT(at + 12, (int32_t)variant);
        PUT(at + 16, (int32_t)subtype);
        PUT(at + 20, F2D(pos.x));
        PUT(at + 28, F2D(pos.y));
        PUT(at + 36, F2D(vel.x));
        PUT(at + 44, F2D(vel.y));
        PUT(at + 52, F2F(G0(float, E_SIZE, e)));
        PUT(at + 56, F2F(sm.x));
        PUT(at + 60, F2F(sm.y));
        PUT(at + 64, (int32_t)G0(int, E_ECC, e));
        PUT(at + 68, (int32_t)G0(int, E_GCC, e));
        PUT(at + 72, F2F(G0(float, E_CDMG, e)));
        PUT(at + 76, (int32_t)G0(int, ANM2_FRAME, spr));
        PUT(at + 80, (int32_t)G0(int, E_FRAMECOUNT, e));
        PUT(at + 84, (uint8_t)kind);
        PUT(at + 85, (uint8_t)flags);
        PUT(at + 86, (uint8_t)anim);
        PUT(at + 87, (uint8_t)gbool(FN(E_FLIPX), e));
        PUT(at + 88, champion);
        PUT(at + 92, hpf);
        PUT(at + 96, a);
        PUT(at + 100, b);
        PUT(at + 104, c);
        PUT(at + 108, (int64_t)word);   // Lua's integer: the same 64 bits
        memset(obs_buf + at + 116, 0, LEAN_ENTITY_BYTES - 116);
        at += LEAN_ENTITY_BYTES;
        count++;
    }
    if (count > 0xffff) return "count";
    PUT(*off, (uint16_t)count);
    *off = at;
    return NULL;
}

// player p's weapon types as bits (bit w: HasWeaponType(w), w = 0 .. 10; the bridge's lean_weapons)
static int obs_weapon_mask(const void *p) {
    int mask = 0;
    for (int w = 0; w <= 10; w++)
        if (FIELD(unsigned char, p, PL_WEAPON_TYPES + w) != 0) mask |= 1 << w;
    return mask;
}

// One LEAN_PLAYER record (OBS_PLAYER_F doubles) at off. 2026-10-08 (bridge abp-0.2.17, character randomisation): the
// 39th is the PlayerType (Entity_Player::GetPlayerType, what the bridge's p:GetPlayerType() calls).
#define OBS_PLAYER_F 39
#define OBS_PLAYER_BYTES (8 * OBS_PLAYER_F)
static void obs_player(const void *p, size_t off) {
    vec2f pos = G0(vec2f, E_POSITION, p), vel = G0(vec2f, E_VELOCITY, p);
    double h = F2D(FIELD(float, p, PL_TEAR_HEIGHT));
    double range = h < 0 ? 260 * h / -23.75 : 260;   // tear_range
    int active = G0(int, P_ACTIVE, p);
    int cooldown = G0(int, P_DMG_COOLDOWN, p);
    double v[OBS_PLAYER_F] = {
        (double)G0(uint32_t, E_INDEX, p), F2D(pos.x), F2D(pos.y), F2D(vel.x), F2D(vel.y), F2D(G0(float, E_SIZE, p)),
        G0(int, P_HEARTS, p), G0(int, P_MAXHEARTS, p), G0(int, P_SOUL, p), G0(int, P_BLACK, p), G0(int, P_BONE, p),
        G0(int, P_ETERNAL, p), G0(int, P_GOLDEN, p), G0(int, P_LIVES, p), G0(int, P_COINS, p), G0(int, P_BOMBS, p),
        G0(int, P_KEYS, p), F2D(FIELD(float, p, PL_DAMAGE)), FIELD(int32_t, p, PL_MAX_FIRE_DELAY),
        F2D(FIELD(float, p, PL_SHOT_SPEED)), range, F2D(FIELD(float, p, PL_MOVE_SPEED)), F2D(FIELD(float, p, PL_LUCK)),
        FIELD(unsigned char, p, PL_CAN_FLY) != 0, active, G0(int, P_CHARGE, p),
        active != 0 && !gbool(FN(P_NEEDS_CHARGE), p), cooldown > 0, gbool(FN(P_CONTROLS), p),
        G0(int, P_HEAD_DIR, p), G0(int, P_FIRE_DIR, p), G0(int, P_MOVE_DIR, p),
        G0(int, ANM2_FRAME, G0(const void *, E_SPRITE, p)), gbool(FN(E_IS_DEAD), p),
        FIELD(int32_t, p, PL_FIRE_DELAY), cooldown, FIELD(int32_t, p, PL_CHARGE), obs_weapon_mask(p),
        G0(int, P_PLAYER_TYPE, p),
    };
    memcpy(obs_buf + off, v, sizeof v);
}

// Lua: charge = abp_player_charge(i): Isaac.GetPlayer(i)'s charge counter (the bridge's lean_charge), nil without it.
static int obs_lua_charge(lua_State *L) {
    const void *p = ((const void *(*)(int))FN(P_GET))((int)LTOINT(L, 1));
    if (!p) {
        LPUSHNIL(L);
        return 1;
    }
    LPUSHINT(L, FIELD(int32_t, p, PL_CHARGE));
    return 1;
}

// Lua: fixed = abp_native_lean(logic_frames, extra_flags) -- header (flags | extra_flags) .. room .. totals .. players
// .. doors .. entities, no lasers; or nil, reason (the Lua code then builds it). extra_flags: 2 terrain block, 8 map
// block (the parts the bridge appends). The bridge does not call it while a player has a cancelled lethal hit.
// The fixed part into obs_buf: NULL and its length in *len, or why the Lua code has to build it (obs_fallbacks counted).
static const char *obs_fixed(long long lf, int extra, size_t *len);

static int obs_lua_lean(lua_State *L) {
    obs_calls++;
    size_t len = 0;
    const char *why = obs_fixed(LTOINT(L, 1), (int)LTOINT(L, 2), &len);
    if (why) return obs_fallback(L, why);
    LPUSHSTR(L, obs_buf, len);
    return 1;
}

static const char *obs_fixed(long long lf, int extra, size_t *len) {
    const char *why = NULL;
#define OBS_GIVE_UP(reason) do { why = (reason); goto give_up; } while (0)
    uintptr_t game = obs_game();
    const void *room = game ? G0(const void *, G_ROOM, (const void *)game) : NULL;
    const void *level = game ? G0(const void *, G_LEVEL, (const void *)game) : NULL;
    const void *player0 = ((const void *(*)(int))FN(P_GET))(0);
    if (!room || !level || !player0) OBS_GIVE_UP("no room or player");
    if (lf < 0 || lf > 0xffffffffLL) OBS_GIVE_UP("logic frames");
    uint32_t n_players = (uint32_t)G0(long, G_NUM_PLAYERS, (const void *)game);
    if (n_players > 255) OBS_GIVE_UP("players");
    int game_frame = G0(int, G_FRAME, (const void *)game);
    if (game_frame < 0) OBS_GIVE_UP("game frame");   // string.pack "I4" refuses it
    // header 16 bytes, room 48, totals 32; the players start at 96 (Isaac.GetPlayer(i)), OBS_PLAYER_BYTES each
    obs_reserve(96 + OBS_PLAYER_BYTES * (size_t)n_players + 8 * 16);
    size_t off = 96;
    for (uint32_t i = 0; i < n_players; i++) {
        const void *p = ((const void *(*)(int))FN(P_GET))((int)i);
        if (!p) OBS_GIVE_UP("player nil");
        obs_player(p, off);
        off += OBS_PLAYER_BYTES;
    }
    // doors (visible ones: not DOOR_HIDDEN)
    int n_doors = 0;
    for (int slot = 0; slot < 8; slot++) {
        const void *d = G1(const void *, R_DOOR, room, slot);
        if (!d || G0(uint32_t, GE_VARIANT, d) == 7) continue;
        vec2f pos = G0(vec2f, GE_POSITION, d);
        int seen = 0;
        const void *behind = G1(const void *, L_ROOM_BY_IDX, level, FIELD(int32_t, d, DOOR_TARGET_INDEX));
        if (behind && FIELD(uint32_t, behind, DESC_VISITED) > 0) seen = gbool(FN(D_CLEAR), behind) ? 3 : 1;
        obs_buf[off] = (uint8_t)slot;
        obs_buf[off + 1] = (uint8_t)gbool(FN(DOOR_OPEN), d);
        obs_buf[off + 2] = (uint8_t)gbool(FN(DOOR_LOCKED), d);
        obs_buf[off + 3] = (uint8_t)seen;
        PUT(off + 4, F2F(pos.x));
        PUT(off + 8, F2F(pos.y));
        PUT(off + 12, FIELD(int32_t, d, DOOR_TARGET_TYPE));
        off += 16;
        n_doors++;
    }
    double monsters_hp = 0, blocking_hp = 0;
    long long blocking_count = 0;
    why = obs_entities((uintptr_t)room, &off, &monsters_hp, &blocking_hp, &blocking_count);
    if (why) goto give_up;
    // header, room, totals
    int clear = gbool(FN(R_CLEAR), room);
    int flags = (clear ? 1 : 0) + (extra & 26) + (gbool(FN(G_PAUSED), (const void *)game) ? 4 : 0);   // 16: items
    PUT(0, (uint32_t)LEAN_MAGIC);
    PUT(4, (uint32_t)lf);
    PUT(8, (uint32_t)game_frame);
    obs_buf[12] = (uint8_t)flags;
    obs_buf[13] = (uint8_t)n_players;
    obs_buf[14] = (uint8_t)n_doors;
    obs_buf[15] = 0;   // lasers: the native path has none
    vec2f tl = G0(vec2f, R_TOPLEFT, room), br = G0(vec2f, R_BOTTOMRIGHT, room);
    int32_t r[8] = {G0(int, R_TYPE, room), G0(int, R_SHAPE, room), G0(int, R_WIDTH, room), G0(int, R_HEIGHT, room),
                    obs_room_index(game), G0(int, L_STAGE, level), G0(int, R_ALIVE, room), G0(int, R_FRAME, room)};
    memcpy(obs_buf + 16, r, sizeof r);
    PUT(48, F2F(tl.x));
    PUT(52, F2F(tl.y));
    PUT(56, F2F(br.x));
    PUT(60, F2F(br.y));
    PUT(64, (double)G0(uint32_t, P_DAMAGE_TAKEN, player0));
    PUT(72, monsters_hp);
    PUT(80, blocking_hp);
    PUT(88, (double)blocking_count);
    *len = off;
    return NULL;
give_up:
    obs_fallbacks++;
    return why;
#undef OBS_GIVE_UP
}

// ---- native step (abp_bridge.lua native_step_obs) ----
// Lua: code, ... = abp_native_step(fd, logic_frames, seq). Builds the fixed part (as abp_native_lean(logic_frames, 0)),
// writes "L <len> step <seq>\n" + payload to the client socket fd (luasocket's, non-blocking; its own buffer is empty),
// then waits for the next command. A step line ("S <repeat> <move> <shoot> <bomb> <item>\n") is consumed and returned:
// 1, repeat, move, shoot, bomb, item. Anything else stays in the socket for luasocket: 2. A malformed step line
// (consumed): 3. Nothing sent because the Lua code has to build the fixed part: 0, why. Socket error or closed: -1, why.
static uint64_t step_calls, step_lines, step_other, step_bad, step_io_errors;
// ABP_STEP_PROF=1 (environment, diagnostic): thread CPU time per part of a native step (getenv("ABP_STEP_PROF") reads
// it): building the fixed part, writing it, waiting for and parsing the next command, and everything between two native
// steps (the frames, the bridge's per-frame callbacks and its per-step Lua).
static int step_prof = -1;
static uint64_t sp_build, sp_send, sp_recv, sp_between, sp_n, sp_last;
static inline uint64_t thread_ns(void) {
    if (step_prof <= 0) return 0;
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int step_wait(int fd, short events) {
    struct pollfd p = {fd, events, 0};
    for (;;) {
        int r = poll(&p, 1, -1);
        if (r >= 0) return 0;
        if (errno != EINTR) return -1;
    }
}

static int step_write_all(int fd, struct iovec *iov, int n) {
    while (n > 0) {
        struct msghdr m;
        memset(&m, 0, sizeof m);
        m.msg_iov = iov;
        m.msg_iovlen = (size_t)n;
        ssize_t w = sendmsg(fd, &m, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            if ((errno == EAGAIN || errno == EWOULDBLOCK) && step_wait(fd, POLLOUT) == 0) continue;
            return -1;
        }
        while (n > 0 && (size_t)w >= iov->iov_len) {
            w -= (ssize_t)iov->iov_len;
            iov++;
            n--;
        }
        if (n > 0) {
            iov->iov_base = (char *)iov->iov_base + w;
            iov->iov_len -= (size_t)w;
        }
    }
    return 0;
}

// Only the bytes of a step line are consumed (never past its newline): luasocket reads every other command itself.
static int step_read(int fd, long long v[6], const char **why) {
    char line[64];
    size_t have = 0;
    for (;;) {
        char peek[64];
        ssize_t n = recv(fd, peek, sizeof peek, MSG_PEEK);
        if (n == 0) {
            *why = "closed";
            return -1;
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            if ((errno == EAGAIN || errno == EWOULDBLOCK) && step_wait(fd, POLLIN) == 0) continue;
            *why = "recv failed";
            return -1;
        }
        if (have == 0 && peek[0] != 'S') return 2;
        char *nl = memchr(peek, '\n', (size_t)n);
        size_t take = nl ? (size_t)(nl - peek) + 1 : (size_t)n;
        if (have + take >= sizeof line) {
            *why = "step line too long";
            return -1;
        }
        size_t got = 0;
        while (got < take) {
            ssize_t r = recv(fd, line + have + got, take - got, 0);
            if (r > 0) {
                got += (size_t)r;
            } else if (r < 0 && errno == EINTR) {
                continue;
            } else {
                *why = "recv failed";
                return -1;
            }
        }
        have += take;
        if (nl) break;
        if (step_wait(fd, POLLIN) != 0) {
            *why = "poll failed";
            return -1;
        }
    }
    line[have - 1] = 0;
    // "^S (%d+) (%d+) (%d+) (%d+) (%d+)$" as the bridge's Lua parse (numbers of up to 9 digits here); 2026-10-06 (items,
    // Phase A): an optional sixth number, the pill / card action ("S <repeat> <move> <shoot> <bomb> <item> <pill>");
    // without it the pill is 0
    const char *c = line + 1;
    v[5] = 0;
    for (int i = 0; i < 6; i++) {
        if (i == 5 && *c == 0) break;
        if (*c != ' ') return 3;
        c++;
        if (*c < '0' || *c > '9') return 3;
        long long x = 0;
        int digits = 0;
        while (*c >= '0' && *c <= '9') {
            x = x * 10 + (*c - '0');
            c++;
            if (++digits > 9) return 3;
        }
        v[i] = x;
    }
    return *c == 0 ? 1 : 3;
}

static int obs_lua_step(lua_State *L) {
    if (step_prof < 0) {
        const char *e = real_getenv ? real_getenv("ABP_STEP_PROF") : NULL;
        step_prof = e && atoi(e) > 0;
    }
    uint64_t p0 = thread_ns();
    if (step_prof && sp_last && LTOBOOL(L, 4)) {   // 4th argument: the step before was a native one too
        sp_between += p0 - sp_last;
        sp_n++;
    }
    step_calls++;
    obs_calls++;
    int fd = (int)LTOINT(L, 1);
    long long lf = LTOINT(L, 2), seq = LTOINT(L, 3);
    size_t len = 0;
    const char *why = obs_fixed(lf, 0, &len);
    if (why) {
        LPUSHINT(L, 0);
        LPUSHSTR(L, why, strlen(why));
        return 2;
    }
    char head[64];
    int hn = snprintf(head, sizeof head, "L %zu step %lld\n", len, seq);
    struct iovec iov[2] = {{head, (size_t)hn}, {obs_buf, len}};
    uint64_t p1 = thread_ns();
    int wrote = step_write_all(fd, iov, 2);
    uint64_t p2 = thread_ns();
    if (wrote != 0) {
        step_io_errors++;
        LPUSHINT(L, -1);
        LPUSHSTR(L, "send failed", 11);
        return 2;
    }
    long long v[6];
    const char *err = "";
    int r = step_read(fd, v, &err);
    if (step_prof) {
        uint64_t p3 = thread_ns();
        sp_build += p1 - p0;
        sp_send += p2 - p1;
        sp_recv += p3 - p2;
        sp_last = p3;
    }
    if (r == 1) {
        step_lines++;
        LPUSHINT(L, 1);
        for (int i = 0; i < 6; i++) LPUSHINT(L, v[i]);
        return 7;
    }
    if (r == 2 || r == 3) {
        if (r == 2) step_other++; else step_bad++;
        LPUSHINT(L, r);
        return 1;
    }
    step_io_errors++;
    LPUSHINT(L, -1);
    LPUSHSTR(L, err, strlen(err));
    return 2;
}

// Lua: abp_pu_arm(n): the next n MC_POST_UPDATE calls are answered by pu_gate (0 disarms); abp_pu_take() -> how many
// were answered since the last take (the bridge adds them to its frame count and its step's frames left).
static int obs_lua_pu_arm(lua_State *L) {
    long long n = LTOINT(L, 1);
    pu_skip = n > 0 ? (long)n : 0;
    pu_arms += n > 0;
    return 0;
}

static int obs_lua_pu_take(lua_State *L) {
    long k = pu_skipped;
    pu_skipped = 0;
    LPUSHINT(L, k);
    return 1;
}

// getenv("ABP_OBS_INIT") from the bridge (inside a Lua call, on the Lua state's own thread): checks every address once
// and registers the Lua globals abp_native_lean and abp_native_terrain (and abp_native_step, abp_native_terrain_key,
// abp_player_charge (2026-10-07), abp_native_player_f (2026-10-08), abp_pu_arm / abp_pu_take). Returns "1", or nil when refused
// (getenv("ABP_OBS_STATUS") says why).
static char *obs_init(void) {
    if (obs_ready < 0) obs_ready = obs_check();
    if (!obs_ready) return NULL;
    uintptr_t engine = *(volatile uintptr_t *)G_LUAENGINE;
    lua_State *L = engine ? *(lua_State **)engine : NULL;
    if (!L) return NULL;
    ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_lean, 0);
    ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_native_lean");
    ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_terrain, 0);
    ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_native_terrain");
    ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_step, 0);
    ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_native_step");
    ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_tkey, 0);
    ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_native_terrain_key");
    ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_charge, 0);
    ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_player_charge");
    // 2026-10-08 (bridge abp-0.2.17): the doubles of a player record obs_player writes; the bridge uses the native obs
    // only when this equals its own LEAN_PLAYER_F
    LPUSHINT(L, OBS_PLAYER_F);
    ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_native_player_f");
    if (pu_patched) {
        ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_pu_arm, 0);
        ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_pu_arm");
        ((void (*)(lua_State *, lua_cfn, int))FN(PUSHCCLOSURE))(L, obs_lua_pu_take, 0);
        ((void (*)(lua_State *, const char *))FN(SETGLOBAL))(L, "abp_pu_take");
    }
    return getenv_ok;
}

// Frame grab (2026-10-06, footage of recorded episodes; off unless switched on, nothing the game reads):
// getenv("ABP_GRAB_ON:<every>:<w>:<h>:<path>") from game code opens <path> for writing (a FIFO an encoder reads, e.g.
// ffmpeg -f rawvideo -pix_fmt rgb24 -s <w>x<h>; the open waits for the reader) and from then on every <every>-th
// presented frame (glXSwapBuffers, real rendering only: not with ABP_NULLGL, not in a clone) is read back from the
// default framebuffer's back buffer before the swap (glReadPixels, the lower-left <w> x <h> pixels, RGB, rows bottom
// up) and written there. The GL state it touches (read framebuffer, read buffer, pack alignment / row length / skips,
// pixel pack buffer) is restored. getenv("ABP_GRAB_OFF") closes it; both, and getenv("ABP_GRAB_STATUS"), answer
// "frames=<written> swaps=<seen while on> win=<window w>x<h> err=<write errors>". SIGPIPE is ignored while it is on.
typedef void (*gl_getiv_fn)(unsigned, int *);
typedef void (*gl_u_fn)(unsigned);
typedef void (*gl_uu_fn)(unsigned, unsigned);
typedef void (*gl_ui_fn)(unsigned, int);
typedef void (*gl_read_fn)(int, int, int, int, unsigned, unsigned, void *);
static int grab_fd = -1, grab_every = 1, grab_w, grab_h, grab_win_w, grab_win_h;
static unsigned long long grab_frames, grab_swaps, grab_errors;
static unsigned char *grab_buf;
static struct sigaction grab_old_pipe;
static char grab_answer[160];

static char *grab_status(void) {
    snprintf(grab_answer, sizeof grab_answer, "frames=%llu swaps=%llu win=%dx%d err=%llu", grab_frames, grab_swaps,
             grab_win_w, grab_win_h, grab_errors);
    return grab_answer;
}

static char *grab_gate(const char *arg) {
    if (strcmp(arg, "_STATUS") == 0) return grab_status();
    if (strcmp(arg, "_OFF") == 0) {
        if (grab_fd >= 0) {
            close(grab_fd);
            grab_fd = -1;
            sigaction(SIGPIPE, &grab_old_pipe, NULL);
        }
        return grab_status();
    }
    if (strncmp(arg, "_ON:", 4) != 0 || is_child) return NULL;
    int every = 0, w = 0, h = 0, n = 0;
    if (sscanf(arg + 4, "%d:%d:%d:%n", &every, &w, &h, &n) != 3 || n <= 0 || every < 1 || w < 1 || h < 1 ||
        w > 8192 || h > 8192) return NULL;
    if (grab_fd >= 0) return NULL;
    unsigned char *buf = realloc(grab_buf, (size_t)w * h * 3);
    if (!buf) return NULL;
    grab_buf = buf;
    struct sigaction ign;
    memset(&ign, 0, sizeof ign);
    ign.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &ign, &grab_old_pipe);
    // non-blocking open: a FIFO nobody reads yet fails (ENXIO; the caller tries again) instead of stopping the game
    int fd = open(arg + 4 + n, O_WRONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
        sigaction(SIGPIPE, &grab_old_pipe, NULL);
        return NULL;
    }
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);   // writes wait for the reader
    grab_fd = fd;
    grab_every = every;
    grab_w = w;
    grab_h = h;
    grab_frames = grab_swaps = grab_errors = 0;
    grab_win_w = grab_win_h = 0;
    return grab_status();
}

static void grab_frame(void *dpy, unsigned long drawable) {
    static gl_getiv_fn getiv;
    static gl_u_fn read_buffer;
    static gl_uu_fn bind_fb, bind_buf;
    static gl_ui_fn pixel_store;
    static gl_read_fn read_pixels;
    if (grab_swaps++ % (unsigned long long)grab_every != 0) return;
    if (!read_pixels) {
        typedef void *(*getproc_fn)(const unsigned char *);
        getproc_fn getproc = (getproc_fn)dlsym(RTLD_DEFAULT, "glXGetProcAddressARB");
        if (!getproc) { grab_errors++; return; }
        getiv = (gl_getiv_fn)getproc((const unsigned char *)"glGetIntegerv");
        read_buffer = (gl_u_fn)getproc((const unsigned char *)"glReadBuffer");
        bind_fb = (gl_uu_fn)getproc((const unsigned char *)"glBindFramebuffer");
        bind_buf = (gl_uu_fn)getproc((const unsigned char *)"glBindBuffer");
        pixel_store = (gl_ui_fn)getproc((const unsigned char *)"glPixelStorei");
        read_pixels = (gl_read_fn)getproc((const unsigned char *)"glReadPixels");
        if (!getiv || !read_buffer || !bind_fb || !bind_buf || !pixel_store || !read_pixels) {
            read_pixels = NULL;
            grab_errors++;
            return;
        }
    }
    if (!grab_win_w) {
        typedef int (*xgwa_raw_fn)(void *, unsigned long, void *);
        static xgwa_raw_fn xgwa;
        if (!xgwa) xgwa = (xgwa_raw_fn)dlsym(RTLD_NEXT, "XGetWindowAttributes");
        unsigned char attrs[136];
        memset(attrs, 0, sizeof attrs);
        if (xgwa && xgwa(dpy, drawable, attrs)) {
            memcpy(&grab_win_w, attrs + 8, 4);   // XWindowAttributes: int x, y, width, height, ...
            memcpy(&grab_win_h, attrs + 12, 4);
        } else {
            grab_win_w = grab_win_h = -1;
        }
    }
    int fb = 0, rb = 0, align = 4, row_len = 0, skip_rows = 0, skip_px = 0, pbo = 0;
    getiv(0x8CAA, &fb);         // GL_READ_FRAMEBUFFER_BINDING
    getiv(0x0C02, &rb);         // GL_READ_BUFFER
    getiv(0x0D05, &align);      // GL_PACK_ALIGNMENT
    getiv(0x0D02, &row_len);    // GL_PACK_ROW_LENGTH
    getiv(0x0D03, &skip_rows);  // GL_PACK_SKIP_ROWS
    getiv(0x0D04, &skip_px);    // GL_PACK_SKIP_PIXELS
    getiv(0x88ED, &pbo);        // GL_PIXEL_PACK_BUFFER_BINDING
    bind_fb(0x8CA8, 0);         // GL_READ_FRAMEBUFFER: the window's
    read_buffer(0x0405);        // GL_BACK
    bind_buf(0x88EB, 0);        // GL_PIXEL_PACK_BUFFER
    pixel_store(0x0D05, 1);
    pixel_store(0x0D02, 0);
    pixel_store(0x0D03, 0);
    pixel_store(0x0D04, 0);
    read_pixels(0, 0, grab_w, grab_h, 0x1907, 0x1401, grab_buf);   // GL_RGB, GL_UNSIGNED_BYTE
    pixel_store(0x0D05, align);
    pixel_store(0x0D02, row_len);
    pixel_store(0x0D03, skip_rows);
    pixel_store(0x0D04, skip_px);
    bind_buf(0x88EB, (unsigned)pbo);
    bind_fb(0x8CA8, (unsigned)fb);
    read_buffer((unsigned)rb);
    size_t left = (size_t)grab_w * grab_h * 3;
    const unsigned char *p = grab_buf;
    while (left > 0) {
        ssize_t k = write(grab_fd, p, left);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) { grab_errors++; return; }
        p += k;
        left -= (size_t)k;
    }
    grab_frames++;
}

char *getenv(const char *name) {
    if (name[0] == 'A' && strncmp(name, "ABP_", 4) == 0 && from_game(__builtin_return_address(0))) {
        if (strcmp(name, "ABP_FORK") == 0) {
            return fork_game();
        }
        if (strncmp(name, "ABP_GRAB", 8) == 0) return grab_gate(name + 8);
        if (strcmp(name, "ABP_OBS_INIT") == 0) {
            return obs_init();
        }
        if (strcmp(name, "ABP_FM_OPEN") == 0) return fm_open();
        if (strncmp(name, "ABP_FM_RESULT:", 14) == 0) return fm_result(name + 14);
        if (strncmp(name, "ABP_FM_COLLECT:", 15) == 0) return fm_collect(atol(name + 15), strstr(name + 15, ":w") != NULL);
        if (strcmp(name, "ABP_FM_PROBE") == 0) return getenv_ok;
        if (strncmp(name, "ABP_FORK_LITE:", 14) == 0) {   // see fork_lite; answers "<lite forks> <refused>"
            static char fl[64];
            fork_lite = atoi(name + 14) > 0;
            snprintf(fl, sizeof fl, "%llu %llu", (unsigned long long)fork_lite_n, (unsigned long long)fork_lite_refused);
            return fl;
        }
        if (strncmp(name, "ABP_MALLOC_INFO", 15) == 0) {   // diagnostic: "<arena> <in use> <free> <top free> <mmapped>"
            // bytes from mallinfo2; ABP_MALLOC_INFO:TRIM calls malloc_trim(0) first (returns free pages to the system)
            static char mi[200];
            int trimmed = -1;
            if (strcmp(name + 15, ":TRIM") == 0) trimmed = malloc_trim(0);
            struct mallinfo2 m = mallinfo2();
            snprintf(mi, sizeof mi, "%zu %zu %zu %zu %zu %d", m.arena, m.uordblks, m.fordblks, m.keepcost, m.hblkhd,
                     trimmed);
            return mi;
        }
        if (strcmp(name, "ABP_FORK_TIMES") == 0) {   // diagnostic, see ft_*
            // "<forks> <fork_ns> <mutex_hold_ns> <fork_syscall_ns> <collect_ns> <isolate_ns> <since_fork_ns>
            //  <since_born_ns> <cpu_ns> <runqueue_wait_ns> <minflt> <majflt> <registered mutexes> <reaped_cpu_ns>":
            //  cpu / wait from /proc/self/schedstat; reaped: CPU of the clones this process has reaped, exit included
            static char ft[320];
            unsigned long long cpu = 0, wait = 0;
            FILE *f = fopen("/proc/self/schedstat", "r");
            if (f) {
                if (fscanf(f, "%llu %llu", &cpu, &wait) != 2) cpu = wait = 0;
                fclose(f);
            }
            struct rusage ru;
            memset(&ru, 0, sizeof ru);
            getrusage(RUSAGE_SELF, &ru);
            uint64_t now = real_ns(CLOCK_MONOTONIC);
            snprintf(ft, sizeof ft, "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %ld %ld %d %llu",
                     (unsigned long long)ft_n, (unsigned long long)ft_total_ns, (unsigned long long)ft_hold_ns,
                     (unsigned long long)ft_sys_ns, (unsigned long long)ft_collect_ns,
                     (unsigned long long)(is_child ? ft_isolate_ns : 0),
                     (unsigned long long)(is_child ? now - ft_t0 : 0),
                     (unsigned long long)(is_child ? now - ft_born_ns : 0), cpu, wait, ru.ru_minflt, ru.ru_majflt,
                     n_game_mutexes, (unsigned long long)ft_reaped_ns);
            return ft;
        }
        if (strcmp(name, "ABP_PROF_FLUSH") == 0) {   // diagnostic: write the profile and a stats line now
            write_stats(1);
            return getenv_ok;
        }
        if (strncmp(name, "ABP_FAST:", 9) == 0) return fast_apply((unsigned)strtoul(name + 9, NULL, 0));
        if (strcmp(name, "ABP_FAST_STATUS") == 0) return fast_status();
        if (strcmp(name, "ABP_PU_STATUS") == 0) {
            static char pu[160];
            snprintf(pu, sizeof pu, "patched=%d calls=%llu skipped=%llu arms=%llu pending=%ld untaken=%ld", pu_patched,
                     (unsigned long long)pu_calls, (unsigned long long)pu_total_skipped, (unsigned long long)pu_arms,
                     (long)pu_skip, pu_skipped);
            return pu;
        }
        if (strncmp(name, "ABP_STUBS:", 10) == 0) {   // apply a stub list now (exactness probes: one clone of two)
            static char applied[16];
            snprintf(applied, sizeof applied, "%d", apply_stub_file(name + 10));
            return applied;
        }
        if (strcmp(name, "ABP_STEP_PROF") == 0) {
            static char sp[200];
            snprintf(sp, sizeof sp, "n=%llu build_ns=%llu send_ns=%llu recv_ns=%llu between_ns=%llu calls=%llu",
                     (unsigned long long)sp_n, (unsigned long long)sp_build, (unsigned long long)sp_send,
                     (unsigned long long)sp_recv, (unsigned long long)sp_between, (unsigned long long)step_calls);
            return sp;
        }
        if (strcmp(name, "ABP_OBS_STATUS") == 0) {
            static char why[400];
            snprintf(why, sizeof why, "ready=%d calls=%llu fallbacks=%llu terrain_calls=%llu step_calls=%llu "
                     "step_lines=%llu step_other=%llu step_bad=%llu step_io_errors=%llu %s", obs_ready,
                     (unsigned long long)obs_calls, (unsigned long long)obs_fallbacks,
                     (unsigned long long)obs_terrain_calls, (unsigned long long)step_calls,
                     (unsigned long long)step_lines, (unsigned long long)step_other, (unsigned long long)step_bad,
                     (unsigned long long)step_io_errors, obs_refusal);
            return why;
        }
        if (strncmp(name, "ABP_INPUT:", 10) == 0) {
            char *end = NULL;
            input_held = (uint32_t)strtoul(name + 10, &end, 10);
            input_triggered = end && *end == ':' ? (uint32_t)strtoul(end + 1, NULL, 10) : 0;
            return getenv_ok;
        }
        if (strncmp(name, "ABP_INPUT_ON:", 13) == 0) {
            if (input_patched != 3) return NULL;
            input_mask = (uint32_t)strtoul(name + 13, NULL, 10);
            input_held = input_triggered = 0;
            native_input = 1;
            return getenv_ok;
        }
        if (strcmp(name, "ABP_INPUT_OFF") == 0) {
            native_input = 0;
            return getenv_ok;
        }
        if (strcmp(name, "ABP_EXIT") == 0) {
            if (is_child) _exit(0);
            return NULL;
        }
        if (strncmp(name, "ABP_ALARM:", 10) == 0) {
            if (!is_child) return NULL;
            long left = atol(name + 10);
            alarm(left > 0 ? (unsigned)left : 0);
            return getenv_ok;
        }
        if (strncmp(name, "ABP_MUTEX_PROBE:", 16) == 0) {
            // Diagnostic: <n> rounds of try-locking every registered game mutex (released at once). Answers "rounds
            // busy_rounds busy_total sound_registered": how often another thread holds a game mutex at a random
            // moment, and whether the sound manager's two locks (KAGE Mutex objects at 0xe025e8, 0xe025d0) are in
            // the table.
            static char mp[96];
            long rounds = atol(name + 16), busy_rounds = 0, busy_total = 0;
            if (!real_mutex_lock) resolve_mutex();
            pthread_mutex_t *s1 = *(pthread_mutex_t **)(0xe025e8UL + 0x10), *s2 = *(pthread_mutex_t **)(0xe025d0UL + 0x10);
            int found = 0;
            registry_acquire();
            for (int i = 0; i < n_game_mutexes; i++) found += game_mutexes[i] == s1 || game_mutexes[i] == s2;
            for (long r = 0; r < rounds; r++) {
                int busy = 0;
                for (int i = 0; i < n_game_mutexes; i++) {
                    if (real_mutex_trylock(game_mutexes[i]) == 0) real_mutex_unlock(game_mutexes[i]);
                    else busy++;
                }
                busy_rounds += busy > 0;
                busy_total += busy;
            }
            registry_release();
            snprintf(mp, sizeof mp, "%ld %ld %ld %d", rounds, busy_rounds, busy_total, found);
            return mp;
        }
        if (strcmp(name, "ABP_FORK_COUNTERS") == 0) {
            static char fc[640];
            reap_children();
            snprintf(fc, sizeof fc, "child=%d pid=%d forks=%llu failed=%llu live=%d isolated_fds=%llu fork_us=%llu "
                     "exit0=%llu signalled=%llu last_signal=%d other=%llu mutexes=%d inits=%llu destroys=%llu "
                     "dropped=%llu missed=%llu self=%llu passes=%llu clone_passed=%llu clone_writes=%llu "
                     "steam_stubs=%d native_input=%d input_answers=%llu input_passed=%llu", is_child, (int)getpid(),
                     (unsigned long long)forks, (unsigned long long)fork_failures, n_children,
                     (unsigned long long)isolated_fds, (unsigned long long)fork_last_us,
                     (unsigned long long)clones_exit0, (unsigned long long)clones_signalled, clone_last_signal,
                     (unsigned long long)clones_other, n_game_mutexes, (unsigned long long)mutex_inits,
                     (unsigned long long)mutex_destroys, (unsigned long long)mutex_dropped,
                     (unsigned long long)fork_mutex_missed, (unsigned long long)fork_mutex_self,
                     (unsigned long long)fork_mutex_passes, (unsigned long long)clone_lock_passed,
                     (unsigned long long)clone_writes, clone_stubbed, native_input,
                     (unsigned long long)input_answers, (unsigned long long)input_passed);
            return fc;
        }
    }
    if (game_init_genrand && strncmp(name, "ABP_RESEED:", 11) == 0 &&
        from_game(__builtin_return_address(0))) {
        game_init_genrand(strtoul(name + 11, NULL, 10));
        reseeds++;
        start_armed = 0;
        return getenv_ok;
    }
    if (strncmp(name, "ABP_STARTSEED:", 14) == 0 && from_game(__builtin_return_address(0))) {
        if (!start_patched || !game_init_genrand) {
            return NULL;
        }
        start_value = strtoul(name + 14, NULL, 10);
        start_armed = 1;
        return getenv_ok;
    }
    if (strncmp(name, "ABP_PLAYERTYPE:", 15) == 0 && from_game(__builtin_return_address(0))) {   // 2026-10-08
        long t = strtol(name + 15, NULL, 10);
        if (!ptype_patched || t < 0 || t > 255) {
            return NULL;
        }
        ptype_armed = (int)t;
        return getenv_ok;
    }
    if (strcmp(name, "ABP_SOUND_RESET") == 0 && from_game(__builtin_return_address(0))) {
        // SoundEffects::Play plays a sound, and draws Random(variants) for it, only if its stamp (+4) is <= the
        // Manager's logic frame or negative; the stamps outlive episodes (A7). All set to -1 ("may play").
        uintptr_t mgr = *(volatile uintptr_t *)G_MANAGER;
        uintptr_t begin = mgr ? *(volatile uintptr_t *)(mgr + MANAGER_SOUNDS) : 0;
        uintptr_t end = mgr ? *(volatile uintptr_t *)(mgr + MANAGER_SOUNDS + 8) : 0;
        if (!begin || end < begin || (end - begin) % SOUND_SIZE != 0) {
            return NULL;
        }
        for (uintptr_t s = begin; s < end; s += SOUND_SIZE) {
            *(volatile int32_t *)(s + 4) = -1;
        }
        return getenv_ok;
    }
    if (strcmp(name, "ABP_MT") == 0 && from_game(__builtin_return_address(0))) {
        // "<index> <FNV-1a of the 624 state words>": compares the global MT19937 across runs (A7 diagnostics)
        static char mt[48];
        const volatile uint64_t *state = (const volatile uint64_t *)MT_STATE;
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 624; i++) {
            h = (h ^ (uint32_t)state[i]) * 1099511628211ULL;
        }
        snprintf(mt, sizeof mt, "%d %016llx", *(volatile int *)MT_INDEX, (unsigned long long)h);
        return mt;
    }
    if (strcmp(name, "ABP_TURBO_COUNTERS") == 0 && from_game(__builtin_return_address(0))) {
        static char counters[256];
        snprintf(counters, sizeof counters,
                 "reseeds=%llu start_patched=%d start_draws=%llu start_overrides=%llu al_stopped=%llu "
                 "ptype_patched=%d ptype_overrides=%llu",
                 (unsigned long long)reseeds, start_patched, (unsigned long long)start_draws,
                 (unsigned long long)start_overrides, (unsigned long long)al_stopped_answers, ptype_patched,
                 (unsigned long long)ptype_overrides);
        return counters;
    }
    if (real_getenv) {
        return real_getenv(name);
    }
    // Before __libc_start_main resolved the real one (other libraries' constructors).
    size_t n = strlen(name);
    for (char **e = environ; n && e && *e; e++) {
        if (strncmp(*e, name, n) == 0 && (*e)[n] == '=') {
            return *e + n + 1;
        }
    }
    return NULL;
}

// ---- abp_row_encode: tok_obs.encode_row in C, for the sampler workers (Python, ctypes), not for the game ----
// The workers load this library with ctypes (RTLD_LOCAL; nothing in it runs at load: the game hooks start in
// __libc_start_main, which only the game calls) and call abp_row_encode on a lean payload (abplus_lean.py's layout)
// instead of LeanDecoder.decode + encode_row. Same values bit for bit (float32 / float64 arithmetic in the order numpy
// does it; checked by abplus_probe_fast_row.py on every record of real episodes), same bytes written (the entity and
// door rows past the counts are left as they are, as encode_row leaves them). Only the common case is done here: a
// negative return hands the record to encode_row (tok_obs.FastRow): a map block or a terrain block that differs from
// the last one in the payload, the first record of an episode, a room change, more than ENT_CAP entities, door records
// other than the ones the grid was built with, a floor episode's first record, an unexpected layout. ctx (doubles,
// tok_obs.FastRow): see the RC_ indices; refs: address and length of the last terrain block's bytes (without its
// version: LeanDecoder.terrain_raw) and of the door records the grid was built with (EpisodeState.version[1]).
// Version 3 (2026-10-06, items / run mode, Phase A): RC_RUN (a run episode: a stage change is not the end, the record
// is handed back to encode_row: -8), RC_ITEMS (0 off, 1 on, 2 on and Curse of the Blind: pedestal ids hidden), a
// payload with the inventory block (flag 16) is handed back (-2), and RO_ENT_ITEM (offs < 0: not written): per entity
// (collectible id of a pedestal (pickup variant 100), 1023 when hidden; trinket id of a trinket (variant 350)), int16.
// With RC_RUN = RC_ITEMS = 0 and offs[RO_ENT_ITEM] < 0 the records are those of version 2.
// Version 4 (2026-10-07, charge): the lean player record has 38 doubles (charge counter, weapon-type bits) and
// RO_PCHARGE (offs < 0: not written) gets tok_obs.encode_row's 'pcharge' (PCHARGE_F float32).
// Version 5 (2026-10-08, abp-0.2.16): lean entity records of LEAN_ENTITY_BYTES (flag word, laser geometry) and RE_F = 56
// entity columns: the 33 of before unchanged, then 16 flag columns (tok_obs.TEAR_FLAG_MASKS for tears and lasers,
// PROJ_FLAG_MASKS for projectiles: 1 when the flag word has a bit of the mask) and 7 laser columns (cos, sin,
// length / 300, end point relative to the player / 200, circle, radius / 100; 0 for everything but a laser).
// Version 6 (2026-10-08, bridge abp-0.2.17, character randomisation): the lean player record has RE_PLAYER_F = 39
// doubles (the 39th the PlayerType) and RO_PCHAR (offs < 0: not written) gets tok_obs.encode_row's 'pchar' (int16, the
// PlayerType clipped to 0 .. RE_N_CHAR - 1).
enum {
    RC_T, RC_LIMIT, RC_FLOOR, RC_STALL, RC_PROGRESS, RC_STAGE0, RC_HP0, RC_PREV_DMG, RC_PREV_HP, RC_HAVE_PREV, RC_ROOM,
    RC_WAS_CLEAR, RC_ORIGIN_X, RC_ORIGIN_Y, RC_N_EXITS, RC_EXITS, RC_TVERSION = RC_EXITS + 16, RC_TSEEN, RC_HURT,
    RC_EPISODE, RC_SEED, RC_GROUP, RC_FIRST, RC_RUN, RC_ITEMS, RC_COUNT
};
// offs: byte offsets of the ROW fields in this order
enum { RO_PLAYER, RO_ENT, RO_ENT_ID, RO_DOORS, RO_PATCH, RO_GRID, RO_MAP, RO_N_ENT, RO_N_DOORS, RO_T, RO_HURT, RO_DAMAGE,
       RO_DONE, RO_BOMBS, RO_EVENTS, RO_EPISODE, RO_SEED, RO_GROUP, RO_FIRST, RO_ENT_ITEM, RO_PCHARGE, RO_PCHAR,
       RO_COUNT };
#define RE_PCHARGE_F 12
#define RE_PLAYER_F 39         // doubles of a lean player record (version 6; 38 in versions 4 and 5)
#define RE_N_CHAR 32           // tok_obs.N_CHAR: 'pchar' is clipped to 0 .. RE_N_CHAR - 1
#define RE_ITEM_UNKNOWN 1023   // a pedestal's collectible id under Curse of the Blind (the game shows "?")
#define RE_CAP 64
#define RE_F0 33            // the entity columns before version 5
#define RE_FLAG_F 16
#define RE_LASER_F 7
#define RE_F (RE_F0 + RE_FLAG_F + RE_LASER_F)
// tok_obs.TEAR_FLAG_MASKS / PROJ_FLAG_MASKS (bits: AB+ resources/scripts/enums.lua TearFlags / ProjectileFlags)
#define B64(k) ((uint64_t)1 << (k))
static const uint64_t RE_TEAR_MASKS[RE_FLAG_F] = {
    B64(0), B64(1), B64(2), B64(3), B64(4), B64(5), B64(6) | B64(18), B64(7), B64(8), B64(9),
    B64(10) | B64(26) | B64(44), B64(12), B64(16), B64(19), B64(22), B64(24)};
static const uint64_t RE_PROJ_MASKS[RE_FLAG_F] = {
    B64(13), B64(4), B64(0), B64(24), B64(2) | B64(8), B64(3) | B64(10) | B64(14), B64(16) | B64(29), B64(27), B64(6),
    B64(7), B64(5) | B64(21) | B64(22) | B64(23), B64(1), B64(11) | B64(12), B64(18) | B64(19) | B64(20), B64(15),
    B64(31)};
#define RE_DOOR_CAP 8
#define RE_DOOR_F 7
#define RE_PATCH 9
#define RE_GC 7
#define RE_GH 16
#define RE_GW 28
#define RE_MAP (8 * 13 * 13)

static inline double py_min(double a, double b) { return b < a ? b : a; }   // Python's min(a, b)
static inline double py_max(double a, double b) { return b > a ? b : a; }   // Python's max(a, b)
static inline double rd_d(const unsigned char *p) { double v; memcpy(&v, p, 8); return v; }
static inline float rd_f(const unsigned char *p) { float v; memcpy(&v, p, 4); return v; }
static inline int32_t rd_i(const unsigned char *p) { int32_t v; memcpy(&v, p, 4); return v; }
static inline void wr_f(unsigned char *p, float v) { memcpy(p, &v, 4); }
static inline long floor_mod(long a, long m) { long r = a % m; return r < 0 ? r + m : r; }

int abp_row_encode_version(void) { return (6 << 16) | (RC_COUNT << 8) | RO_COUNT; }

int abp_row_encode(const unsigned char *pl, long len, unsigned char *row, double *ctx, const unsigned char *grid,
                   const unsigned char *padded, const unsigned char *map, const long *offs, const long *refs) {
    if (len < 96 || rd_i(pl) != (int32_t)LEAN_MAGIC) return -1;
    int flags = pl[12], n_players = pl[13], n_doors = pl[14], n_lasers = pl[15];
    if (flags & (8 | 16)) return -2;                 // a map or inventory block: the decoder must see it
    if (n_players < 1) return -1;
    ctx[RC_TSEEN] = 0;
    if (!(ctx[RC_HAVE_PREV] != 0)) return -4;          // the episode's first record
    const unsigned char *room = pl + 16;
    int room_type = rd_i(room), room_idx = rd_i(room + 16), stage = rd_i(room + 20);
    if ((double)room_idx != ctx[RC_ROOM]) return -4;   // another room
    int floor_ep = ctx[RC_FLOOR] != 0;
    if (floor_ep && isnan(ctx[RC_STAGE0])) return -6;
    int run_ep = floor_ep && ctx[RC_RUN] != 0;
    if (run_ep && (double)stage != ctx[RC_STAGE0]) return -8;   // a run's next floor: encode_row's EV_EXIT
    int items = (int)ctx[RC_ITEMS];
    long doors_at = 96 + 8L * RE_PLAYER_F * n_players, ents_at = doors_at + 16L * n_doors;
    if (len < ents_at + 2) return -1;
    unsigned n_ent;
    {
        uint16_t c;
        memcpy(&c, pl + ents_at, 2);
        n_ent = c;
    }
    if (n_ent > RE_CAP) return -3;
    const unsigned char *ents = pl + ents_at + 2;
    if (len < ents_at + 2 + (long)LEAN_ENTITY_BYTES * n_ent) return -1;
    // the door records, against the ones the grid was built with (encode_row's _terrain key)
    if (refs[3] != 16L * n_doors || memcmp(pl + doors_at, (const void *)refs[2], (size_t)refs[3]) != 0) return -5;
    if (flags & 2) {   // a terrain block: only the same as the last one (the bridge resends it every 30 logic frames)
        long at = ents_at + 2 + (long)LEAN_ENTITY_BYTES * n_ent;
        for (int i = 0; i < n_lasers; i++) {   // <qBddddddq + k samples of 2 doubles
            if (len < at + 65) return -1;
            int64_t k;
            memcpy(&k, pl + at + 57, 8);
            if (k < 0 || k > 100000) return -1;
            at += 65 + 16 * k;
        }
        if (!refs[0] || len - (at + 4) != refs[1] || memcmp(pl + at + 4, (const void *)refs[0], (size_t)refs[1]) != 0)
            return -2;
        ctx[RC_TVERSION] = (double)(uint32_t)rd_i(pl + at);
        ctx[RC_TSEEN] = 1;
    }
    double pv[RE_PLAYER_F];   // player 0 (abplus_lean.PLAYER_FIELDS)
    memcpy(pv, pl + 96, sizeof pv);
    double px = pv[1], py = pv[2];
    double tlx = rd_f(room + 32), tly = rd_f(room + 36), brx = rd_f(room + 40), bry = rd_f(room + 44);
    double w = py_max(brx - tlx, 1.0), h = py_max(bry - tly, 1.0);
    double damage_taken = rd_d(pl + 64), monsters_hp = rd_d(pl + 72), blocking_count = rd_d(pl + 88);
    int clear = flags & 1;
    double units = ((pv[6] + pv[8]) + pv[11]) + 2 * pv[10];
    long t = (long)ctx[RC_T];
    double limit = ctx[RC_LIMIT], stall = ctx[RC_STALL], progress_t = ctx[RC_PROGRESS];
    double pvec[31] = {
        (px - tlx) / w, (py - tly) / h, pv[3] / 10, pv[4] / 10, pv[6] / 12, pv[8] / 12,
        pv[7] / 12, units / 12, py_min(pv[15], 10) / 10, py_min(pv[16], 10) / 10, py_min(pv[14], 50) / 50,
        pv[17] / 10, pv[18] / 20, pv[19] / 2, pv[20] / 500, pv[21] / 2,
        pv[22] / 5, pv[23], pv[27], py_max(-1.0, py_min(1.0, pv[34] / 20)),
        py_min(pv[35], 120) / 60, pv[24] != 0 ? 1.0 : 0.0, pv[26], clear ? 1.0 : 0.0,
        py_min((double)t / limit, 1.0), pv[5] / 20, rd_i(room + 8) / 28.0, rd_i(room + 12) / 16.0,
        (rd_i(room + 24) < 10 ? rd_i(room + 24) : 10) / 10.0,
        py_min(blocking_count, 10) / 10,
        (floor_ep && stall > 0) ? py_min((double)(t - (long)progress_t) / stall, 1.0) : 0.0,
    };
    unsigned char *pr = row + offs[RO_PLAYER];
    for (int i = 0; i < 31; i++) wr_f(pr + 4 * i, (float)pvec[i]);
    {
        double b = py_min((double)(long)pv[15], 255);   // min(int(p['bombs']), 255)
        row[offs[RO_BOMBS]] = (unsigned char)(long)b;
    }
    if (offs[RO_PCHARGE] >= 0) {   // tok_obs.encode_row's pcharge
        unsigned char *pc = row + offs[RO_PCHARGE];
        long mask = (long)pv[37];
        wr_f(pc, (float)(py_min(pv[36], 120) / 60));
        wr_f(pc + 4, (float)(py_min(pv[36] / py_max(pv[18], 1.0), 3.0) / 3));
        for (int w = 1; w <= 10; w++) wr_f(pc + 4 * (w + 1), (float)((mask >> w) & 1));
    }
    if (offs[RO_PCHAR] >= 0) {   // version 6: tok_obs.encode_row's pchar
        long c = (long)pv[38];
        int16_t ch = (int16_t)(c < 0 ? 0 : c > RE_N_CHAR - 1 ? RE_N_CHAR - 1 : c);
        memcpy(row + offs[RO_PCHAR], &ch, 2);
    }
    // entities
    unsigned char *out = row + offs[RO_ENT];
    unsigned char *ids = row + offs[RO_ENT_ID];
    unsigned char *eitem = offs[RO_ENT_ITEM] >= 0 ? row + offs[RO_ENT_ITEM] : NULL;
    for (unsigned i = 0; i < n_ent; i++) {
        const unsigned char *e = ents + (long)LEAN_ENTITY_BYTES * i;
        unsigned char *o = out + 4L * RE_F * i;
        double ex = rd_d(e + 20), ey = rd_d(e + 28);
        float dx = (float)(ex - px), dy = (float)(ey - py);
        float dist = hypotf(dx, dy);
        float v[RE_F];
        memset(v, 0, sizeof v);
        int kind = e[84] < 6 ? e[84] : 6, fl = e[85];
        float size = rd_f(e + 52), cdmg = rd_f(e + 72);
        int32_t aframe = rd_i(e + 76), age = rd_i(e + 80);
        v[0] = dx / 200.0f;
        v[1] = dy / 200.0f;
        v[2] = dist / 300.0f;
        v[3] = (float)(rd_d(e + 36) / 10);
        v[4] = (float)(rd_d(e + 44) / 10);
        v[5] = (float)((ex - tlx) / w);
        v[6] = (float)((ey - tly) / h);
        v[7] = size / 20.0f;
        v[8] = rd_f(e + 56);
        v[9] = rd_f(e + 60);
        v[10] = (float)(rd_i(e + 64) / 4.0);
        v[11] = (float)(rd_i(e + 68) / 8.0);
        v[12] = cdmg > 0 ? 1.0f : 0.0f;
        v[13] = (isnan(cdmg) || cdmg < 4.0f ? cdmg : 4.0f) / 2.0f;   // np.minimum keeps a NaN
        v[14] = (float)((aframe < 120 ? aframe : 120) / 30.0);
        v[15] = (float)((age < 360 ? age : 360) / 90.0);
        v[16 + kind] = 1.0f;
        v[23] = (float)(fl & 1);
        v[24] = (float)((fl >> 1) & 1);
        v[25] = (float)((fl >> 2) & 1);
        v[26] = (float)((fl >> 3) & 1);
        v[27] = rd_f(e + 92);
        v[28] = rd_f(e + 96) / 30.0f;
        v[29] = rd_f(e + 100) / 10.0f;
        v[30] = rd_f(e + 104);
        v[31] = (float)(e[86] / 8.0);
        v[32] = (float)e[87];
        {   // version 5: flag columns, laser columns
            uint64_t word;
            memcpy(&word, e + 108, 8);
            if (word) {
                const uint64_t *m = kind == 2 ? RE_PROJ_MASKS : RE_TEAR_MASKS;
                for (int j = 0; j < RE_FLAG_F; j++) v[RE_F0 + j] = (word & m[j]) ? 1.0f : 0.0f;
            }
            if (e[84] == 3) {
                float *l = v + RE_F0 + RE_FLAG_F;
                l[0] = rd_f(e + 116);
                l[1] = rd_f(e + 120);
                l[2] = rd_f(e + 124) / 300.0f;
                l[3] = (float)(rd_f(e + 128) - px) / 200.0f;
                l[4] = (float)(rd_f(e + 132) - py) / 200.0f;
                l[5] = (float)rd_i(e + 140);
                l[6] = rd_f(e + 136) / 100.0f;
            }
        }
        memcpy(o, v, sizeof v);
        int32_t type = rd_i(e + 8);
        int16_t idv[3] = {(int16_t)(type < 0 ? 0 : type > 1023 ? 1023 : type), (int16_t)floor_mod(rd_i(e + 12), 1024),
                          (int16_t)floor_mod(rd_i(e + 16), 256)};
        memcpy(ids + 6L * i, idv, sizeof idv);
        if (eitem) {   // tok_obs.encode_row's ent_item
            int16_t it[2] = {0, 0};
            if (type == 5) {
                int32_t variant = rd_i(e + 12), sub = rd_i(e + 16);
                if (variant == 100)
                    it[0] = (int16_t)(items == 2 ? RE_ITEM_UNKNOWN : sub < 0 ? 0 : sub > 1023 ? 1023 : sub);
                else if (variant == 350) {
                    int32_t tr = sub & 0x7fff;   // without Repentance's golden-trinket flag (32768)
                    it[1] = (int16_t)(tr > 255 ? 255 : tr);
                }
            }
            memcpy(eitem + 4L * i, it, sizeof it);
        }
    }
    // doors
    int k = n_doors < RE_DOOR_CAP ? n_doors : RE_DOOR_CAP;
    if (k) {
        unsigned char *d = row + offs[RO_DOORS];
        float pxf = (float)px, pyf = (float)py;
        for (int i = 0; i < k; i++) {
            const unsigned char *dr = pl + doors_at + 16L * i;
            float v[RE_DOOR_F] = {(rd_f(dr + 4) - pxf) / 200.0f, (rd_f(dr + 8) - pyf) / 200.0f, (float)dr[1], (float)dr[2],
                                  (float)(rd_i(dr + 12) / 30.0), (float)(dr[3] & 1), (float)((dr[3] >> 1) & 1)};
            memcpy(d + 4L * RE_DOOR_F * i, v, sizeof v);
        }
    }
    {
        int32_t kk = k;
        memcpy(row + offs[RO_N_DOORS], &kk, 4);
    }
    // the floor's exits as entity tokens
    int n = (int)n_ent;
    int n_exits = (int)ctx[RC_N_EXITS];
    for (int i = 0; i < n_exits && i < 8; i++) {
        if (n >= RE_CAP) break;
        double ex = ctx[RC_EXITS + 2 * i], ey = ctx[RC_EXITS + 2 * i + 1];
        unsigned char *o = out + 4L * RE_F * n;
        float v[RE_F];
        memset(v, 0, sizeof v);
        v[0] = (float)((ex - px) / 200);
        v[1] = (float)((ey - py) / 200);
        v[2] = (float)(hypot(ex - px, ey - py) / 300);
        v[5] = (float)((ex - tlx) / w);
        v[6] = (float)((ey - tly) / h);
        v[7] = 1.0f;
        v[16] = 1.0f;
        memcpy(o, v, sizeof v);
        int16_t idv[3] = {1023, 0, 0};
        memcpy(ids + 6L * n, idv, sizeof idv);
        if (eitem) memset(eitem + 4L * n, 0, 4);
        n++;
    }
    {
        int32_t nn = n;
        memcpy(row + offs[RO_N_ENT], &nn, 4);
    }
    memcpy(row + offs[RO_GRID], grid, RE_GC * RE_GH * RE_GW);
    if (floor_ep) memcpy(row + offs[RO_MAP], map, RE_MAP);
    else memset(row + offs[RO_MAP], 0, RE_MAP);
    {
        long col = (long)nearbyint((px - ctx[RC_ORIGIN_X]) / 40.0), r = (long)nearbyint((py - ctx[RC_ORIGIN_Y]) / 40.0);
        col = col < 0 ? 0 : col > RE_GW - 1 ? RE_GW - 1 : col;
        r = r < 0 ? 0 : r > RE_GH - 1 ? RE_GH - 1 : r;
        unsigned char *patch = row + offs[RO_PATCH];
        const int ph = RE_GH + RE_PATCH - 1, pw = RE_GW + RE_PATCH - 1;
        for (int c = 0; c < RE_GC; c++)
            for (int y = 0; y < RE_PATCH; y++)
                memcpy(patch + (c * RE_PATCH + y) * RE_PATCH, padded + ((long)c * ph + r + y) * pw + col, RE_PATCH);
    }
    // rewards and the episode's end (the same room as the record before)
    int events = 0;
    float hurt = (float)(damage_taken - ctx[RC_PREV_DMG]);
    float dmg = (float)py_min(py_max(0.0, ctx[RC_PREV_HP] - monsters_hp) / ctx[RC_HP0], 1.0);
    if (clear && !(ctx[RC_WAS_CLEAR] != 0)) events |= 1 | (room_type == 5 ? 4 : 0);
    wr_f(row + offs[RO_HURT], hurt);
    wr_f(row + offs[RO_DAMAGE], dmg);
    row[offs[RO_EVENTS]] = (unsigned char)events;
    ctx[RC_ROOM] = room_idx;
    ctx[RC_WAS_CLEAR] = clear ? 1 : 0;
    ctx[RC_PREV_DMG] = damage_taken;
    ctx[RC_PREV_HP] = monsters_hp;
    int dead = pv[33] != 0;
    int done;
    if (floor_ep) {
        if (events || dmg > 0) ctx[RC_PROGRESS] = (double)t;   // a new or cleared room, or a monster hurt
        int stalled = stall > 0 && (double)t - ctx[RC_PROGRESS] >= stall;
        done = dead ? 2 : (!run_ep && (double)stage != ctx[RC_STAGE0]) ? 1 : ((double)t >= limit || stalled) ? 3 : 0;
    } else {
        done = dead ? 2 : clear ? 1 : (double)t >= limit ? 3 : 0;
    }
    int32_t tt = (int32_t)t;
    memcpy(row + offs[RO_T], &tt, 4);
    row[offs[RO_DONE]] = (unsigned char)done;
    int64_t episode = (int64_t)ctx[RC_EPISODE], seed = (int64_t)ctx[RC_SEED];
    int32_t group = (int32_t)ctx[RC_GROUP];
    memcpy(row + offs[RO_EPISODE], &episode, 8);
    memcpy(row + offs[RO_SEED], &seed, 8);
    memcpy(row + offs[RO_GROUP], &group, 4);
    row[offs[RO_FIRST]] = (unsigned char)(ctx[RC_FIRST] != 0);
    ctx[RC_HURT] = hurt > 0;
    return done;
}

static const char *const nullgl_names[] = {
    "glDrawArrays", "glDrawArraysInstanced", "glDrawElements", "glDrawElementsInstanced",
    "glClear", "glTexSubImage2D", "glBufferSubData", "glCopyTexImage2D", "glCopyTexSubImage2D",
    "glBlitFramebuffer", "glFlush", "glFinish", "glXSwapBuffers",
};
static int nullgl_default;          // ABP_NULLGL=1
static char (*null_extra)[64];      // ABP_NULLGL_LIST=file: "name" (void no-op), "name value"
static size_t null_extra_n;         //   (returns the constant, e.g. glCheckFramebufferStatus 0x8CD5)
                                    //   or "name *value" (writes it to the 3rd argument)
static void **null_extra_target;
static int nullgl_count;
static unsigned char *const_stub_mem;
static size_t const_stub_used;

static void abp_gl_noop(void) {
}

// GLFW (statically linked) swaps with a direct PLT call to libGL's glXSwapBuffers
// (_glfwPlatformSwapBuffers, VA 0x8fb9f0), bypassing the libepoxy slot, so the swap is
// interposed here. With ABP_NULLGL=1 no buffer is presented; swaps are only counted.
typedef void (*glxswap_fn)(void *, unsigned long);
static glxswap_fn real_glxswap;

void glXSwapBuffers(void *dpy, unsigned long drawable) {
    if (nullgl_default || is_child) {
        swaps_skipped++;
        return;
    }
    if (grab_fd >= 0) grab_frame(dpy, drawable);   // ABP_GRAB_ON (footage; off by default)
    if (!real_glxswap) real_glxswap = (glxswap_fn)dlsym(RTLD_NEXT, "glXSwapBuffers");
    real_glxswap(dpy, drawable);
}

// X11 round trips made through GLFW every frame (window size / position queries, event
// polling) are counted. With ABP_X11CACHE=1, once the main loop runs, window-attribute and
// coordinate-translation replies are answered from the first real reply for the same
// arguments: the window is never moved or resized in these runs, so the answers are the
// same, only the server round trip is saved. XPending is counted, never changed.
#define XWA_SIZE 136  // sizeof(XWindowAttributes) on LP64 (no X11 headers on the host)
typedef int (*xgwa_fn)(void *, unsigned long, void *);
typedef int (*xtc_fn)(void *, unsigned long, unsigned long, int, int, int *, int *, unsigned long *);
typedef int (*xpending_fn)(void *);
static xgwa_fn real_xgwa;
static xtc_fn real_xtc;
static xpending_fn real_xpending;
static struct { unsigned long w; int ret; unsigned char attrs[XWA_SIZE]; } gwa_cache[4];
static int gwa_cache_n;
static struct { unsigned long src, dst; int x, y, ret, dx, dy; unsigned long child; } tc_cache[8];
static int tc_cache_n;

int XGetWindowAttributes(void *dpy, unsigned long w, void *attrs) {
    if (!real_xgwa) real_xgwa = (xgwa_fn)dlsym(RTLD_NEXT, "XGetWindowAttributes");
    x_gwa++;
    if (x11_cache && ticks > 0) {
        for (int i = 0; i < gwa_cache_n; i++) {
            if (gwa_cache[i].w == w) { memcpy(attrs, gwa_cache[i].attrs, XWA_SIZE); x_cached++; return gwa_cache[i].ret; }
        }
        if (is_child) return 0;   // a clone has no X connection: an uncached window reads as gone
        int ret = real_xgwa(dpy, w, attrs);
        if (gwa_cache_n < 4) {
            gwa_cache[gwa_cache_n].w = w; gwa_cache[gwa_cache_n].ret = ret;
            memcpy(gwa_cache[gwa_cache_n].attrs, attrs, XWA_SIZE); gwa_cache_n++;
        }
        return ret;
    }
    if (is_child) return 0;
    return real_xgwa(dpy, w, attrs);
}

int XTranslateCoordinates(void *dpy, unsigned long src, unsigned long dst, int x, int y,
                          int *dx, int *dy, unsigned long *child) {
    if (!real_xtc) real_xtc = (xtc_fn)dlsym(RTLD_NEXT, "XTranslateCoordinates");
    x_tc++;
    if (x11_cache && ticks > 0) {
        for (int i = 0; i < tc_cache_n; i++) {
            if (tc_cache[i].src == src && tc_cache[i].dst == dst && tc_cache[i].x == x && tc_cache[i].y == y) {
                *dx = tc_cache[i].dx; *dy = tc_cache[i].dy; *child = tc_cache[i].child; x_cached++;
                return tc_cache[i].ret;
            }
        }
        if (is_child) return 0;
        int ret = real_xtc(dpy, src, dst, x, y, dx, dy, child);
        if (tc_cache_n < 8) {
            tc_cache[tc_cache_n].src = src; tc_cache[tc_cache_n].dst = dst; tc_cache[tc_cache_n].x = x;
            tc_cache[tc_cache_n].y = y; tc_cache[tc_cache_n].ret = ret; tc_cache[tc_cache_n].dx = *dx;
            tc_cache[tc_cache_n].dy = *dy; tc_cache[tc_cache_n].child = *child; tc_cache_n++;
        }
        return ret;
    }
    if (is_child) return 0;
    return real_xtc(dpy, src, dst, x, y, dx, dy, child);
}

int XPending(void *dpy) {
    if (!real_xpending) real_xpending = (xpending_fn)dlsym(RTLD_NEXT, "XPending");
    x_pending++;
    if (is_child) return 0;   // a clone has no X connection: no events
    return real_xpending(dpy);
}

int XFlush(void *dpy) {
    static int (*real)(void *);
    if (is_child) return 1;
    if (!real) real = (int (*)(void *))dlsym(RTLD_NEXT, "XFlush");
    return real(dpy);
}

int XSync(void *dpy, int discard) {
    static int (*real)(void *, int);
    if (is_child) return 1;
    if (!real) real = (int (*)(void *, int))dlsym(RTLD_NEXT, "XSync");
    return real(dpy, discard);
}

// "name render" list entries: a thunk that returns at once while IsaacRender runs (gl_bank != 0)
// and otherwise jumps to the driver entry, so logic-path users (texel read-back) are untouched.
static void *render_only_thunk(void *real) {
    unsigned char *mem = mmap(NULL, 64, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return NULL;
    uint64_t bank = (uint64_t)(uintptr_t)&gl_bank, tgt = (uint64_t)(uintptr_t)real;
    mem[0] = 0x48; mem[1] = 0xB8; memcpy(mem + 2, &bank, 8);      // movabs rax, &gl_bank
    mem[10] = 0x48; mem[11] = 0x83; mem[12] = 0x38; mem[13] = 0x00; // cmp qword [rax], 0
    mem[14] = 0x74; mem[15] = 0x01;                               // je +1 (outside render)
    mem[16] = 0xC3;                                               // ret (inside render)
    mem[17] = 0x48; mem[18] = 0xB8; memcpy(mem + 19, &tgt, 8);    // movabs rax, real
    mem[27] = 0xFF; mem[28] = 0xE0;                               // jmp rax
    return mem;
}

// ABP_GLCOUNT=path + ABP_GLCOUNT_LIST=file (GL names without the epoxy_ prefix): each listed
// libepoxy dispatch pointer is pointed at a thunk that counts the call in one of two banks
// (outside / inside IsaacRender) and jumps to the driver entry from glXGetProcAddressARB, or to
// the no-op for nulled names. Taking the driver entry directly means epoxy never re-resolves
// (and overwrites) the slot. Counts are rewritten with every stats line; ABP_GLCOUNT_SNAP_S=s
// keeps one copy (path.snap) from the first stats line at or after s real seconds.
typedef void *(*getproc_fn)(const unsigned char *);
static const char *glcount_path;
static double glcount_snap_s = -1;
static char (*gl_names)[64];
static size_t gl_n;
static uint64_t *gl_counts;

static size_t load_names(const char *path, char (**out)[64]) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return 0;
    }
    size_t cap = 256, n = 0;
    char (*names)[64] = malloc(cap * sizeof *names);
    char line[128];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, " \t\r\n")] = 0;
        if (!line[0] || line[0] == '#') continue;
        if (n == cap) names = realloc(names, (cap *= 2) * sizeof *names);
        snprintf(names[n++], 64, "%s", line);
    }
    fclose(f);
    *out = names;
    return n;
}

// `mov eax, imm32; ret` for GL entries whose (int/enum) result the game reads, or, for
// out3 != 0, `mov dword [rdx], imm32; ret`: glGet*iv(target, pname, params) answered with a
// constant written to the third argument.
static void *const_stub(uint32_t value, int out3) {
    if (!const_stub_mem) {
        const_stub_mem = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (const_stub_mem == MAP_FAILED) { const_stub_mem = NULL; return NULL; }
    }
    if (const_stub_used + 8 > 4096) return NULL;
    unsigned char *p = const_stub_mem + const_stub_used;
    const_stub_used += 8;
    if (out3) {
        p[0] = 0xC7; p[1] = 0x02; memcpy(p + 2, &value, 4); p[6] = 0xC3;
    } else {
        p[0] = 0xB8; memcpy(p + 1, &value, 4); p[5] = 0xC3;
    }
    return p;
}

// Parses ABP_NULLGL_LIST entries ("name" or "name value") into names and targets.
static void load_null_extra(const char *path) {
    char (*lines)[64] = NULL;
    size_t n = load_names(path, &lines);  // load_names cut each line at the first blank
    FILE *f = fopen(path, "r");
    null_extra = malloc((n ? n : 1) * sizeof *null_extra);
    null_extra_target = calloc(n ? n : 1, sizeof *null_extra_target);
    char line[128];
    size_t i = 0;
    while (f && i < n && fgets(line, sizeof line, f)) {
        char name[64] = "", value[32] = "";
        if (sscanf(line, "%63s %31s", name, value) < 1 || name[0] == '#') continue;
        snprintf(null_extra[i], 64, "%s", name);
        if (!value[0]) {
            null_extra_target[i] = (void *)&abp_gl_noop;
        } else if (strcmp(value, "render") == 0) {
            getproc_fn getproc = (getproc_fn)dlsym(RTLD_DEFAULT, "glXGetProcAddressARB");
            void *real = getproc ? getproc((const unsigned char *)name) : NULL;
            null_extra_target[i] = real ? render_only_thunk(real) : NULL;
        } else if (value[0] == '*') {
            null_extra_target[i] = const_stub((uint32_t)strtoul(value + 1, NULL, 0), 1);
        } else {
            null_extra_target[i] = const_stub((uint32_t)strtoul(value, NULL, 0), 0);
        }
        i++;
    }
    if (f) fclose(f);
    null_extra_n = i;
    free(lines);
}

// Target replacing a GL entry, or NULL if the entry stays real.
static void *null_target(const char *name) {
    for (size_t i = 0; i < null_extra_n; i++) {
        if (strcmp(name, null_extra[i]) == 0) return null_extra_target[i];
    }
    if (nullgl_default) {
        for (size_t i = 0; i < sizeof nullgl_names / sizeof nullgl_names[0]; i++) {
            if (strcmp(name, nullgl_names[i]) == 0) return (void *)&abp_gl_noop;
        }
    }
    return NULL;
}

static void **epoxy_slot(const char *name) {
    char sym[80];
    snprintf(sym, sizeof sym, "epoxy_%s", name);
    void **slot = (void **)dlsym(RTLD_DEFAULT, sym);
    return (slot && (uintptr_t)slot >= 0xd67000UL && (uintptr_t)slot < 0xe07000UL) ? slot : NULL;
}

static void *count_thunk(unsigned char *p, uint64_t *pair, void *target) {
    uint64_t bank = (uint64_t)(uintptr_t)&gl_bank, cnt = (uint64_t)(uintptr_t)pair;
    uint64_t tgt = (uint64_t)(uintptr_t)target;
    p[0] = 0x48; p[1] = 0xB8; memcpy(p + 2, &bank, 8);            // movabs rax, &gl_bank
    p[10] = 0x48; p[11] = 0x8B; p[12] = 0x00;                     // mov rax, [rax]
    p[13] = 0x49; p[14] = 0xBB; memcpy(p + 15, &cnt, 8);          // movabs r11, pair
    p[23] = 0x4C; p[24] = 0x01; p[25] = 0xD8;                     // add rax, r11
    p[26] = 0xF0; p[27] = 0x48; p[28] = 0xFF; p[29] = 0x00;       // lock inc qword [rax]
    p[30] = 0x48; p[31] = 0xB8; memcpy(p + 32, &tgt, 8);          // movabs rax, target
    p[40] = 0xFF; p[41] = 0xE0;                                   // jmp rax (rax/r11 are scratch)
    return p;
}

static void install_gl_hooks(void) {
    const char *extra = getenv("ABP_NULLGL_LIST");
    if (extra) load_null_extra(extra);
    glcount_path = getenv("ABP_GLCOUNT");
    const char *list = getenv("ABP_GLCOUNT_LIST");
    const char *snap = getenv("ABP_GLCOUNT_SNAP_S");
    if (snap) glcount_snap_s = atof(snap);
    if (glcount_path && list && (gl_n = load_names(list, &gl_names)) > 0) {
        getproc_fn getproc = (getproc_fn)dlsym(RTLD_DEFAULT, "glXGetProcAddressARB");
        gl_counts = calloc(2 * gl_n, sizeof *gl_counts);
        unsigned char *mem = mmap(NULL, (gl_n * 48 + 4095) & ~4095UL, PROT_READ | PROT_WRITE | PROT_EXEC,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        for (size_t i = 0; mem != MAP_FAILED && getproc && i < gl_n; i++) {
            void **slot = epoxy_slot(gl_names[i]);
            void *nulled = null_target(gl_names[i]);
            void *target = nulled ? nulled : getproc((const unsigned char *)gl_names[i]);
            if (!slot || !target) continue;
            *slot = count_thunk(mem + i * 48, &gl_counts[2 * i], target);
            nullgl_count += nulled != NULL;
        }
        return;
    }
    if (nullgl_default) {
        for (size_t i = 0; i < sizeof nullgl_names / sizeof nullgl_names[0]; i++) {
            void **slot = epoxy_slot(nullgl_names[i]);
            if (slot) { *slot = (void *)&abp_gl_noop; nullgl_count++; }
            else fprintf(stderr, "ABP_TURBO nullgl: %s not found in the executable\n", nullgl_names[i]);
        }
    }
    for (size_t i = 0; i < null_extra_n; i++) {
        void **slot = epoxy_slot(null_extra[i]);
        if (slot && null_extra_target[i]) { *slot = null_extra_target[i]; nullgl_count++; }
        else fprintf(stderr, "ABP_TURBO nullgl: %s not found in the executable\n", null_extra[i]);
    }
}

static void glcount_write(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# iterations=%llu renders=%llu game_frames=%u\n", (unsigned long long)iterations,
            (unsigned long long)renders, read_u32(G_GAME, GAME_FRAMECOUNT));
    for (size_t i = 0; i < gl_n; i++) {
        if (gl_counts[2 * i] || gl_counts[2 * i + 1]) {
            fprintf(f, "%s %llu %llu\n", gl_names[i], (unsigned long long)gl_counts[2 * i],
                    (unsigned long long)gl_counts[2 * i + 1]);
        }
    }
    fclose(f);
}

static void glcount_flush(double real_s) {
    if (!glcount_path || !gl_counts) return;
    glcount_write(glcount_path);
    if (glcount_snap_s >= 0 && real_s >= glcount_snap_s) {
        char snap[4096];
        snprintf(snap, sizeof snap, "%s.snap", glcount_path);
        glcount_write(snap);
        glcount_snap_s = -1;
    }
}

static void render_gate(void) {
    iterations++;
    if (render_every > 0 && iterations % (uint64_t)render_every == 0) {
        renders++;
        gl_bank = 8;
        ((void (*)(void))ISAAC_RENDER)();
        gl_bank = 0;
    }
    if (!turbo) {
        write_stats(0);
    }
}

// Jump stubs (`movabs rax, target; jmp rax`, 16 bytes each) for redirected calls, all in one page within rel32 range
// of the executable: the first free page from 0x20000000 up. A fixed 0x20000000 was sometimes already taken (once in
// 18 launches, EXPERIMENTS.md A7), which left the render gate and the start-seed override uninstalled.
static unsigned char *stub_page;
static int stub_count;

static unsigned char *alloc_stub(void *target) {
    if (!stub_page) {
        for (uintptr_t a = 0x20000000UL; a < 0x60000000UL && !stub_page; a += 0x100000UL) {
            void *p = mmap((void *)a, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                           -1, 0);
            if (p != MAP_FAILED) {
                stub_page = p;
            }
        }
        if (!stub_page) {
            fprintf(stderr, "ABP_TURBO no free stub page in 0x20000000-0x60000000\n");
            return NULL;
        }
    } else if (mprotect(stub_page, 4096, PROT_READ | PROT_WRITE) != 0) {
        perror("ABP_TURBO mprotect stub page");
        return NULL;
    }
    if (stub_count >= 4096 / 16) {
        return NULL;
    }
    unsigned char *stub = stub_page + 16 * stub_count++;
    uint64_t t = (uint64_t)(uintptr_t)target;
    stub[0] = 0x48; stub[1] = 0xB8;             // movabs rax, imm64
    memcpy(stub + 2, &t, 8);
    stub[10] = 0xFF; stub[11] = 0xE0;           // jmp rax
    mprotect(stub_page, 4096, PROT_READ | PROT_EXEC);
    return stub;
}

// Redirect the 5-byte `call rel32` at `site` to `target` through a stub. Refuses unless the bytes are exactly a call
// of `expected`, so a different build is left untouched. Returns 1 when installed.
static int redirect_call(uintptr_t site, uintptr_t expected, void *target, const char *what) {
    unsigned char *code = (unsigned char *)site;
    int32_t rel;
    memcpy(&rel, code + 1, 4);
    if (code[0] != 0xE8 || site + 5 + (int64_t)rel != expected) {
        fprintf(stderr, "ABP_TURBO unexpected bytes at 0x%lx; %s not installed\n", site, what);
        return 0;
    }
    unsigned char *stub = alloc_stub(target);
    if (!stub) {
        return 0;
    }
    int32_t new_rel = (int32_t)((int64_t)(uintptr_t)stub - (int64_t)(site + 5));
    uintptr_t page = site & ~0xFFFUL;
    if (mprotect((void *)page, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        perror("ABP_TURBO mprotect text");
        return 0;
    }
    memcpy(code + 1, &new_rel, 4);
    mprotect((void *)page, 0x2000, PROT_READ | PROT_EXEC);
    return 1;
}

// main's `call IsaacRender` (0x4ef141) goes to render_gate.
static void patch_render_call(void) {
    render_patched = redirect_call(MAIN_CALL_RENDER, ISAAC_RENDER, (void *)&render_gate, "render gate");
}

// The three calls of LuaEngine::PreActionHookB / PreActionHookF in Manager's input queries go to input_hook_b / _f.
static void patch_input_calls(void) {
    Dl_info info;
    if (!dladdr((void *)ENTITY_GET_TYPE, &info) || (uintptr_t)info.dli_saddr != ENTITY_GET_TYPE || !info.dli_sname ||
        strcmp(info.dli_sname, "_ZNK6Entity7GetTypeEv") != 0) {
        fprintf(stderr, "ABP_TURBO native input not installed: Entity::GetType is not at 0x%lx\n", ENTITY_GET_TYPE);
        return;
    }
    input_patched += redirect_call(CALL_HOOK_PRESSED, PRE_ACTION_HOOK_B, (void *)&input_hook_b, "native input (pressed)");
    input_patched += redirect_call(CALL_HOOK_TRIGGERED, PRE_ACTION_HOOK_B, (void *)&input_hook_b,
                                   "native input (triggered)");
    input_patched += redirect_call(CALL_HOOK_VALUE, PRE_ACTION_HOOK_F, (void *)&input_hook_f, "native input (value)");
}

// Seeds::SetStartSeed(0)'s `call RandomU32` (0x863680) goes to start_seed_draw.
static void patch_start_seed_call(void) {
    start_patched = redirect_call(START_SEED_CALL, RANDOM_U32, (void *)&start_seed_draw, "start-seed override");
}

// ---- ABP_FAST (2026-10-04, frame-cost work): exact fast paths for engine functions, see the header ----
// bit 0 (1): Camera::smooth_samples in C; bit 1 (2): the decoded-PNG cache; bit 2 (4): camera check mode (both run,
// the engine's result is used, differences counted); bit 3 (8): PNG check mode (every load decodes; a cached entry is
// compared with the fresh decode). Patches are installed on first use (start-up ABP_FAST=<mask>, or
// getenv("ABP_FAST:<mask>") from game code, e.g. in one fork clone of two), and the flags are read on every call.
static volatile unsigned fast_flags;
static int cam_patched, png_patched;
static uint64_t cam_calls, cam_fallbacks, cam_checks, cam_mismatches;

#define CAM_SMOOTH_SAMPLES 0x51a870UL   // Camera::smooth_samples(RingBuffer<Vector2> const&, float) const-ish
#define VEC2_ZERO 0xe02e20UL            // KAGE::Math::Vector2::Zero
static const uintptr_t CAM_SMOOTH_CALLS[] = {0x51aae2UL, 0x51b2dcUL, 0x51b8b1UL, 0x51b8cdUL, 0x51b8e9UL, 0x51b90cUL,
                                             0x51bb6bUL};

typedef struct { float x, y; } cam_v2;
typedef cam_v2 (*cam_smooth_fn)(void *, const void *, float);

// The engine's smooth_samples (0x51a870, read from its disassembly), with the Vector2 / InverseLerp calls inlined; same
// single-precision operations in the same order (SSE scalar, no contraction):
//   n = count (+0xc); r = (int)((float)n * p) (cvttss2si); f = (float)r; acc = Vector2::Zero; sum = 0
//   for i < n: a = i <= r ? -1.0f : (float)n + 1.0f; w = ((float)i - a) / (f - a)   (KAGE::Math::InverseLerp(a, f, i))
//              sum = w + sum; v = data[(start (+0x10) + i) % capacity (+8)]; acc.x += v.x * w; acc.y += w * v.y
//   inv = 1.0f / sum; acc *= inv
// Cases handed to the engine's own code: n <= 0, (float)n * p out of int range or NaN, and a == f (InverseLerp then
// logs an assertion before dividing).
__attribute__((optimize("fp-contract=off", "no-fast-math")))
static int cam_smooth_c(const void *rb, float p, cam_v2 *out) {
    const char *r = (const char *)rb;
    int n = *(const int *)(r + 0xc);
    if (n <= 0) return 0;
    float x1 = (float)n * p;
    if (!(x1 > -2147483648.0f && x1 < 2147483648.0f)) return 0;
    int rr = (int)x1;
    float f = (float)rr;
    float hi = (float)n + 1.0f;
    if (f == -1.0f || f == hi) return 0;
    int start = *(const int *)(r + 0x10), cap = *(const int *)(r + 0x8);
    if (cap <= 0) return 0;
    const cam_v2 *data = *(const cam_v2 *const *)r;
    cam_v2 acc = *(const volatile cam_v2 *)VEC2_ZERO;
    float sum = 0.0f;
    // (start + i) % capacity without a division per sample when 0 <= start < capacity and n <= capacity (then the sum is
    // below 2 x capacity and one subtraction is the remainder)
    int wrap = start >= 0 && start < cap && n <= cap;
    for (int i = 0; i < n; i++) {
        float a = rr >= i ? -1.0f : hi;
        float w = ((float)i - a) / (f - a);
        sum = w + sum;
        int idx = start + i;
        if (wrap) {
            if (idx >= cap) idx -= cap;
        } else {
            idx %= cap;
        }
        cam_v2 v = data[idx];
        acc.x = acc.x + v.x * w;
        acc.y = acc.y + w * v.y;
    }
    float inv = 1.0f / sum;
    acc.x = acc.x * inv;
    acc.y = acc.y * inv;
    *out = acc;
    return 1;
}

static cam_v2 cam_smooth_gate(void *cam, const void *rb, float p) {
    unsigned fl = fast_flags;
    if (!(fl & 1)) return ((cam_smooth_fn)CAM_SMOOTH_SAMPLES)(cam, rb, p);
    cam_calls++;
    cam_v2 mine;
    if (!cam_smooth_c(rb, p, &mine)) {
        cam_fallbacks++;
        return ((cam_smooth_fn)CAM_SMOOTH_SAMPLES)(cam, rb, p);
    }
    if (fl & 4) {
        cam_v2 ref = ((cam_smooth_fn)CAM_SMOOTH_SAMPLES)(cam, rb, p);
        cam_checks++;
        if (memcmp(&ref, &mine, sizeof ref) != 0) cam_mismatches++;
        return ref;
    }
    return mine;
}

// Camera::smooth_focus_samples(RingBuffer<Vector2> const&, float, Vector2 focus) (0x51a9c0, its one call 0x51b9cc in
// update_ultrasmooth): copies the ring's storage into a new[] array (capacity entries), replaces every one of the
// `count` samples (in ring order) equal to -Vector2::One (ucomiss equality of both components) by focus, runs
// smooth_samples on that copy (same capacity, count and start) and delete[]s it. Here the copy is on the stack (up to
// CAM_FOCUS_MAX entries, else the engine's code) and smooth_samples is the C version above (or the engine's when it
// hands a case over), so the samples, their order and the arithmetic are the same.
#define CAM_FOCUS_SAMPLES 0x51a9c0UL
#define CALL_CAM_FOCUS 0x51b9ccUL
#define VEC2_ONE 0xe02e18UL
#define CAM_FOCUS_MAX 1024
typedef cam_v2 (*cam_focus_fn)(void *, const void *, float, cam_v2);
static uint64_t camf_calls, camf_checks, camf_mismatches;
static int camf_patched;

static cam_v2 cam_focus_gate(void *cam, const void *rb, float p, cam_v2 focus) {
    unsigned fl = fast_flags;
    const char *r = (const char *)rb;
    int cap = *(const int *)(r + 0x8), n = *(const int *)(r + 0xc), start = *(const int *)(r + 0x10);
    if (!(fl & 1) || cap <= 0 || cap > CAM_FOCUS_MAX) return ((cam_focus_fn)CAM_FOCUS_SAMPLES)(cam, rb, p, focus);
    camf_calls++;
    cam_v2 tmp[CAM_FOCUS_MAX];
    memcpy(tmp, *(const cam_v2 *const *)r, (size_t)cap * sizeof(cam_v2));
    cam_v2 one = *(const volatile cam_v2 *)VEC2_ONE, neg;
    uint32_t bits;
    memcpy(&bits, &one.x, 4); bits ^= 0x80000000u; memcpy(&neg.x, &bits, 4);   // Vector2::operator-(): sign bits
    memcpy(&bits, &one.y, 4); bits ^= 0x80000000u; memcpy(&neg.y, &bits, 4);
    for (int i = 0; i < n; i++) {
        int idx = (start + i) % cap;
        if (tmp[idx].x == neg.x && tmp[idx].y == neg.y) tmp[idx] = focus;
    }
    struct { cam_v2 *data; int cap, count, start; } ring = {tmp, cap, n, start};
    cam_v2 mine;
    if (!cam_smooth_c(&ring, p, &mine)) mine = ((cam_smooth_fn)CAM_SMOOTH_SAMPLES)(cam, &ring, p);
    if (fl & 4) {
        cam_v2 ref = ((cam_focus_fn)CAM_FOCUS_SAMPLES)(cam, rb, p, focus);
        camf_checks++;
        if (memcmp(&ref, &mine, sizeof ref) != 0) camf_mismatches++;
        return ref;
    }
    return mine;
}

// Decoded-PNG cache. KAGE::Graphics::ImagePng::Load(char const*) (vtable slot +0xc8, called by ImageManager::LoadImage and
// ImagePng::Reload) opens the file (FileManager::OpenRead), decodes it in load_png_data (libpng; texel buffer from
// Memory::ManagerBase::AllocateTexelBuffer, alpha premultiplied, handed to ImagePlatformBase::SetTexelData, which makes
// the GL texture and frees the buffer; the image's fields +0x18 format, +0xc0/+0xc2 width/height, +0xc4/+0xc6 padded
// size, +0xc8 channels are set) and then calls ImageBase::Load(char const*) (stores the path). The engine frees images
// nobody references (ANM2::load_graphics -> AnmCache::FreeImageIfUnreferenced, Room::Init) and decodes them again at the
// next spawn of an effect using them: 4-5% of a lean clone's main thread. Cached (miss: the texel bytes handed to
// SetTexelData and those fields after a successful load, non-paletted images only), a load does the same calls without
// the file and the decode: fields restored, AllocateTexelBuffer, the cached texels copied in, SetTexelData,
// ImageBase::Load. The cache is a MAP_SHARED anonymous region made at start-up, so the root and all its fork clones
// share it (entries are written once and published with a state word; a clone killed mid-write leaves a dead slot).
#define IMAGEPNG_VT_LOAD_SLOT (0x9a74c0UL + 0x10 + 0xc8)
#define IMAGEPNG_LOAD 0x875f90UL          // KAGE::Graphics::ImagePng::Load(char const*)
#define IMAGEBASE_LOAD_PATH 0x86fcc0UL    // KAGE::Graphics::ImageBase::Load(char const*)
#define SET_TEXEL_DATA 0x881fe0UL         // KAGE::Graphics::ImagePlatformBase::SetTexelData(void**)
#define ALLOC_TEXEL_BUFFER 0x87a580UL     // KAGE::Memory::ManagerBase::AllocateTexelBuffer(unsigned)
#define MEMORY_MANAGER 0xe02a00UL         // KAGE::Memory::g_Manager (an object)
#define CALL_PNG_ALLOC 0x87595fUL         // load_png_data's AllocateTexelBuffer call
#define CALL_PNG_SET_TEXEL 0x875a77UL     // load_png_data's SetTexelData call
#define PC_SLOTS 8192
#define PC_PATH 232
typedef struct {
    volatile uint32_t state;   // 0 free, 1 being written, 2 ready, 3 dead
    uint32_t size;
    uint64_t hash;
    uint64_t off;
    unsigned char f18[4], fc0[12];
    char path[PC_PATH];
} pc_slot;
typedef struct {
    volatile uint64_t used;
    uint64_t cap;
    pc_slot slots[PC_SLOTS];
} pc_shared;
static pc_shared *pc;
static unsigned char *pc_arena;
static uint64_t png_loads, png_hits, png_misses, png_stored, png_uncached, png_checks, png_check_mismatches;
// capture of one original load (main thread only)
static void *png_cap_this;
static void *png_cap_buf;
static uint32_t png_cap_size;
static unsigned char *png_cap_copy;
static uint32_t png_cap_copy_cap, png_cap_len;
static int png_cap_done;

// The region is made at start-up only when it can be used: ABP_FAST with bit 1 (2), or ABP_PNG_CACHE_MB > 0 (e.g. for an
// instance whose clones switch the cache on at run time); the default path's address space stays as it was.
// ABP_PNG_CACHE_FILE=<path> (2026-10-06, the memory of long whole-floor runs): the region is that file (in /dev/shm),
// mapped MAP_SHARED, so every instance started with the same path (all roots of a run and all their clones) shares one
// cache instead of one per root. Each root's own cache had filled with the same images, about 3 MB a minute per root,
// up to 512 MiB each (16 roots: up to 8 GiB of shared memory that no process maps once the clones that wrote it are
// gone). The file is created (zero-filled: an empty table) by the first process and sized to its capacity; later
// processes use the capacity written in it. The texels are those of the same resource files, so a hit is the same
// whichever process decoded it (ABP_FAST bit 8 compares every hit with a fresh decode). Unset: one anonymous region per
// root, as before. The file outlives the processes: the tok sampler removes it when it closes.
static int pc_file_mapped;

static void pc_init(void) {
    const char *mb = getenv("ABP_PNG_CACHE_MB"), *fm = getenv("ABP_FAST"), *file = getenv("ABP_PNG_CACHE_FILE");
    long cap_mb = mb ? atol(mb) : ((fm && (strtoul(fm, NULL, 0) & 2)) ? 512 : 0);
    if (cap_mb <= 0) return;
    size_t total = sizeof(pc_shared) + (size_t)cap_mb * 1048576;
    void *m = MAP_FAILED;
    if (file && *file) {
        int fd = open(file, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        struct stat st;
        if (fd >= 0 && fstat(fd, &st) == 0) {
            if ((size_t)st.st_size < total && ftruncate(fd, (off_t)total) != 0) total = (size_t)st.st_size;
            if (total > sizeof(pc_shared)) {
                m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
            }
        }
        if (fd >= 0) close(fd);
        if (m != MAP_FAILED) {
            pc = (pc_shared *)m;
            uint64_t zero = 0, cap = (uint64_t)(total - sizeof(pc_shared));
            __atomic_compare_exchange_n(&pc->cap, &zero, cap, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
            if (pc->cap > cap) {   // a file sized by a process with a larger capacity: only this much is mapped here
                munmap(m, total);
                m = MAP_FAILED;
                pc = NULL;
            } else {
                pc_file_mapped = 1;
            }
        }
    }
    if (m == MAP_FAILED) {
        m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) return;
        pc = (pc_shared *)m;
        pc->cap = (uint64_t)cap_mb * 1048576;
    }
    pc_arena = (unsigned char *)m + sizeof(pc_shared);
}

static uint64_t pc_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) h = (h ^ (unsigned char)*s) * 1099511628211ULL;
    return h;
}

static pc_slot *pc_find(const char *path, uint64_t h) {
    for (uint32_t k = 0; k < PC_SLOTS; k++) {
        pc_slot *s = &pc->slots[(h + k) % PC_SLOTS];
        uint32_t st = __atomic_load_n(&s->state, __ATOMIC_ACQUIRE);
        if (st == 0) return NULL;
        if (st == 2 && s->hash == h && strcmp(s->path, path) == 0) return s;
    }
    return NULL;
}

static void pc_store(const char *path, uint64_t h, const unsigned char *img, const unsigned char *data, uint32_t size) {
    for (uint32_t k = 0; k < PC_SLOTS; k++) {
        pc_slot *s = &pc->slots[(h + k) % PC_SLOTS];
        uint32_t st = __atomic_load_n(&s->state, __ATOMIC_ACQUIRE);
        if (st != 0) {
            if (st == 2 && s->hash == h && strcmp(s->path, path) == 0) return;   // another process stored it
            continue;
        }
        uint32_t zero = 0;
        if (!__atomic_compare_exchange_n(&s->state, &zero, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        uint64_t need = ((uint64_t)size + 63) & ~63ULL;
        uint64_t off = __atomic_fetch_add(&pc->used, need, __ATOMIC_ACQ_REL);
        if (off + need > pc->cap) {
            __atomic_store_n(&s->state, 3, __ATOMIC_RELEASE);
            return;
        }
        memcpy(pc_arena + off, data, size);
        s->size = size;
        s->hash = h;
        s->off = off;
        memcpy(s->f18, img + 0x18, 4);
        memcpy(s->fc0, img + 0xc0, 12);
        snprintf(s->path, sizeof s->path, "%s", path);
        __atomic_store_n(&s->state, 2, __ATOMIC_RELEASE);
        png_stored++;
        return;
    }
}

static void *png_alloc_gate(void *mgr, unsigned size) {
    void *buf = ((void *(*)(void *, unsigned))ALLOC_TEXEL_BUFFER)(mgr, size);
    if (png_cap_this && is_main_thread) {
        png_cap_buf = buf;
        png_cap_size = size;
    }
    return buf;
}

static char png_set_texel_gate(void *img, void **data) {
    if (png_cap_this == img && data && *data == png_cap_buf && png_cap_buf) {
        if (png_cap_copy_cap < png_cap_size) {
            unsigned char *b = realloc(png_cap_copy, png_cap_size);
            if (b) {
                png_cap_copy = b;
                png_cap_copy_cap = png_cap_size;
            }
        }
        if (png_cap_copy_cap >= png_cap_size) {
            memcpy(png_cap_copy, *data, png_cap_size);   // the texels as the GL texture gets them
            png_cap_len = png_cap_size;
            png_cap_done = 1;
        }
    }
    return ((char (*)(void *, void **))SET_TEXEL_DATA)(img, data);
}

typedef uint32_t (*png_load_fn)(void *, const char *);

// The original load with the texels captured; *captured: 1 when this load's texels and fields can be cached.
static uint32_t png_load_original(void *img, const char *path, int *captured) {
    png_cap_this = img;
    png_cap_buf = NULL;
    png_cap_done = 0;
    png_cap_len = 0;
    uint32_t r = ((png_load_fn)IMAGEPNG_LOAD)(img, path);
    png_cap_this = NULL;
    // non-paletted (format 1 RGB / 2 RGBA), a successful load (ImageBase::Load's bool), one capture
    *captured = (r & 0xff) && png_cap_done && (*(uint32_t *)((char *)img + 0x18) == 1 ||
                                               *(uint32_t *)((char *)img + 0x18) == 2);
    return r;
}

static uint32_t png_load_gate(void *img, const char *path) {
    unsigned fl = fast_flags;
    if (!(fl & 2) || !pc || !is_main_thread || !path || strlen(path) >= PC_PATH) {
        if (fl & 2) png_uncached++;
        return ((png_load_fn)IMAGEPNG_LOAD)(img, path);
    }
    png_loads++;
    uint64_t h = pc_hash(path);
    pc_slot *s = pc_find(path, h);
    if (s && !(fl & 8)) {
        png_hits++;
        memcpy((char *)img + 0x18, s->f18, 4);
        memcpy((char *)img + 0xc0, s->fc0, 12);
        void *buf = ((void *(*)(void *, unsigned))ALLOC_TEXEL_BUFFER)((void *)MEMORY_MANAGER, s->size);
        if (!buf) return ((png_load_fn)IMAGEPNG_LOAD)(img, path);
        memcpy(buf, pc_arena + s->off, s->size);
        if (!((char (*)(void *, void **))SET_TEXEL_DATA)(img, &buf)) return 0;
        return ((uint32_t (*)(void *, const char *))IMAGEBASE_LOAD_PATH)(img, path);
    }
    int captured = 0;
    uint32_t r = png_load_original(img, path, &captured);
    if (s) {   // check mode: a cached entry against the fresh decode
        png_checks++;
        if (!captured || s->size != png_cap_len || memcmp(pc_arena + s->off, png_cap_copy, png_cap_len) != 0 ||
            memcmp(s->f18, (char *)img + 0x18, 4) != 0 || memcmp(s->fc0, (char *)img + 0xc0, 12) != 0) {
            png_check_mismatches++;
        }
        return r;
    }
    png_misses++;
    if (captured) pc_store(path, h, (const unsigned char *)img, png_cap_copy, png_cap_len);
    else png_uncached++;
    return r;
}

// access() memo (ABP_FAST bit 4 (16); bit 5 (32) check mode). KAGE::Filesys::File::Exists(path) is access(path, F_OK)
// on the cleaned path; FileManager::GetExpandedPath calls it (get_loose_file) for every resource it resolves (loose files
// before the archive), so each ANM2 load and image lookup of an effect spawn is a failing syscall. Only F_OK checks of
// relative paths (the game's resources, below its working directory, which nothing writes while it runs) made by the
// game's code are memoised, with their result and errno; absolute paths (the save directory) always go to the kernel.
#define AM_SLOTS 8192
static struct { uint64_t hash; char *path; int result, err; } am_table[AM_SLOTS];
static int am_count;
static uint64_t am_calls, am_hits, am_checks, am_mismatches;
static int (*real_access)(const char *, int);

int access(const char *path, int mode) {
    if (!real_access) real_access = (int (*)(const char *, int))dlsym(RTLD_NEXT, "access");
    unsigned fl = fast_flags;
    if (!(fl & 16)|| mode != 0 || path[0] == '/' || !is_main_thread || ticks == 0 ||
        !from_game(__builtin_return_address(0))) {
        return real_access(path, mode);
    }
    am_calls++;
    uint64_t h = 1469598103934665603ULL;
    for (const char *s = path; *s; s++) h = (h ^ (unsigned char)*s) * 1099511628211ULL;
    for (uint32_t k = 0; k < AM_SLOTS; k++) {
        uint32_t i = (uint32_t)((h + k) % AM_SLOTS);
        if (!am_table[i].path) {
            int r = real_access(path, mode), e = errno;
            if (am_count < AM_SLOTS / 2) {
                char *copy = strdup(path);
                if (copy) {
                    am_table[i].hash = h;
                    am_table[i].result = r;
                    am_table[i].err = e;
                    am_table[i].path = copy;
                    am_count++;
                }
            }
            errno = e;
            return r;
        }
        if (am_table[i].hash == h && strcmp(am_table[i].path, path) == 0) {
            am_hits++;
            if (fl & 32) {
                int r = real_access(path, mode), e = errno;
                am_checks++;
                if (r != am_table[i].result || (r != 0 && e != am_table[i].err)) am_mismatches++;
                errno = e;
                return r;
            }
            if (am_table[i].result != 0) errno = am_table[i].err;
            return am_table[i].result;
        }
    }
    return real_access(path, mode);
}

static int patch_vtable_slot(uintptr_t slot, uintptr_t expected, void *target) {
    if (*(volatile uintptr_t *)slot != expected) {
        fprintf(stderr, "ABP_TURBO unexpected vtable entry at 0x%lx\n", slot);
        return 0;
    }
    uintptr_t page = slot & ~0xFFFUL;
    if (mprotect((void *)page, 0x2000, PROT_READ | PROT_WRITE) != 0) return 0;
    *(volatile uintptr_t *)slot = (uintptr_t)target;
    mprotect((void *)page, 0x2000, PROT_READ);
    return 1;
}

static int check_export(uintptr_t va, const char *name) {
    Dl_info info;
    return dladdr((void *)va, &info) && (uintptr_t)info.dli_saddr == va && info.dli_sname &&
           strcmp(info.dli_sname, name) == 0;
}

// Installs the patches the mask needs (once) and sets the flags. Returns "cam=<sites> png=<0/1> flags=<mask>".
static char *fast_apply(unsigned mask) {
    static char answer[96];
    if ((mask & 1) && !cam_patched &&
        check_export(CAM_SMOOTH_SAMPLES, "_ZN6Camera14smooth_samplesERK10RingBufferIN4KAGE4Math7Vector2EEf")) {
        int n = 0;
        for (size_t i = 0; i < sizeof CAM_SMOOTH_CALLS / sizeof CAM_SMOOTH_CALLS[0]; i++) {
            n += redirect_call(CAM_SMOOTH_CALLS[i], CAM_SMOOTH_SAMPLES, (void *)&cam_smooth_gate, "camera smooth");
        }
        cam_patched = n;
        if (check_export(CAM_FOCUS_SAMPLES,
                         "_ZN6Camera20smooth_focus_samplesERK10RingBufferIN4KAGE4Math7Vector2EEfS3_")) {
            camf_patched = redirect_call(CALL_CAM_FOCUS, CAM_FOCUS_SAMPLES, (void *)&cam_focus_gate, "camera focus");
        }
    }
    if ((mask & 2) && !png_patched && pc && check_export(IMAGEPNG_LOAD, "_ZN4KAGE8Graphics8ImagePng4LoadEPKc") &&
        check_export(IMAGEBASE_LOAD_PATH, "_ZN4KAGE8Graphics9ImageBase4LoadEPKc") &&
        check_export(SET_TEXEL_DATA, "_ZN4KAGE8Graphics17ImagePlatformBase12SetTexelDataEPPv") &&
        check_export(ALLOC_TEXEL_BUFFER, "_ZN4KAGE6Memory11ManagerBase19AllocateTexelBufferEj")) {
        int a = redirect_call(CALL_PNG_ALLOC, ALLOC_TEXEL_BUFFER, (void *)&png_alloc_gate, "png alloc capture");
        int b = a && redirect_call(CALL_PNG_SET_TEXEL, SET_TEXEL_DATA, (void *)&png_set_texel_gate, "png texel capture");
        png_patched = b && patch_vtable_slot(IMAGEPNG_VT_LOAD_SLOT, IMAGEPNG_LOAD, (void *)&png_load_gate);
    }
    unsigned ok = mask;
    if (cam_patched != (int)(sizeof CAM_SMOOTH_CALLS / sizeof CAM_SMOOTH_CALLS[0])) ok &= ~5u;
    if (!png_patched) ok &= ~10u;
    fast_flags = ok;
    snprintf(answer, sizeof answer, "cam=%d png=%d flags=%u", cam_patched, png_patched, ok);
    return answer;
}

static char *fast_status(void) {
    static char st[800];
    snprintf(st, sizeof st, "flags=%u cam_patched=%d cam_calls=%llu cam_fallbacks=%llu cam_checks=%llu "
             "cam_mismatches=%llu camf_patched=%d camf_calls=%llu camf_checks=%llu camf_mismatches=%llu "
             "png_patched=%d png_loads=%llu png_hits=%llu png_misses=%llu png_stored=%llu "
             "png_uncached=%llu png_checks=%llu png_check_mismatches=%llu pc_used=%llu am_calls=%llu am_hits=%llu "
             "am_checks=%llu am_mismatches=%llu am_entries=%d pc_file=%d pc_cap=%llu",
             fast_flags, cam_patched, (unsigned long long)cam_calls, (unsigned long long)cam_fallbacks,
             (unsigned long long)cam_checks, (unsigned long long)cam_mismatches, camf_patched,
             (unsigned long long)camf_calls, (unsigned long long)camf_checks, (unsigned long long)camf_mismatches,
             png_patched,
             (unsigned long long)png_loads, (unsigned long long)png_hits, (unsigned long long)png_misses,
             (unsigned long long)png_stored, (unsigned long long)png_uncached, (unsigned long long)png_checks,
             (unsigned long long)png_check_mismatches, pc ? (unsigned long long)pc->used : 0ULL,
             (unsigned long long)am_calls, (unsigned long long)am_hits, (unsigned long long)am_checks,
             (unsigned long long)am_mismatches, am_count, pc_file_mapped, pc ? (unsigned long long)pc->cap : 0ULL);
    return st;
}

// ABP_STUB_LIST=file: lines "0xVA name-part [ret0]". The function starting at VA returns at once
// (`ret`, or `xor eax,eax; ret` with ret0). A line is applied only if dladdr() finds an exported
// symbol starting exactly at VA whose (mangled) name contains name-part; otherwise it is refused.
// Only for functions that merely prepare pixels (e.g. vertex batching) and whose callers already
// handle a zero result.
static int stubs_applied;

// Applies one stub list file to this process now (start-up: ABP_STUB_LIST; at run time: getenv("ABP_STUBS:<file>") from
// game code, e.g. in a fork clone only, whose code pages are its own copy after the write). Returns the lines applied.
static int apply_stub_file(const char *path) {
    int before = stubs_applied;
    FILE *f = path ? fopen(path, "r") : NULL;
    char line[256];
    while (f && fgets(line, sizeof line, f)) {
        unsigned long va = 0;
        char part[128] = "", mode[16] = "";
        if (line[0] == '#' || sscanf(line, "%lx %127s %15s", &va, part, mode) < 2) continue;
        Dl_info info;
        if (!dladdr((void *)va, &info) || (unsigned long)info.dli_saddr != va || !info.dli_sname ||
            !strstr(info.dli_sname, part)) {
            fprintf(stderr, "ABP_TURBO stub refused: 0x%lx %s\n", va, part);
            continue;
        }
        unsigned char code[3] = {0x31, 0xC0, 0xC3};  // xor eax, eax; ret
        int ret0 = strcmp(mode, "ret0") == 0;
        uintptr_t page = va & ~0xFFFUL;
        if (mprotect((void *)page, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) continue;
        if (ret0) memcpy((void *)va, code, 3);
        else *(unsigned char *)va = 0xC3;
        mprotect((void *)page, 0x2000, PROT_READ | PROT_EXEC);
        stubs_applied++;
    }
    if (f) fclose(f);
    return stubs_applied - before;
}

static void install_stubs(void) {
    apply_stub_file(getenv("ABP_STUB_LIST"));
}

int __libc_start_main(int (*main_fn)(int, char **, char **), int argc, char **argv,
                      void (*init)(void), void (*fini)(void), void (*rtld_fini)(void),
                      void *stack_end) {
    resolve();
    start_main_fn real = (start_main_fn)dlvsym(RTLD_NEXT, "__libc_start_main", "GLIBC_2.2.5");
    if (!real) {
        real = (start_main_fn)dlsym(RTLD_NEXT, "__libc_start_main");
    }
    if ((uintptr_t)main_fn != GAME_MAIN) {
        // Not the AB+ executable (e.g. a shell that inherited LD_PRELOAD): stay inert.
        return real(main_fn, argc, argv, init, fini, rtld_fini, stack_end);
    }
    // Runs after the DRM stub decrypted .text and before any game code.
    {
        // 2026-10-06 (diagnostic): a game process that died with SIGSEGV at start-up left an empty stdout.log; this
        // line tells a crash before this point (DRM stub, loader) from one in the patching below (the ABP_TURBO line)
        static const char mark[] = "ABP_TURBO start-up\n";
        if (write(2, mark, sizeof mark - 1) < 0) {}
    }
    real_getenv = (getenv_fn)dlsym(RTLD_NEXT, "getenv");
    void *genrand = dlsym(RTLD_DEFAULT, "_Z12init_genrandm");
    if ((uintptr_t)genrand == INIT_GENRAND) {
        game_init_genrand = (init_genrand_fn)genrand;
    } else {
        fprintf(stderr, "ABP_TURBO reseed disabled: init_genrand at %p\n", genrand);
    }
    const char *t = getenv("ABP_TURBO");
    const char *r = getenv("ABP_RENDER_EVERY");
    stats_path = getenv("ABP_STATS");
    const char *ms = getenv("ABP_STATS_MS");
    if (ms && atol(ms) > 0) {
        stats_interval_ns = (uint64_t)atol(ms) * 1000000ULL;
    }
    is_main_thread = 1;
    turbo = t && atoi(t) != 0;
    if (r) {
        render_every = atol(r);
    }
    real_start_ns = real_ns(CLOCK_MONOTONIC);
    realtime_offset_ns = (int64_t)(real_ns(CLOCK_REALTIME) - real_start_ns);
    v_mono_ns = real_start_ns;
    const char *ft = getenv("ABP_FIXED_TIME");
    if (ft) {
        fixed_time = atoll(ft);
        if (turbo) {
            v_mono_ns = (uint64_t)fixed_time * 1000000000ULL;
            realtime_offset_ns = 0;
        }
    }
    patch_render_call();
    if (game_init_genrand) {
        patch_start_seed_call();
    }
    // 2026-10-08: Game::StartDebug's PlayerManager::Init call goes to start_debug_player_init (ABP_PLAYERTYPE)
    ptype_patched = redirect_call(START_DEBUG_PM_INIT_CALL, PLAYER_MANAGER_INIT, (void *)&start_debug_player_init,
                                  "player-type override");
    patch_input_calls();
    const char *pg = getenv("ABP_PU_GATE");
    if (!pg || atoi(pg) != 0) {
        pu_patched = redirect_call(CALL_POST_UPDATE, LUA_POST_UPDATE, (void *)&pu_gate, "post-update gate");
    }
    const char *al = getenv("ABP_AL_STOPPED");
    al_stopped = al && atoi(al) != 0;
    const char *ng = getenv("ABP_NULLGL");
    nullgl_default = ng && atoi(ng) != 0;
    const char *xc = getenv("ABP_X11CACHE");
    x11_cache = xc && atoi(xc) != 0;
    install_gl_hooks();
    install_stubs();
    pc_init();
    const char *fm = getenv("ABP_FAST");
    if (fm && strtoul(fm, NULL, 0) != 0) fast_apply((unsigned)strtoul(fm, NULL, 0));
    prof_start();
    stack_dump_install();
    {
        void *warm[2];
        backtrace(warm, 2);   // loads the unwinder now, so the clone watchdog's handler does not have to
    }
    fprintf(stderr, "ABP_TURBO turbo=%d render_every=%ld render_gate=%d nullgl=%d stubs=%d start_seed=%d al_stopped=%d "
            "input_calls=%d pu_gate=%d fast=%u\n", turbo, render_every, render_patched, nullgl_count, stubs_applied,
            start_patched, al_stopped, input_patched, pu_patched, fast_flags);
    return real(main_fn, argc, argv, init, fini, rtld_fini, stack_end);
}

// ---------------------------------------------------------------------------------------------------------------
// ABP_WATCH_PID=<pid> (environment, read once when the library loads): the instance ends itself with _exit(0) within
// about two seconds of that process disappearing. The fork sampler's workers set it to their own pid: a worker that
// is terminated cannot stop its instance, which would otherwise run on without a client at full speed. The watcher
// is a thread of the root process only (threads do not survive fork()); clones end with their connection.
static void *watch_pid_main(void *arg) {
    pid_t pid = (pid_t)(long)arg;
    for (;;) {
        struct timespec ts = {2, 0};
        nanosleep(&ts, NULL);
        if (kill(pid, 0) != 0 && errno == ESRCH) _exit(0);
    }
    return NULL;
}

__attribute__((constructor)) static void watch_pid_init(void) {
    const char *v = secure_getenv("ABP_WATCH_PID");
    long pid = v ? atol(v) : 0;
    // only in the game itself: the library is preloaded into the shell commands the game runs too (cp, rm for its mod
    // folders), and those crashed about half the time with this thread running
    char exe[512];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return;
    exe[n] = 0;
    const char *base = strrchr(exe, '/');
    if (strncmp(base ? base + 1 : exe, "isaac", 5) != 0) return;
    if (pid > 1) {
        // 2026-10-09 (the start-up SIGSEGV of B16, root cause found on the HPC deployment, 116/128 starts there): this
        // constructor can run before the one that resolves the real clock functions, and the watcher's first
        // nanosleep() then goes through the interposer to a null real_nanosleep (rip = 0). Resolve first; 128/128 after.
        resolve();
        pthread_t t;
        if (pthread_create(&t, NULL, watch_pid_main, (void *)pid) == 0) pthread_detach(t);
    }
}
