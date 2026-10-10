"""Sampler of the token policy's training and evaluation (EXPERIMENTS.md C55 on), built on the fork sampler (B10).

worker process x N (no torch): a root AB+ instance (FORK_ENV) that only resets; a parked template clone of each start
  state; every episode a clone of the template (global MT reseeded, the bridge's lean mode). Each decision the worker
  writes one tok_obs.ROW into its rows of a shared array and asks the server for an action.
The rule of the environment (user decision 2026-10-03): an action takes effect one decision late. The worker steps the
  game with the answer to the record before the one it has just sent, so the game runs while the server works, and the
  policy has a reaction time of one decision (133 ms at 4 logic frames). An episode's first step is a no-op.
server (the trainer or the evaluator): TokSampler.poll() gives the workers whose records are ready and where they are,
  TokSampler.reply() writes their actions and wakes them.

Hindsight search teacher (cfg.teacher_share > 0; EXPERIMENTS.md C56): after an episode, for some of the steps that
  hurt the player the worker restores the episode shortly before (a clone of the template with the episode's reseed
  and its first actions replayed: exact, A19) and lets nine clones of that state each hold one of the nine moves until
  a few decisions after the hurt, everything else as it was. The record of that state, the action that was under way
  and which moves stayed unhurt go to the trainer through a ring of teacher records. The clones share the episode's
  random numbers: the teacher knows the future the policy does not (user decision 2026-10-03: allowed for training
  targets, not for evaluation). The latest state with a safe move is searched for: depth 2, then 4, then 8 decisions
  before the hurting step. The searches wait in a queue per worker and run while the trainer holds the workers back
  (control[1]: during the update in train_tok's synchronous mode; with --overlap only while the actor waits for the
  learner): the workers have nothing to do then and the CPU is free, so the teacher costs no sampling time. With
  teacher_share > 0 a worker also spends up to that share of its time on searches right after an episode (with
  --overlap this budget is the teacher's main source of time).
Teacher cost options (2026-10-05): teacher_slots (off: searches at once over all workers: forks and copy-on-write
  faults slow down in proportion to how many run at once, while the machine's search rate hardly grows past 4-8),
  teacher_pair (the queued hurts of one episode share one restore replay: hindsight_shared, same records and lost[];
  train_tok's --teacher-pair defaults to 1, this config field to False), teacher_skip_known (off: no clone for the move
  the episode itself held up to the hurt: same teacher records, lost[] of that move not measured); ABP_FORK_LITE
  (abp_turbo: a clone with one thread and every registered mutex free forks without the mutex protocol, same memory;
  on by default in the workers' instances, see INSTANCE_ENV_DEFAULTS); ISAAC_RL_TEACH_PROF=<dir> writes per-search
  cost records (diagnostic).
Instance defaults (2026-10-05, each proven exact in B14 / the teacher's label probes): a worker launches its root
  instance with INSTANCE_ENV_DEFAULTS (ABP_FAST=3, ISAAC_RL_PU_SKIP=1, ABP_FORK_LITE=1) unless the variable is already
  set in its environment (set it to 0 to switch that one off), and with the stub list cfg.stub_list, or when that is
  empty default_stub_list(): stub_render_h.txt next to the instances' libabp_turbo.so, else stub_render_g.txt (train_tok
  --stub-list <file> chooses another, e.g. the previous default stub_render_g.txt).
Memory (2026-10-06, a 16-worker whole-floor run on host 2 measured by role, abplus_probe_memory.py): the decoded-PNG
  caches of the roots (one 512 MiB shared region per root, filling with the same images) grew by about 1.2 GB in 2 h,
  the trainer's host copy of the teacher records by 0.8 GB, the archive's parked clones stayed at about 7.7 GB for
  ~140 entries. Now: one PNG cache file for all instances of a sampler (ABP_PNG_CACHE_FILE, removed in close();
  ISAAC_RL_PNG_SHARED=0: one per root as before), malloc_trim in every parked clone (cfg.trim_parked, trim_parked),
  train_tok keeps only the counts of the teacher records on the host with --teach-gpu 1; a root above
  cfg.root_anon_mib RssAnon can be replaced at a state boundary (recycle_root; off: the roots grew ~3 MiB in 2 h).
Start-state time limit (2026-10-05): the root instance's reset runs in a helper thread (StartBuilder); a build that has
  not finished after cfg.build_timeout s (the first build of a launch: launch, connect, reset) or cfg.reset_timeout s (a
  later one; 2026-10-06) counts as an error (and a stall), the instance is killed (Instance.kill: SIGKILL to its process
  group and every game process of its name, checked gone), relaunched, and (2026-10-06) the same seed is built once
  more, then the next one. Before the kill stall_evidence saves what the game was doing. The cause of the hangs seen
  in training (2026-10-06): the bridge port was taken when the game bound it (most often by a parked clone's connection
  whose ephemeral local port it was), the bridge disabled itself and the game ran on without a client; the instances
  now get ISAAC_RL_PORT_FILE and the bridge binds a free port then (abp_bridge.lua try_bind, counted in port_moves).
  A worker ending normally kills a root whose build is still in flight; SIGTERM (TokSampler.close's terminate) ends the worker through the same path;
  a worker killed outright (SIGKILL, OOM) takes its root with it: the root is launched with PR_SET_PDEATHSIG = SIGKILL
  (abplus.launch_abplus die_with_parent; clones end with their connections). The ABP_WATCH_PID watcher (set by
  default, see WATCH_NOTE) only sees a worker gone once it is reaped: TokSampler reaps a worker when its pipe closes.
  Fault injection for tests: ISAAC_RL_FAULT_HANG=<worker>:<build>:<stop|spin|crash>[,...] makes that worker's root stop
  answering at the start of its <build>-th build (1 = the first launch): stop = SIGSTOP, spin = a Lua busy loop
  (needs a connected instance; a first build falls back to stop), crash = SIGSEGV (2026-10-06).
  A second cause of first-build stalls (2026-10-06): the game process sometimes dies with SIGSEGV during start-up,
  before main (apport.log; host 2 10-05 22:26 and 10-06 12:47, each at a run's start); the client's connect loop now
  gives up as soon as the process has ended (AbplusTrainingEnv.process_exit) instead of waiting for the time limit.

Training: worker i builds start states from seeds seed + i + N k of its group (cfg.assign), the start half hearts drawn
  from cfg.start_hp, and plays cfg.episodes_per_state episodes from each (more while the next state is not ready).
Evaluation (cfg.eval_seeds): worker i plays its seeds once each, the clone reseeded with the seed itself, full
  health, then stops; everything else is the same code.
"""
import dataclasses
import json
import multiprocessing as mp
import os
import select
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import connection, shared_memory
from pathlib import Path

import numpy as np

from .abplus import ABP_HOME, FORK_ENV, action_code
from .abplus_goexplore import GxConfig, Instance
from .abplus_lean import LeanDecoder, read_lean, read_lean_raw
from .env import BridgeError
from .tok_obs import ROW, EpisodeState, FastRow, encode_row, fast_row_function

STATS = ('decisions', 'frames', 'episodes', 'forks', 'states', 'errors', 'wins', 'deaths', 'timeouts', 'empty',
         'step_s', 'wait_s', 'fork_s', 'encode_s', 'finished', 'extra_episodes', 'teach', 'teach_s', 'searches',
         'avoidable', 'hurts', 'snaps', 'archive', 'archive_starts',   # these three: tok_floor
         'overrun_s', 'queue',   # searches past the end of the trainer's hold (s); hurts queued for a search now
         'state_s', 'close_s',   # seconds taking up a new start state (and waiting for it); closing episodes
         'dropped',   # queued hurts dropped without a search (queue full, or their start state replaced)
         'stalls',    # start-state builds given up after cfg.build_timeout (also counted in errors)
         'retries',   # 2026-10-06: builds of a seed tried again on the relaunched instance after a failure or stall
         'port_moves',   # 2026-10-06: root launches whose bridge port was taken (the bridge bound a free one instead)
         'recycles',  # 2026-10-06: root instances replaced at a state boundary (cfg.root_anon_mib)
         'trims',     # 2026-10-06: parked clones whose free heap was given back (cfg.trim_parked)
         # 2026-10-07 (counterfactual branches, tok_branch.py; tok_floor only, zero unless cfg.branch_share > 0):
         'bp_item', 'bp_door', 'bp_item_left', 'bp_exit',   # branch points detected, by kind
         'bp_item_far',   # collectibles gained without a pedestal of that id within reach at the record before
         'bp_dropped',    # points dropped from the queue (lowest priority) or at a recovery
         'b_points', 'b_branches', 'b_dec', 'b_frames',   # points processed, branches run, their decisions and frames
         'b_s', 'b_cpu', 'b_prep_s',   # wall s of the points (all), worker-thread CPU s, wall s of restores + prefixes
         'b_invalid', 'b_walk_fail', 'b_skip_took', 'b_errors', 'choices', 'b_queue', 'b_replicates',
         'b_ped_seen',    # room changes with a pedestal still holding an item at the record before (all, any kind)
         'b_game_cpu',    # CPU s of the branch clones and restore clones (their main threads, /proc schedstat at close)
         # 2026-10-07 (Phase B2): item_left pedestals passed over (shop item, no path), twins run and their decisions
         # with equal states but other actions (must stay 0 with common random numbers), pairs run (all, unconditional
         # replicate, conditional), branch states parked in the archive
         'b_shop_skip', 'b_unreach_skip', 'b_twins', 'b_twin_act_diff', 'b_pairs', 'b_noise_pairs', 'b_cond_pairs',
         'b_archived',
         # 2026-10-08 (build lab, tok_lab.py; tok_floor only, zero unless cfg.lab_share > 0): jobs taken, panel states
         # run (lab episodes), their decisions and logic frames, wall / worker-thread CPU / game-clone CPU seconds,
         # errors, result records written, panel states parked now, seconds building them, panel (re)builds
         'lab_jobs', 'lab_states', 'lab_dec', 'lab_frames', 'lab_s', 'lab_cpu', 'lab_game_cpu', 'lab_errors',
         'lab_results', 'lab_panel', 'lab_panel_s', 'lab_panel_builds',
         # 2026-10-08: start builds transplanted (cfg.start_build_prob), stat augmentations applied (cfg.stat_aug_prob)
         'start_builds', 'stat_augs',
         # 2026-10-08 (cfg.characters): floor starts built as a character other than Isaac
         'char_starts',
         # 2026-10-10 (cfg.archive_plr): written by the TRAINER, read by the worker: the number of the worker's last
         # finished episode and its learning-potential score |discounted return - value at its first record|
         'plr_episode', 'plr_score',
         # 2026-10-09 (death teacher): fatal hurts searched, of them with a safe move found, their search seconds
         'death_searches', 'death_avoidable', 'death_s',
         # 2026-10-10 (teacher v2, tok_teacher2.py; tok_floor only, zero unless cfg.teacher_v2): points searched (and of
         # them random / fatal ones), depths tried, points with an improving branch, sums of best - taken at the first
         # and at the last depth, sum of the replicate spreads and replicates run, branches run, their decisions and
         # logic frames, imitation records written, wall / worker-thread CPU / lane-wait seconds, errors, points
         # queued now, dropped, without a possible depth
         'v2_points', 'v2_random', 'v2_fatal', 'v2_depths', 'v2_improving', 'v2_gain0', 'v2_gain', 'v2_spread',
         'v2_reps', 'v2_branches', 'v2_dec', 'v2_frames', 'v2_records', 'v2_s', 'v2_cpu', 'v2_wait_s', 'v2_errors',
         'v2_queue', 'v2_dropped', 'v2_nodepth')
TEACH_RING = 128     # teacher records per worker the trainer may fall behind by
TEACH_META = 16      # per record: the action under way (3), nine danger bits, the move taken, depth, safe moves, 0
ST = {k: i for i, k in enumerate(STATS)}
ACTION_F = 5    # move, shoot, bomb, item, pill (2026-10-06: the last two only with cfg.items)
FORK_MANY = os.environ.get('ISAAC_RL_FORK_MANY', '1') != '0'   # hindsight: the bridge's fork_many when offered
# WATCH_NOTE. ISAAC_RL_WATCH_PID=0: the workers do not set ABP_WATCH_PID (default 1: they do). Found 2026-10-04 on
# host 2: with ABP_WATCH_PID in the environment, libabp_turbo's watcher thread also starts in the programs the game
# runs at launch (sh -c "cp ..." / "rm ..." for its mod folders, which inherit LD_PRELOAD and the variable), and some
# of them crash with SIGSEGV (`LD_PRELOAD=libabp_turbo.so ABP_WATCH_PID=<pid> sh -c "cp a b"`: 2 of 4 tries; 0 of 4
# without the variable). Host 2 hands every crash to apport, which holds the crashed process (and the instance waiting
# for it) for seconds, one crash at a time: 10 of 16 workers had not produced a decision after minutes. Without the
# variable a terminated worker leaves its instance running (no watcher).
# multiprocessing.Connection's framing of a one-byte message (big-endian length, then the byte)
_MSG0, _MSG1, _REPLY = b'\x00\x00\x00\x010', b'\x00\x00\x00\x011', b'\x00\x00\x00\x01a'


@dataclass
class TokSamplerConfig:
    workers: int = 16
    specs: list = field(default_factory=list)     # task specs (goexplore_abplus.load_spec), one per group
    assign: list = field(default_factory=list)    # per worker: index into specs
    frames_per_decision: int = 4
    episodes_per_state: int = 16
    episode_seconds: float = 0.0                  # time limit (0: the group's own)
    start_hp: tuple = (6, 6)                      # half hearts at an episode's start: drawn uniformly per state
    bombs: int = 1
    seed: int = 1000
    eval_seeds: list = field(default_factory=list)   # per worker: its seeds (evaluation); empty: training
    name: str = 'tk'
    port: int = 28300
    nice: int = 10
    bridge_lua: str = ''
    preload: str = ''
    stub_list: str = ''
    clone_alarm: int = 900
    teacher: bool = False                         # hindsight search teacher, run during the trainer's updates
    teacher_share: float = 0.0                    # and, beyond that, up to this share of a worker's sampling time
    teacher_queue: int = 24                       # hurts waiting to be searched per worker (the newest are kept)
    teacher_depths: tuple = (2, 4, 8)             # decisions before the hurting step, tried in this order
    teacher_margin: int = 4                       # decisions after the hurting step that must stay unhurt too
    teacher_per_episode: int = 2                  # hurts searched per episode at most
    teacher_slots: int = 0                        # searches running at once over all workers at most (0: no limit)
    teacher_skip_known: bool = False              # no clone for a move whose hurt is known (known_hurt_move)
    teacher_pair: bool = False                    # the queued hurts of one episode share a restore (hindsight_shared)
    # 2026-10-09 (death teacher, SCALING_THESIS.md S5; off unless teacher_death_depths is set): the fatal hurt of an
    # episode is always searched (never dropped by teacher_per_episode), with these depths and this margin instead
    teacher_death_depths: tuple = ()              # e.g. (2, 4, 8, 16, 32): up to 4.3 s before the fatal step
    teacher_death_margin: int = 0                 # 0: teacher_margin
    # 2026-10-10 (floor / run modes): 1 = the searches run in a thread of the worker, so a search never holds the
    # worker's episode (with a fast learner the searches no longer fit into the update and stalled the collection:
    # C77 on the HPC, collect 1.4 -> 12 s); teacher_share then bounds the thread's search time outside the updates.
    # 0 = as before (searches during the updates and, within teacher_share, in the worker's own loop)
    teacher_thread: int = 0
    teacher_fork: int = 0                        # 2026-10-10: with teacher_thread, each hurt search in a forked child
    # 2026-10-10 (teacher v2, tok_teacher2.py; floor / run modes, needs teacher_thread; off = nothing changes): whole
    # branches from decision points (hurts, deaths, random records) with movement intents x the actor's per-step answers
    # (through a lane of the worker), the best one's every step an imitation record when it beats the episode's own
    # continuation by more than the margin and its replicate's spread
    teacher_v2: bool = False
    teacher_v2_hold: tuple = (2, 4, 8)            # decisions a held-move intent lasts (one drawn per move and depth)
    teacher_v2_sticky: float = 4.0                # mean decisions a sticky-policy intent keeps a sampled move (0: none)
    teacher_v2_sticky_n: int = 1                  # sticky-policy intents per depth
    teacher_v2_seconds: float = 20.0              # a branch's game-time cap (then the value bootstrap)
    teacher_v2_margin: float = 0.3                # best - taken must exceed this (and the replicate's spread)
    teacher_v2_cap: float = 2.0                   # an imitation record's weight: min(best - taken, this)
    teacher_v2_random: float = 0.0                # random decision points per game hour of play (0: hurts / deaths only)
    teacher_v2_queue: int = 8                     # points waiting per worker (each holds its room's parked entry)
    teacher_v2_depths: tuple = ()                 # hurts' depths (empty: teacher_depths); deaths: teacher_death_depths
    # 2026-10-11 (train_tok --memory gru): the policy's GRU state size; with teacher_v2 the lanes and the V2 ring then
    # carry states too (tok_teacher2.attach_v2_mem). 0: nothing changes
    memory_dim: int = 0
    mode: str = 'room'                          # 'room': one room per episode; 'floor' / 'run': tok_floor.py
    items: bool = False                           # 2026-10-06: the bridge's lean_items, ROW's item fields, 5 actions
    run_seconds: float = 1800.0                   # run: an episode's time limit (game seconds)
    floor_seconds: float = 480.0                  # floor: an episode's time limit
    floor_stall_seconds: float = 60.0             # floor: and how long it may go without a new or a cleared room
    archive_size: int = 12                        # floor: parked room-entry states per worker
    archive_prob: float = 0.7                     # floor: share of the episodes that start from one of them
    snapshot_prob: float = 0.5                    # floor: share of the room entries that are parked
    archive_boss: float = 0.0                     # floor: share of the archive starts taken from boss-room entries
    build_timeout: float = 60.0                   # s the first build of a launch may take (launch, connect, reset;
                                                  # StartBuilder; 0: no limit; 120 before 2026-10-06)
    reset_timeout: float = 20.0                   # s a later build (a reset) may take (2026-10-06; 0: build_timeout)
    root_anon_mib: float = 0.0                    # 2026-10-06: a root above this RssAnon is replaced at a state
                                                  # boundary (its clones stay; recycle_root); 0: never
    trim_parked: bool = True                      # 2026-10-06: malloc_trim in every parked clone (trim_parked)
    # 2026-10-07: counterfactual branches at decision points (tok_branch.py; floor / run modes; off at share 0)
    branch_share: float = 0.0                     # share of a worker's wall time spent on branch points (0: off)
    branch_seconds: float = 60.0                  # a branch episode's game-time cap (Phase B2: 60, was 180)
    branch_max: int = 3                           # options run per point at most (progressive widening)
    branch_close: float = 0.5                     # widen while every |outcome - option 0's| is below this
    branch_noise: float = 0.0                     # probability of an unconditional replicate pair (the noise floor)
    branch_queue: int = 4                         # points waiting per worker (each holds its room's parked entry)
    branch_kinds: str = 'item,item_left'          # kinds branched (exit is only counted; Phase B2: doors on request)
    branch_item_back: int = 1                     # item points: the restore decision is this many before the pickup step
    branch_walk_seconds: float = 10.0             # scripted walks (other doors, take of item_left) at most
    branch_weights: str = ''                      # outcome weights (tok_branch.DEFAULT_WEIGHTS when empty)
    branch_uncert: float = 0.0                    # weight of the uncertainty term in a point's priority
    # 2026-10-07 (Phase B2, tok_branch's second docstring)
    branch_crn: bool = True                       # common random numbers for the policy in branch pairs (ROW 'crn')
    branch_twin: float = 0.0                      # probability of a twin of the first branch (same option, R and K)
    branch_replicate_max: int = 2                 # conditional replicate pairs per point at most
    branch_noise_prior: float = 1.0               # noise variance of a label before the trainer has an estimate
    branch_reward: str = ''                       # the trainer's reward terms (the score); '' tok_branch.DEFAULT_WEIGHTS
    branch_gamma: float = 0.995                   # the trainer's discount (the score)
    branch_archive: int = 0                       # 1: the take branch's took state and end state into the archive
    branch_left_back: int = 4                     # item_left: restored this many decisions before the leaving step
    # 2026-10-08: the build lab (tok_lab.py; floor / run modes; off at share 0)
    lab_share: float = 0.0                        # share of a worker's wall time on lab jobs (>= 1: whenever a job waits)
    lab_seconds: float = 90.0                     # a lab episode's game-time cap
    lab_specs: list = field(default_factory=list)  # task specs of the panel's groups (load_spec)
    lab_panel: list = field(default_factory=list)  # panel states: (index into lab_specs, seed)
    lab_owner: list = field(default_factory=list)  # the worker of each panel state (tok_lab.assign_panel; empty: k mod W)
    # 2026-10-08: start builds (floor / run modes, training only): this share of the ordinary episodes that start from
    # the floor's start state begin with a random build of 1 .. start_build_max eligible collectibles
    # (tok_lab.transplant_lua into the episode's clone); archive entries carry their episode's build along
    start_build_prob: float = 0.0
    start_build_max: int = 3
    start_build_items: str = ''                   # catalog/lab_items.json ('' : the default next to the bridge)
    # 2026-10-08 (user: augment the base stats for generalisation): stat_aug_prob of the floor-start training episodes
    # get random multipliers of the player's stats, log-uniform within stat_aug's ranges ("damage=0.7:2.5,tears=0.7:2,
    # range=0.8:1.5,shot_speed=0.8:1.3,speed=0.9:1.3"), applied as AbpSetStats offsets on Isaac's base values
    # (abplus.STAT_BASE) in the episode's clone; archive entries carry their episode's stats along
    stat_aug_prob: float = 0.0
    stat_aug: str = ''
    # 2026-10-08: deeper floors first: archive entries are drawn with weight archive_deep^(floor - 1) and the
    # shallowest floor's entries are evicted first (1: uniform, as before)
    archive_deep: float = 1.0
    # 2026-10-10 (SCALING_THESIS.md S5b, regret curriculum, the cheap proxy = Prioritized Level Replay): archive
    # entries drawn by the rank of their score (an EMA of |discounted return - V(first record)| of the episodes started
    # from them; unscored entries rank first), weight rank^(-archive_plr); the lowest score is evicted first. 0: off
    archive_plr: float = 0.0
    # 2026-10-08 (character randomisation; floor / run modes): the PlayerTypes a floor start restarts the run as, with
    # weights ("0,1,2" or "0:4,7:1"; tok_floor.parse_characters); each floor start's character is drawn from the worker's
    # own rng (evaluation: from the seed); archive starts keep the character of the episode they were parked from.
    # '': Isaac only, as before
    characters: str = ''


def attach(names, workers):
    """(blocks, rows [2 x workers], actions, stats, control) from the shared-memory names. Two rows per worker: it
    writes its next record while the server may still read the one before."""
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    rows = np.ndarray((2 * workers,), ROW, buffer=blocks[0].buf)
    actions = np.ndarray((workers, ACTION_F), np.int32, buffer=blocks[1].buf)
    stats = np.ndarray((workers, len(STATS)), np.float64, buffer=blocks[2].buf)
    control = np.ndarray((4,), np.int64, buffer=blocks[3].buf)   # [0] stop, [1] the trainer is updating
    return blocks, rows, actions, stats, control


def attach_teacher(names, workers):
    """(blocks, teacher rows [workers, TEACH_RING], their meta [workers, TEACH_RING, TEACH_META])."""
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    rows = np.ndarray((workers, TEACH_RING), ROW, buffer=blocks[0].buf)
    meta = np.ndarray((workers, TEACH_RING, TEACH_META), np.int32, buffer=blocks[1].buf)
    return blocks, rows, meta


def attach_choices(names, workers):
    """(blocks, choice rows [workers, CHOICE_RING], their meta [workers, CHOICE_RING, CHOICE_META] float64, the aux
    block [workers + 1, AUX_F] float64 (Phase B2: tok_branch AUX_*))."""
    from .tok_branch import AUX_F, CHOICE_META, CHOICE_RING
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    rows = np.ndarray((workers, CHOICE_RING), ROW, buffer=blocks[0].buf)
    meta = np.ndarray((workers, CHOICE_RING, CHOICE_META), np.float64, buffer=blocks[1].buf)
    aux = np.ndarray((workers + 1, AUX_F), np.float64, buffer=blocks[2].buf)
    return blocks, rows, meta, aux


def attach_lab(names, workers):
    """(blocks, jobs [LAB_JOBS, JOB_F] int64, results [workers, LAB_RING, RES_F] float64) of the build lab (tok_lab)."""
    from .tok_lab import JOB_F, LAB_JOBS, LAB_RING, RES_F
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    jobs = np.ndarray((LAB_JOBS, JOB_F), np.int64, buffer=blocks[0].buf)
    res = np.ndarray((workers, LAB_RING, RES_F), np.float64, buffer=blocks[1].buf)
    return blocks, jobs, res


def play_actions(clone, decoder, acts, repeat):
    """Apply the (move, shoot, bomb[, item, pill]) actions back to back in a lean clone; the observation after the last
    one."""
    clone._send({"cmd": "play", "actions": [action_code(*a) for a in acts], "repeat": repeat, "stop_clear": False})
    return read_lean(clone, decoder)


def _sched(pid):
    """(CPU ns, run-queue wait ns) of a process's main thread from /proc/<pid>/schedstat; (0, 0) when gone."""
    try:
        with open(f'/proc/{int(pid)}/schedstat') as f:
            a = f.read().split()
        return int(a[0]), int(a[1])
    except (OSError, ValueError, IndexError, TypeError):
        return 0, 0


TEACH_PROF = os.environ.get('ISAAC_RL_TEACH_PROF', '')   # diagnostic: a directory for per-search cost records


def parse_cpus(text):
    """'1-7,9-15' -> {1, ..., 7, 9, ..., 15}; '' -> None."""
    out = set()
    for part in (text or '').split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out or None


# 2026-10-05 (the teacher's cost; off unless set): ISAAC_RL_WORKER_CPUS pins each worker, and so the instance it
# launches and every clone of it, to these CPUs (e.g. "1-7,9-15", leaving a core to the trainer started with taskset
# or --trainer-cpus); ISAAC_RL_SEARCH_CPUS pins a search's restore clone (and the clones it makes) to these.
WORKER_CPUS = parse_cpus(os.environ.get('ISAAC_RL_WORKER_CPUS', ''))
SEARCH_CPUS = parse_cpus(os.environ.get('ISAAC_RL_SEARCH_CPUS', ''))


KNOWN_HURT = 1.0   # lost[] of a move not searched because its outcome is known to be a hurt (teacher_skip_known)


def known_hurt_move(applied, d, hurt_at):
    """The move whose held clone would replay the episode itself up to the hurt, or None (teacher_skip_known,
    2026-10-05). The clone of move m plays applied[d], then (m, the episode's shot, no bomb) at steps d+1 .. hurt_at - 1
    and the margin: when the episode itself held m without a bomb over d+1 .. hurt_at - 1, that is the episode's own
    action sequence from the restored (exact) state, so the clone takes the episode's damage at step hurt_at - 1: its
    danger bit is 1 without a search. Only the bit is known: its lost[] entry is KNOWN_HURT, not the damage."""
    if d + 1 >= hurt_at:
        return None
    m = applied[d + 1][0]
    for j in range(d + 1, hurt_at):
        if applied[j][0] != m or applied[j][2] != 0:
            return None
    return int(m)


def hindsight(base, reseed, applied, hurt_at, cfg, limit, offset=0, fork_many=None, timing=None, prof=None):
    """Search one hurt of an episode. applied[j]: the action of step j (observation j -> j + 1); hurt_at: the
    observation that showed the damage (step hurt_at - 1 did it). base: a parked clone in the state of observation
    `offset` (the episode's start state, to be reseeded as the episode was, or a snapshot the episode left behind,
    reseed None). Returns [(depth, observation at the restored decision, action under way there, half hearts lost per
    held move, move the episode took)], latest depth first, ending with the first depth that has a safe move.
    fork_many (None: FORK_MANY, i.e. on unless ISAAC_RL_FORK_MANY=0): the nine held moves through the bridge's
    fork_many when the clone offers it (2026-10-04; the same clones, plays and damage totals as one fork + play + close
    per move). timing: a dict that collects seconds per part. prof (diagnostic): a list that gets one dict per depth
    with wall seconds, the CPU / run-queue wait of the base (its side of the restore fork), of the restore clone and
    of each child (/proc schedstat, abp_turbo ABP_FORK_TIMES)."""
    out = []
    many = FORK_MANY if fork_many is None else fork_many
    pc = time.perf_counter
    for k in cfg.teacher_depths:
        d = hurt_at - 1 - k
        if d < offset + 1:
            break
        t0 = pc()
        if prof is not None:
            b0 = _sched(getattr(base, 'pid', 0))
        state = base.fork(reseed=reseed, lean=True, alarm=120)
        try:
            if SEARCH_CPUS:
                try:
                    os.sched_setaffinity(state.pid, SEARCH_CPUS)
                except OSError:
                    pass
            t1 = pc()
            if prof is not None:
                b1 = _sched(getattr(base, 'pid', 0))
                s1 = _sched(state.pid)
            at = play_actions(state, LeanDecoder(), applied[offset:d], cfg.frames_per_decision)
            t2 = pc()
            if prof is not None:
                s2 = _sched(state.pid)
            lost, nb = held_moves(state, at, applied, d, hurt_at, cfg, many, True, prof is not None)
            t3 = pc()
        finally:
            state.close()
        if timing is not None:
            for key, v in (('fork_s', t1 - t0), ('replay_s', t2 - t1), ('children_s', t3 - t2),
                           ('close_s', pc() - t3), ('replayed', d - offset), ('depths', 1)):
                timing[key] = timing.get(key, 0) + v
        if prof is not None:
            prof.append(dict(k=k, d=d, replayed=d - offset, w_fork=t1 - t0, w_replay=t2 - t1, w_children=t3 - t2,
                             w_close=pc() - t3, base_cpu=b1[0] - b0[0], base_wait=b1[1] - b0[1],
                             st_hello=list(s1), st_replay=[s2[0] - s1[0], s2[1] - s1[1]],
                             fm=getattr(state, 'fm_stat', {}), lost=lost, nb=nb))
        out.append((k, at, applied[d], lost, applied[d + 1][0]))
        if min(lost) == 0:
            break
    return out


def held_moves(state, at, applied, d, hurt_at, cfg, many, self_last, stat=False):
    """The nine held moves from `state` (the restored decision d, observation `at`) up to teacher_margin decisions after
    the hurt: (lost[] per move, clones made). self_last: `state` plays the last batch itself and is used up."""
    shots = [a[1] for a in applied[d + 1:hurt_at]] + [applied[hurt_at - 1][1]] * cfg.teacher_margin
    known = known_hurt_move(applied, d, hurt_at) if cfg.teacher_skip_known else None
    moves = [m for m in range(9) if m != known]
    batches = [[applied[d]] + [(move, s, 0) for s in shots] for move in moves]
    found = []
    if many and state.hello.get('fork_many'):
        # one command: nine connectionless clones that report only their end (bridge fork_many)
        ends = state.fork_many([[action_code(*a) for a in b] for b in batches],
                               repeat=cfg.frames_per_decision, alarm=120, self_last=self_last, stat=stat)
        if any(e is None for e in ends):
            raise RuntimeError('fork_many: a clone did not report')
        for damage, dead, _, _ in ends:
            found.append(float(damage - at.damage_taken) + (100.0 if dead else 0.0))
    else:
        for batch in batches:
            clone = state.fork(lean=True, alarm=120)
            try:
                end = play_actions(clone, LeanDecoder(), batch, cfg.frames_per_decision)
                found.append(float(end.damage_taken - at.damage_taken) + (100.0 if end.dead else 0.0))
            finally:
                clone.close()
    lost = [KNOWN_HURT] * 9
    for m, v in zip(moves, found):
        lost[m] = v
    return lost, len(batches)


def hindsight_shared(base, reseed, applied, hurts, cfg, limit, offset=0, fork_many=None, prof=None):
    """Several hurts of one episode with one restore for their first depth (teacher_pair, 2026-10-05): the restore
    clone plays up to the earliest hurt's first restore point, its held moves are cloned there (the clone stays), it
    plays on to the next hurt's point, and so on; the last one plays its own batch. A play continued after a stop is the
    same game as one play (abplus_probe_teacher_exact.py checks the records and the lost[] vectors against
    hindsight). Hurts that need a deeper depth go on through hindsight from there. Returns per hurt (in the given
    order) what hindsight returns."""
    many = FORK_MANY if fork_many is None else fork_many
    k0 = cfg.teacher_depths[0]
    results = {h: [] for h in hurts}
    chain = sorted(h for h in set(hurts) if h - 1 - k0 >= offset + 1)
    if chain:
        t0 = time.perf_counter()
        if prof is not None:
            b0 = _sched(getattr(base, 'pid', 0))
        state = base.fork(reseed=reseed, lean=True, alarm=120)
        try:
            if SEARCH_CPUS:
                try:
                    os.sched_setaffinity(state.pid, SEARCH_CPUS)
                except OSError:
                    pass
            if prof is not None:
                b1 = _sched(getattr(base, 'pid', 0))
            decoder, pos = LeanDecoder(), offset
            for j, h in enumerate(chain):
                d = h - 1 - k0
                s1 = _sched(state.pid) if prof is not None else None
                at = play_actions(state, decoder, applied[pos:d], cfg.frames_per_decision)
                s2 = _sched(state.pid) if prof is not None else None
                last = j == len(chain) - 1
                lost, nb = held_moves(state, at, applied, d, h, cfg, many, last, prof is not None and last)
                if prof is not None:
                    prof.append(dict(k=k0, d=d, replayed=d - pos, w_fork=0.0, w_replay=0.0, w_children=0.0,
                                     w_close=0.0, base_cpu=(b1[0] - b0[0]) if j == 0 else 0,
                                     base_wait=(b1[1] - b0[1]) if j == 0 else 0, st_hello=[0, 0],
                                     st_replay=[s2[0] - s1[0], s2[1] - s1[1]],
                                     fm=dict(getattr(state, 'fm_stat', {})) if last else {}, lost=lost,
                                     nb=nb if last else nb + 1, shared=j))
                    if j == 0 and prof:
                        prof[-1]['w_fork'] = time.perf_counter() - t0
                results[h].append((k0, at, applied[d], lost, applied[d + 1][0]))
                pos = d
        finally:
            state.close()
    out = []
    for h in hurts:
        found = results[h]
        if found and min(found[-1][3]) != 0 and len(cfg.teacher_depths) > 1:
            deeper = dataclasses.replace(cfg, teacher_depths=tuple(cfg.teacher_depths[1:]))
            found = found + hindsight(base, reseed, applied, h, deeper, limit, offset, fork_many, prof=prof)
        out.append(found)
    return out


class SearchSlots:
    """At most cfg.teacher_slots searches at once over the workers of one sampler (2026-10-05): a search first takes
    one of that many file locks (flock on /dev/shm files named after the sampler's instance name and port); without a
    free one the worker does not start a search now. Forks and copy-on-write faults get slower in proportion to how
    many processes do them at once (measured 2026-10-05), so fewer searches at once cost fewer CPU seconds each.
    0: no limit."""

    def __init__(self, cfg):
        self.n = int(cfg.teacher_slots or 0)
        # 2026-10-10: the trainer's pid in the names: stale lock files of an earlier launch (held by orphaned search
        # children) blocked every search of C84 on HPC128 after a relaunch
        self.paths = [f'/dev/shm/abp-teach-{cfg.name}-{cfg.port}-{os.getppid()}-{j}' for j in range(self.n)]
        self.fds = []
        for p in self.paths:
            self.fds.append(os.open(p, os.O_RDWR | os.O_CREAT, 0o600))
        self.order = list(range(self.n))

    def take(self):
        """The index of a slot now held, None when all are taken; -1 without a limit."""
        if not self.n:
            return -1
        import fcntl
        for j in self.order:
            try:
                fcntl.flock(self.fds[j], fcntl.LOCK_EX | fcntl.LOCK_NB)
                return j
            except OSError:
                continue
        return None

    def give(self, j):
        if j is not None and j >= 0:
            import fcntl
            fcntl.flock(self.fds[j], fcntl.LOCK_UN)

    def close(self):
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass


def prof_write(fh, prof, t0, t_enc, c0, held, held_end, n_applied, hurt_at, n_found, queue_len, hurts=1):
    """One search's cost record (ISAAC_RL_TEACH_PROF): wall seconds in all and in the teacher records' encoding, the
    worker thread's CPU seconds, whether the trainer held the workers at its start and end, the run queue then."""
    now = time.perf_counter()
    try:
        with open('/proc/loadavg') as f:
            running = int(f.read().split()[3].split('/')[0])
    except (OSError, ValueError, IndexError):
        running = -1
    fh.write(json.dumps(dict(t=time.time(), wall=now - t0, encode=now - t_enc, worker_cpu=time.thread_time() - c0,
                             held=int(held), held_end=held_end, applied=n_applied, hurt_at=hurt_at, records=n_found,
                             queue=queue_len, running=running, hurts=hurts, depths=prof)) + '\n')
    fh.flush()


def note_overrun(mine, control, held, t0):
    """A search started while the trainer held the workers back (held) and ended after it let them go: the seconds
    past that moment (control[2], time.monotonic_ns of the release) go to the overrun_s stat."""
    if held and not control[1] and control[2] > 0:
        mine[ST['overrun_s']] += max(0.0, min(time.perf_counter() - t0, (time.monotonic_ns() - int(control[2])) / 1e9))


def row_library(cfg):
    """The libabp_turbo.so whose abp_row_encode the workers use (tok_obs.FastRow): the instances' own."""
    return cfg.preload or str(ABP_HOME / 'tools' / 'libabp_turbo.so')


def lean_step(bridge, decoder, action, repeat):
    # the step line when offered; item and pill from a five-part action (2026-10-06)
    bridge.step_line(int(action[0]), int(action[1]), int(action[2]), int(action[3]) if len(action) > 4 else 0, repeat,
                     pill=int(action[4]) if len(action) > 4 else 0)
    return read_lean(bridge, decoder)


def applied_action(action, items):
    """The tuple a worker records for a step (the teacher replays it): (move, shoot, bomb), with items + (item, pill)."""
    return (action[0], action[1], action[2], action[3], action[4]) if items else (action[0], action[1], action[2])


def step_action(episode, action, items, repeat):
    """Send one step of `action` (a worker's ACTION_F answer) as the step line."""
    if items:
        episode.step_line(action[0], action[1], action[2], action[3], repeat, pill=action[4])
    else:
        episode.step_line(action[0], action[1], action[2], 0, repeat)


def teach_meta_row(under_way, lost, taken, k, safe):
    """A teacher record's meta: the action under way (3), nine danger bits, the move taken, the depth, the safe moves,
    and (2026-10-06, items) the under-way item / pill bits in the last column (0 without items)."""
    extra = (int(under_way[3]) | (int(under_way[4]) << 1)) if len(under_way) > 4 else 0
    return list(under_way[:3]) + [int(v > 0) for v in lost] + [taken, k, len(safe), extra]


# The workers' instance defaults (2026-10-05; module docstring "Instance defaults"): abp_turbo's exact fast paths
# (ABP_FAST 1 camera smoothing in C, 2 the shared PNG cache; B14), the bridge's post-update gate (B14), lite forks of
# single-threaded clones (the teacher's label probes). A variable already set in the worker's environment wins.
INSTANCE_ENV_DEFAULTS = {'ABP_FAST': '3', 'ISAAC_RL_PU_SKIP': '1', 'ABP_FORK_LITE': '1'}


def default_stub_list(preload=''):
    """The stub list a worker's instances get when cfg.stub_list is empty: stub_render_h.txt next to the library
    (B14), else stub_render_g.txt; '' when neither is there (the launch mode's own list)."""
    tools = Path(preload or (ABP_HOME / 'tools' / 'libabp_turbo.so')).parent
    for name in ('stub_render_h.txt', 'stub_render_g.txt'):
        if (tools / name).is_file():
            return str(tools / name)
    return ''


def apply_instance_defaults(environ=None):
    """INSTANCE_ENV_DEFAULTS into the environment the instances inherit, where not set already. Returns the values."""
    environ = os.environ if environ is None else environ
    for k, v in INSTANCE_ENV_DEFAULTS.items():
        environ.setdefault(k, v)
    return {k: environ[k] for k in INSTANCE_ENV_DEFAULTS}


def parse_faults(index, text=None):
    """ISAAC_RL_FAULT_HANG=<worker>:<build>:<stop|spin|crash>[,...] -> {build number: mode} of worker `index` (tests)."""
    out = {}
    for part in (os.environ.get('ISAAC_RL_FAULT_HANG', '') if text is None else text).split(','):
        bits = part.strip().split(':')
        if len(bits) == 3 and bits[0].isdigit() and int(bits[0]) == index:
            out[int(bits[1])] = bits[2]
    return out


class StartBuilder:
    """A worker's start states, built by its root instance in a helper thread (the worker plays on meanwhile), with a
    time limit (2026-10-05). start() begins the next build (the seed from next_seed, prepare(inst) first in the thread);
    ready is set when it is done and take() gives (seed, exception or None). A build abandoned by a newer start()
    (after a stall: stalled() is true once it has run cfg.build_timeout s) can no longer deliver: each build has a
    number and only the newest one's result is kept."""

    def __init__(self, index, inst, cfg, next_seed, prepare=None):
        self.index, self.inst, self.cfg, self.next_seed, self.prepare = index, inst, cfg, next_seed, prepare
        self.lock, self.ready = threading.Lock(), threading.Event()
        self.number = 0      # builds started (1 = the one with the first launch)
        self.result = None
        self.t0 = 0.0
        self.seed = None
        self.retried = None  # the seed whose build is being tried a second time (recover), None
        self.limit = cfg.build_timeout   # s the build in flight may take (start)
        self.done = []       # diagnostic: (build number, seed, seconds, ok, first of a launch) of finished builds
        self.faults = parse_faults(index)

    def start(self, seed=None):
        """The next build: the next seed, or `seed` again (2026-10-06: a retry after a failed or stalled build)."""
        if seed is None:
            seed = self.next_seed()
        # the first build of a launch (it connects first) has cfg.build_timeout, a later one cfg.reset_timeout
        first = not getattr(self.inst, 'digest_ready', True)
        self.limit = self.cfg.build_timeout if first or self.cfg.reset_timeout <= 0 else self.cfg.reset_timeout
        with self.lock:
            self.number += 1
            number = self.number
            self.result = None
            self.ready.clear()
        self.seed, self.t0 = seed, time.monotonic()
        threading.Thread(target=self._run, args=(seed, number), daemon=True).start()

    def _fault(self, number):
        mode = self.faults.get(number)
        if not mode:
            return
        inst = self.inst
        env = inst.env
        if mode == 'spin' and env is not None and getattr(env, 'connected', False):
            env.bridge._send({"cmd": "lua", "code": "while true do end"})   # the reply never comes
        elif mode == 'crash':   # 2026-10-06: the game process dies (as the start-up SIGSEGV seen in training)
            os.kill(inst.proc.pid, signal.SIGSEGV)
        else:
            os.kill(inst.proc.pid, signal.SIGSTOP)
        print(f'tok worker {self.index}: fault injected at build {number}: {mode} (pid {inst.proc.pid})',
              file=sys.stderr, flush=True)

    def _run(self, seed, number):
        exc = None
        t0 = time.monotonic()
        first = not getattr(self.inst, 'digest_ready', True)   # the first reset of this launch (connects first)
        try:
            if seed is not None:
                self._fault(number)
                if self.prepare is not None:
                    self.prepare(self.inst)
                self.inst.reset(seed)
        except Exception as e:   # the worker relaunches the instance
            exc = e
        with self.lock:
            if number == self.number:
                self.result = (seed, exc)
                self.ready.set()
                if seed is not None:
                    self.done.append((number, seed, round(time.monotonic() - t0, 3), exc is None, first))
                    del self.done[:-64]

    def take(self):
        with self.lock:
            out, self.result = self.result, None
        return out

    def stalled(self):
        return (self.limit > 0 and not self.ready.is_set() and time.monotonic() - self.t0 > self.limit)

    def wait(self, stop, on_stall):
        """Until the build is done (True) or stop() (False); a stalled build goes to on_stall(), which starts the next."""
        while not self.ready.wait(0.2):
            if stop():
                return False
            if self.stalled():
                on_stall()
        return True


def stall_dir(name):
    """Where a root instance's stall evidence goes (stall_evidence): $ISAAC_RL_STALL_DIR/<name>, else the instance's
    directory under ABP_HOME/instances."""
    base = os.environ.get('ISAAC_RL_STALL_DIR', '')
    return Path(base) / name.lower() if base else ABP_HOME / 'instances' / name.lower() / 'stalls'


def _copy_text(src, dst, tail=0):
    try:
        with open(src, 'rb') as f:
            data = f.read()
        if tail:
            data = b'\n'.join(data.split(b'\n')[-tail:])
        with open(dst, 'wb') as f:
            f.write(data)
    except OSError:
        pass


def stall_evidence(inst, builder, why, index):
    """Diagnostic (2026-10-06): what a root instance whose start-state build failed or stalled was doing, saved before
    the worker kills it, in stall_dir(name)/<time>-w<worker>-b<build>/: info.json (why, seed, seconds, pid, port, the
    client's state: connected, hello, last command), /proc of the game (status, wchan, stat, per thread stat / comm /
    wchan, stack and syscall where readable), three main-thread stacks (SIGUSR2 to a game launched with
    ABP_STACK_DUMP_DIR, abp_turbo), the instance's stdout.log and the tail of its log.txt (the bridge's log lines), and
    the sockets on its port (ss). Returns the directory ('' when nothing could be saved); never raises."""
    try:
        proc = inst.proc
        pid = proc.pid if proc is not None else None
        d = stall_dir(inst.name) / f"{time.strftime('%m%d-%H%M%S')}-w{index}-b{builder.number}"
        d.mkdir(parents=True, exist_ok=True)
        bridge = inst.env.bridge if inst.env is not None else None
        info = dict(why=why, worker=index, build=builder.number, seed=builder.seed, retried=builder.retried,
                    seconds=round(time.monotonic() - builder.t0, 2), pid=pid, port=inst.port, name=inst.name,
                    launches=inst.launches, time=time.time(),
                    exit_status=proc.poll() if proc is not None else None,
                    connected=bool(bridge is not None and getattr(bridge, '_sock', None) is not None),
                    bridge_port=getattr(bridge, 'port', None), hello=getattr(bridge, 'hello', None),
                    last_cmd=getattr(bridge, 'last_cmd', None), port_file=getattr(bridge, 'port_file', None))
        if pid:
            stacks = Path(getattr(inst, 'stack_dir', '') or '')
            if str(stacks) and stacks.is_dir():
                for _ in range(3):
                    try:
                        os.kill(pid, signal.SIGUSR2)
                    except OSError:
                        break
                    time.sleep(0.3)
                time.sleep(0.3)
                for f in stacks.glob(f'stack-{pid}-*.txt'):
                    try:
                        os.replace(f, d / f.name)
                    except OSError:
                        pass
            proc_dir = Path(f'/proc/{pid}')
            for name in ('status', 'wchan', 'stat', 'stack', 'syscall', 'sched'):
                _copy_text(proc_dir / name, d / f'proc-{name}.txt')
            try:
                lines = []
                for t in sorted(proc_dir.joinpath('task').iterdir(), key=lambda p: int(p.name)):
                    parts = []
                    for name in ('comm', 'wchan', 'stat'):
                        try:
                            parts.append((t / name).read_text().strip())
                        except OSError:
                            parts.append('?')
                    lines.append(f'{t.name}\t' + '\t'.join(parts))
                (d / 'threads.txt').write_text('\n'.join(lines) + '\n')
            except OSError:
                pass
        home = ABP_HOME / 'instances' / inst.name.lower()
        _copy_text(home / 'stdout.log', d / 'stdout.log')
        _copy_text(home / 'data' / 'binding of isaac afterbirth+' / 'log.txt', d / 'log-tail.txt', tail=300)
        try:
            import subprocess
            ports = {str(inst.port)} | ({str(info['bridge_port'])} if info['bridge_port'] else set())
            out = subprocess.run(['ss', '-tanpH'], capture_output=True, text=True, timeout=10).stdout
            keep = [ln for ln in out.splitlines() if any(f':{p} ' in ln + ' ' for p in ports)]
            (d / 'ss.txt').write_text('\n'.join(keep) + f'\n# sockets in all: {len(out.splitlines())}\n')
        except Exception:
            pass
        (d / 'info.json').write_text(json.dumps(info, default=str, indent=1))
        return str(d)
    except Exception:
        return ''


def note_port(inst, mine, index, seen):
    """Once per launch (seen: a one-element list, the launch last looked at): a root whose bridge listens on another
    port than the configured one (it was taken: abp_bridge.lua's ISAAC_RL_PORT_FILE fallback, 2026-10-06) is counted
    (port_moves) and reported."""
    if inst.env is None or seen[0] == inst.launches:
        return
    seen[0] = inst.launches
    actual = getattr(inst.env.bridge, 'port', inst.port)
    if actual and actual != inst.port:
        mine[ST['port_moves']] += 1
        print(f'tok worker {index}: port {inst.port} was taken; the bridge of {inst.name} listens on {actual}',
              file=sys.stderr, flush=True)


TRIM_LUA = "return tostring(os.getenv('ABP_MALLOC_INFO:TRIM'))"


def trim_parked(clone, cfg, mine):
    """2026-10-06 (cfg.trim_parked): a clone that is parked (a template, a room entry of tok_floor) gives its free heap
    pages back: glibc malloc_trim(0) in the clone (abp_turbo ABP_MALLOC_INFO:TRIM, through the bridge's lua command;
    about 2-3 ms). The pages inside free malloc chunks are unmapped; once the processes it shared them with have moved
    on they would otherwise stay as the parked clone's private memory. The game from a trimmed clone is the same as from
    an untrimmed one (abplus_probe_trim.py: payloads and digests). A failure is ignored (the clone stays untrimmed)."""
    if not cfg.trim_parked or clone is None:
        return
    try:
        clone.lua(TRIM_LUA)
        mine[ST['trims']] += 1
    except (BridgeError, OSError, RuntimeError, ValueError):
        pass


def recycle_root(inst, cfg, mine, index):
    """2026-10-06: a root instance whose anonymous RSS (RssAnon: its heap; the shared PNG cache and the mapped files
    left out) is above cfg.root_anon_mib is replaced, at a state boundary: right after the
    new template was forked from it and before the next build starts. Only the root's process ends (Instance.retire);
    everything cloned from it (the template just forked, archive entries, queued restore points, a running episode) is
    a process of its own and stays. A new root is launched; the builder's next build connects to it (its first build:
    launch + connect + reset, under the same time limit). The training's states and episodes are the same as without
    it (the next seed is built as before); only the root's process is new. Returns True when it recycled."""
    if cfg.root_anon_mib <= 0 or inst.proc is None:
        return False
    rss = inst.rss_mib('RssAnon')
    if rss <= cfg.root_anon_mib:
        return False
    inst.retire()
    inst.launch()
    mine[ST['recycles']] += 1
    print(f'tok worker {index}: root {inst.name} recycled at {rss:.0f} MiB RssAnon (launch {inst.launches})',
          file=sys.stderr, flush=True)
    return True


def term_to_exit():
    """SIGTERM (multiprocessing's terminate) ends a worker through its finally block (its instance killed), once."""
    flag = []

    def handler(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        flag.append(signum)
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, handler)
    return flag


def worker_main(index, cfg, names, conn, teacher_names=None):
    blocks, rows, actions, stats, control = attach(names, cfg.workers)
    teaching = cfg.teacher and teacher_names is not None and not cfg.eval_seeds
    if teaching:
        teach_blocks, teach_rows, teach_meta = attach_teacher(teacher_names, cfg.workers)
        blocks = blocks + teach_blocks
    t_begin = time.perf_counter()
    queue = []      # hurts to search: (template, reseed, applied actions, hurt_at, seed, episode)

    def search(item):
        """One queued hurt (with teacher_pair: and the next queued ones of the same episode, one restore for their
        first depth): its teacher records into the ring."""
        tpl, reseed_, applied_, hurt_at, seed_, episode_ = item
        items = [item]
        while cfg.teacher_pair and queue and queue[-1][0] is tpl and queue[-1][2] is applied_:
            items.append(queue.pop())
        t0 = time.perf_counter()
        held = control[1]
        prof = [] if prof_file is not None else None
        c0 = time.thread_time()
        try:
            if len(items) > 1:
                founds = hindsight_shared(tpl, reseed_, applied_, [it[3] for it in items], cfg, limit, prof=prof)
            else:
                founds = [hindsight(tpl, reseed_, applied_, hurt_at, cfg, limit, prof=prof)]
        except (BridgeError, OSError, RuntimeError, ValueError):
            mine[ST['errors']] += 1
            founds = [[] for _ in items]
        mine[ST['searches']] += len(items)
        mine[ST['queue']] = len(queue)
        note_overrun(mine, control, held, t0)
        t_enc = time.perf_counter()
        n_found = 0
        for it, found in zip(items, founds):
            n_found += len(found)
            for k, at, under_way, lost, taken in found:
                slot = int(mine[ST['teach']]) % TEACH_RING
                trow = teach_rows[index, slot:slot + 1]
                encode_row(at, EpisodeState(limit, items=cfg.items), trow, it[3] - 1 - k)
                trow['episode'], trow['seed'], trow['group'], trow['first'] = it[5], it[4], group, 0
                safe = [m for m in range(9) if lost[m] == 0]
                teach_meta[index, slot] = teach_meta_row(under_way, lost, taken, k, safe)
                mine[ST['teach']] += 1
                mine[ST['avoidable']] += bool(safe)
        mine[ST['teach_s']] += time.perf_counter() - t0
        if prof is not None:
            prof_write(prof_file, prof, t0, t_enc, c0, held, int(control[1]), len(applied_), hurt_at, n_found,
                       len(queue), len(items))

    def search_gated():
        """The newest queued hurt searched when a search slot is free (cfg.teacher_slots); False when none was."""
        j = gate.take()
        if j is None:
            return False
        try:
            search(queue.pop())
        finally:
            gate.give(j)
        return True

    def receive():
        """The server's answer. While the trainer updates, queued searches run instead of waiting."""
        while queue and not conn.poll(0.0005):
            if control[1] and not control[0]:
                search_gated()
        conn.recv_bytes()
    prof_file = open(os.path.join(TEACH_PROF, f'w{index}.jsonl'), 'a') if TEACH_PROF and teaching else None
    gate = SearchSlots(cfg) if teaching else None
    if WORKER_CPUS:
        os.sched_setaffinity(0, WORKER_CPUS)
    if cfg.nice > 0:
        os.nice(cfg.nice)
    mine = stats[index]
    slots = (rows[2 * index:2 * index + 1], rows[2 * index + 1:2 * index + 2])
    slot_address = (slots[0].ctypes.data, slots[1].ctypes.data)
    fast_fn = fast_row_function(row_library(cfg))   # None: LeanDecoder + encode_row as before
    group = cfg.assign[index]
    spec = cfg.specs[group]
    os.environ.update(FORK_ENV)
    apply_instance_defaults()
    if cfg.items:   # the root's bridge (and every clone of it) sends the inventory block
        os.environ['ISAAC_RL_LEAN_ITEMS'] = '1'
    if os.environ.get('ISAAC_RL_WATCH_PID', '1') != '0':   # see tok_sampler.WATCH_NOTE
        os.environ['ABP_WATCH_PID'] = str(os.getpid())   # the instance ends when this worker is gone
    gx = GxConfig(bridge_lua=cfg.bridge_lua, preload=cfg.preload, al_stopped=True, nice=cfg.nice,
                  frames_per_decision=cfg.frames_per_decision, start_hp=cfg.start_hp[1], bombs=cfg.bombs,
                  stub_list=cfg.stub_list or default_stub_list(cfg.preload), die_with_parent=True)
    rng = np.random.default_rng([cfg.seed, index])
    seconds = cfg.episode_seconds or float(spec['seconds'])
    limit = int(round(seconds * 30 / cfg.frames_per_decision))
    evaluation = bool(cfg.eval_seeds)
    todo = list(cfg.eval_seeds[index]) if evaluation else None
    state_n = 0
    inst = template = episode = builder = None
    terminated = term_to_exit()
    parent = mp.parent_process()
    port_seen = [0]

    def next_seed():
        nonlocal state_n
        if evaluation:
            return todo.pop(0) if todo else None
        seed = cfg.seed + index + cfg.workers * state_n
        state_n += 1
        return seed

    def prepare(inst_):
        inst_.cfg.start_hp = int(cfg.start_hp[1] if evaluation else rng.integers(cfg.start_hp[0], cfg.start_hp[1] + 1))

    def stopping():
        return bool(control[0]) or (parent is not None and not parent.is_alive())

    def recover(why):
        """The root instance failed or hung: what was cloned from it dropped, the instance killed (checked gone) and
        relaunched, the same seed built once more (relaunch_after)."""
        nonlocal template
        mine[ST['errors']] += 1
        if why == 'stall':
            mine[ST['stalls']] += 1
        if template is not None:
            n_q = len(queue)
            queue[:] = [q for q in queue if q[0] is not template]
            mine[ST['dropped']] += n_q - len(queue)
            try:
                template.close()
            except Exception:
                pass
            template = None
        relaunch_after(why)

    def relaunch_after(why):
        """Evidence saved, the instance killed and relaunched, the same seed built once more (2026-10-06; a seed whose
        second build also fails is given up: the next one)."""
        seconds = time.monotonic() - builder.t0
        where = stall_evidence(inst, builder, why, index)
        left = inst.kill()
        retry = builder.seed if builder.seed is not None and builder.seed != builder.retried else None
        print(f'tok worker {index}: start state of seed {builder.seed} {why} (build {builder.number}, '
              f'{seconds:.1f} s); instance killed{" (left: %s)" % left if left else ""}, relaunched; '
              f'{"the same seed again" if retry is not None else "the next seed"}'
              f'{"; evidence " + where if where else ""}', file=sys.stderr, flush=True)
        time.sleep(1.0)
        inst.launch()
        builder.retried = retry
        if retry is not None:
            mine[ST['retries']] += 1
        builder.start(seed=retry)

    try:
        inst = Instance(f'{cfg.name}{index}', cfg.port + index, gx, spec)
        builder = StartBuilder(index, inst, cfg, next_seed, prepare)
        builder.start()
        episodes_left, episode_n, state_seed = 0, 0, None
        while not stopping():
            if builder.stalled():   # the next start state's build hangs (training went on with the old one)
                recover('stall')
            # the next start state: when this one has served its episodes and the next is built (training goes on with
            # the old one meanwhile; an evaluation waits, each of its seeds is played once)
            if template is None or (episodes_left <= 0 and (evaluation or builder.ready.is_set())):
                t_state = time.perf_counter()
                if not builder.wait(stopping, lambda: recover('stall')):
                    break
                state_seed, exc = builder.take()
                if exc is not None:
                    recover(f'failed ({type(exc).__name__}: {str(exc)[:200]})')
                    continue
                if state_seed is None:   # evaluation: no seed left
                    break
                note_port(inst, mine, index, port_seen)
                builder.retried = None
                try:
                    fresh = inst.env.bridge.fork(tag='template', alarm=0)
                except (BridgeError, OSError, RuntimeError, ValueError) as exc:
                    recover(f'not cloned ({type(exc).__name__}: {str(exc)[:200]})')
                    continue
                trim_parked(fresh, cfg, mine)   # cfg.trim_parked (on by default, 2026-10-06)
                if template is not None:
                    n_q = len(queue)
                    queue[:] = [q for q in queue if q[0] is not template]   # its hurts can no longer be restored
                    mine[ST['dropped']] += n_q - len(queue)
                    template.close()
                template = fresh
                mine[ST['states']] += 1
                episodes_left = 1 if evaluation else cfg.episodes_per_state
                recycle_root(inst, cfg, mine, index)   # cfg.root_anon_mib (off by default)
                builder.start()
                mine[ST['state_s']] += time.perf_counter() - t_state
            elif episodes_left <= 0:
                mine[ST['extra_episodes']] += 1
            t0 = time.perf_counter()
            try:
                reseed = state_seed % (2 ** 31 - 1) + 1 if evaluation else int(rng.integers(1, 2 ** 31 - 1))
                episode = template.fork(alarm=cfg.clone_alarm, reseed=reseed, lean=True)
                decoder = LeanDecoder()
                episode._send({"cmd": "obs"})
                obs = read_lean(episode, decoder)
            except (BridgeError, OSError, RuntimeError, ValueError):
                mine[ST['errors']] += 1
                template = None
                continue
            mine[ST['fork_s']] += time.perf_counter() - t0
            mine[ST['forks']] += 1
            episodes_left -= 1
            if obs.clear or obs.dead:   # nothing to fight in this room
                mine[ST['empty']] += 1
                episode.close()
                n_q = len(queue)
                queue[:] = [q for q in queue if q[0] is not template]
                mine[ST['dropped']] += n_q - len(queue)
                template.close()
                episode = template = None   # wait for the next state
                continue
            episode_n += 1
            st, t_ep, waiting, action = EpisodeState(limit, items=cfg.items), 0, False, (0, 0, 0, 0, 0)
            enc = FastRow(fast_fn, st, decoder, episode_n, state_seed, group) if fast_fn is not None else None
            item = obs   # the record's observation: a LeanObs, or with FastRow the step's raw payload
            applied, hurts = [], []
            try:
                while True:
                    t0 = time.perf_counter()
                    row = slots[t_ep & 1]
                    if enc is not None:
                        done = enc.encode(item, row, slot_address[t_ep & 1], t_ep)
                        hurt = enc.hurt
                    else:
                        done = encode_row(obs, st, row, t_ep)
                        row['episode'], row['seed'], row['group'], row['first'] = (episode_n, state_seed, group,
                                                                                   t_ep == 0)
                        hurt = row['hurt'][0] > 0
                    if hurt:
                        hurts.append(t_ep)
                    t1 = time.perf_counter()
                    mine[ST['encode_s']] += t1 - t0
                    if waiting:   # the answer to the record before this one: the action of the step to come
                        receive()
                        action = tuple(int(v) for v in actions[index])
                    conn.send_bytes(b'1' if t_ep & 1 else b'0')
                    waiting = True
                    if done or control[0]:
                        receive()   # nothing may be outstanding when the next episode starts
                        waiting = False
                    t2 = time.perf_counter()
                    mine[ST['wait_s']] += t2 - t1
                    if done or control[0]:
                        break
                    applied.append(applied_action(action, cfg.items))
                    if enc is not None:
                        step_action(episode, action, cfg.items, cfg.frames_per_decision)
                        item = read_lean_raw(episode)
                    else:
                        obs = lean_step(episode, decoder, applied_action(action, cfg.items), cfg.frames_per_decision)
                    mine[ST['step_s']] += time.perf_counter() - t2
                    t_ep += 1
                    mine[ST['decisions']] += 1
                    mine[ST['frames']] += cfg.frames_per_decision
                mine[ST['episodes']] += 1
                if done:
                    mine[ST[{1: 'wins', 2: 'deaths', 3: 'timeouts'}[done]]] += 1
                mine[ST['hurts']] += len(hurts)
                if teaching and hurts:
                    picks = [hurts[i] for i in sorted(rng.permutation(len(hurts))[:cfg.teacher_per_episode])]
                    queue.extend((template, reseed, applied, hurt_at, state_seed, episode_n) for hurt_at in picks)
                    mine[ST['dropped']] += max(0, len(queue) - cfg.teacher_queue)
                    del queue[:-cfg.teacher_queue]
                    mine[ST['queue']] = len(queue)
                    while queue and not control[0] and \
                            mine[ST['teach_s']] < cfg.teacher_share * (time.perf_counter() - t_begin):
                        if not search_gated():
                            break
            except (BridgeError, OSError, RuntimeError, ValueError):
                # the server may hold a record of this episode without its end: the next record is a first one
                mine[ST['errors']] += 1
                if waiting:
                    conn.recv_bytes()
            finally:
                t_close = time.perf_counter()
                try:
                    episode.close()
                except OSError:
                    pass
                episode = None
                mine[ST['close_s']] += time.perf_counter() - t_close
    except Exception:
        traceback.print_exc()
        mine[ST['errors']] += 1000
    finally:
        mine[ST['finished']] = 1
        # killed outright (SIGTERM) or a build still in flight (it may hang): the instance and its clones are killed;
        # otherwise closed as before
        hard = bool(terminated) or (builder is not None and not builder.ready.is_set())
        if not hard:
            for c in (episode, template):
                try:
                    if c is not None:
                        c.close()
                except OSError:
                    pass
        if inst is not None:
            if hard:
                inst.kill()
            else:
                inst.close()
        if gate is not None:
            gate.close()
        try:
            conn.send_bytes(b'x')
        except OSError:
            pass
        for b in blocks:
            b.close()


class TokSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.workers
        sizes = (ROW.itemsize * 2 * n, 4 * ACTION_F * n, 8 * len(STATS) * n, 8 * 4)
        self.blocks = [shared_memory.SharedMemory(create=True, size=s) for s in sizes]
        names = [b.name for b in self.blocks]
        self.attached, self.rows, self.actions, self.stats, self.control = attach(names, n)
        teacher_names = None
        self.teach_rows = self.teach_meta = None
        if cfg.teacher and not cfg.eval_seeds:
            more = [shared_memory.SharedMemory(create=True, size=s)
                    for s in (ROW.itemsize * n * TEACH_RING, 4 * n * TEACH_RING * TEACH_META)]
            teacher_names = [b.name for b in more]
            self.blocks += more
            attached, self.teach_rows, self.teach_meta = attach_teacher(teacher_names, n)
            self.attached += attached
            self.teach_seen = np.zeros(n, np.int64)
        # 2026-10-07: the choice records of counterfactual branches (tok_branch; floor / run modes, training only)
        choice_names = None
        self.choice_rows = self.choice_meta = self.aux = None
        if cfg.branch_share > 0 and cfg.mode in ('floor', 'run') and not cfg.eval_seeds:
            from .tok_branch import AUX_F, CHOICE_META, CHOICE_RING
            more = [shared_memory.SharedMemory(create=True, size=s)
                    for s in (ROW.itemsize * n * CHOICE_RING, 8 * n * CHOICE_RING * CHOICE_META, 8 * (n + 1) * AUX_F)]
            choice_names = [b.name for b in more]
            self.blocks += more
            attached, self.choice_rows, self.choice_meta, self.aux = attach_choices(choice_names, n)
            self.aux[:] = 0
            self.attached += attached
            self.choice_seen = np.zeros(n, np.int64)
        # 2026-10-08: the build lab's job slots and result rings (tok_lab; floor / run modes, training only)
        lab_names = None
        self.lab_jobs = self.lab_res = None
        if cfg.lab_share > 0 and cfg.mode in ('floor', 'run') and not cfg.eval_seeds and cfg.lab_panel:
            from .tok_lab import JOB_F, LAB_JOBS, LAB_RING, RES_F
            more = [shared_memory.SharedMemory(create=True, size=s)
                    for s in (8 * LAB_JOBS * JOB_F, 8 * n * LAB_RING * RES_F)]
            lab_names = [b.name for b in more]
            self.blocks += more
            attached, self.lab_jobs, self.lab_res = attach_lab(lab_names, n)
            self.lab_jobs[:] = 0
            self.attached += attached
            self.lab_seen = np.zeros(n, np.int64)
        # 2026-10-10 (teacher v2, tok_teacher2; floor / run modes, training only): the lanes (a second record slot and
        # pipe per worker that the actor answers outside the rollout) and the imitation records' rings
        v2_names = None
        self.lane_rows = self.lane_io = self.lane_out = self.main_value = self.v2_rows = self.v2_meta = None
        self.main_h = self.lane_h = self.v2_h = None   # 2026-10-11 (memory, with teacher v2)
        self.lane_ready = []
        self._lanes = {}
        if cfg.teacher_v2 and cfg.mode in ('floor', 'run') and not cfg.eval_seeds:
            from .tok_teacher2 import attach_v2, v2_sizes
            more = [shared_memory.SharedMemory(create=True, size=s) for s in v2_sizes(n)]
            v2_names = [b.name for b in more]
            self.blocks += more
            attached, self.lane_rows, self.lane_io, self.lane_out, self.main_value, self.v2_rows, self.v2_meta = \
                attach_v2(v2_names, n)
            self.lane_rows[:] = np.zeros(1, ROW)[0]
            self.lane_io[:] = 0
            self.lane_out[:] = 0
            self.main_value[:] = 0
            self.attached += attached
            self.v2_seen = np.zeros(n, np.int64)
            if cfg.memory_dim > 0:   # 2026-10-11 (memory): the GRU states of the main records, lanes and V2 ring
                from .tok_teacher2 import attach_v2_mem, v2_mem_sizes
                more = [shared_memory.SharedMemory(create=True, size=s) for s in v2_mem_sizes(n, cfg.memory_dim)]
                v2_names = v2_names + [b.name for b in more]
                self.blocks += more
                attached, self.main_h, self.lane_h, self.v2_h = attach_v2_mem(v2_names[6:], n, cfg.memory_dim)
                self.main_h[:] = 0
                self.lane_h[:] = 0
                self.attached += attached
        self.stats[:] = 0
        self.control[:] = 0
        # 2026-10-06: one decoded-PNG cache for all the instances of this sampler (abp_turbo ABP_PNG_CACHE_FILE) instead
        # of one per root; the workers inherit the variable, removed again in close(). ISAAC_RL_PNG_SHARED=0: as before.
        self.png_file = None
        if os.environ.get('ISAAC_RL_PNG_SHARED', '1') != '0' and 'ABP_PNG_CACHE_FILE' not in os.environ:
            self.png_file = f'/dev/shm/abp-png-v1-{cfg.name}-{cfg.port}'
            os.environ['ABP_PNG_CACHE_FILE'] = self.png_file
        ctx = mp.get_context('spawn')
        self.conns, self.procs = [], []
        target = worker_main
        if cfg.mode in ('floor', 'run'):
            from .tok_floor import floor_worker_main as target
        self.lane_conns = []
        for i in range(n):
            ours, theirs = ctx.Pipe()
            # (tok_floor's worker only; 2026-10-08: and the lab's blocks)
            extra = (choice_names, lab_names) if lab_names is not None else \
                (choice_names,) if choice_names is not None else ()
            kw = {}
            if v2_names is not None:   # 2026-10-10 (teacher v2): the worker's lane
                lane_ours, lane_theirs = ctx.Pipe()
                kw = dict(v2_names=v2_names, v2_conn=lane_theirs)
            proc = ctx.Process(target=target, args=(i, cfg, names, theirs, teacher_names) + extra, kwargs=kw,
                               daemon=True)
            proc.start()
            theirs.close()
            if v2_names is not None:
                lane_theirs.close()
                self.lane_conns.append(lane_ours)
            self.conns.append(ours)
            self.procs.append(proc)
        self.index = {c: i for i, c in enumerate(self.conns)}
        self.live = list(self.conns)
        # poll_ready: one persistent poll set over the workers' pipes (connection.wait builds a selector per call)
        self._poller = select.poll()
        self._by_fd = {}
        self._fds = [c.fileno() for c in self.conns]
        for c in self.conns:
            self._poller.register(c.fileno(), select.POLLIN)
            self._by_fd[c.fileno()] = c
        # 2026-10-10 (teacher v2): the lanes' pipes in the same poll set (a lane request wakes the actor's poll; it is
        # collected in lane_ready, not returned as a record) and in a poll set of their own (wait_lanes)
        self._lane_poller = select.poll() if self.lane_conns else None
        for i, c in enumerate(self.lane_conns):
            self._lanes[c.fileno()] = i
            self._poller.register(c.fileno(), select.POLLIN)
            self._lane_poller.register(c.fileno(), select.POLLIN)

    def _lane_message(self, fd):
        """A lane's request (one 5-byte message) into lane_ready; a closed lane leaves the poll sets."""
        try:
            msg = os.read(fd, 5)
        except OSError:
            msg = b''
        if msg in (_MSG0, _MSG1):
            self.lane_ready.append(self._lanes[fd])
            return
        for p in (self._poller, self._lane_poller):
            try:
                p.unregister(fd)
            except (KeyError, ValueError, OSError):
                pass
        self._lanes.pop(fd, None)

    def wait_lanes(self, timeout):
        """Teacher v2: waits up to `timeout` s for lane requests (while the actor has no records to answer, e.g. while
        it waits for the learner); True when some are in lane_ready."""
        if self._lane_poller is None:
            return False
        if not self.lane_ready:
            for fd, _ in self._lane_poller.poll(max(0, int(timeout * 1000))):
                if fd in self._lanes:
                    self._lane_message(fd)
        return bool(self.lane_ready)

    def take_lanes(self):
        """The workers whose lane requests are waiting (int64 array, distinct), lane_ready emptied."""
        idx = np.unique(np.array(self.lane_ready, np.int64))
        self.lane_ready = []
        return idx

    def reply_lanes(self, idx):
        for i in idx:
            try:
                os.write(self.lane_conns[i].fileno(), _REPLY)
            except OSError:
                pass

    def v2_records(self, with_h=False):
        """The teacher-v2 imitation records written since the last call: (rows, meta [n, V2_META]); None without.
        with_h (2026-10-11, memory): (rows, meta, h [n, memory_dim] float32: each record's GRU input state; zeros where
        the ring has none)."""
        if self.v2_rows is None:
            return None
        from .tok_teacher2 import V2_META, V2_RING
        rows, meta, hs = [], [], []
        dim = int(self.cfg.memory_dim)
        for i in range(self.cfg.workers):
            count = int(self.stats[i, ST['v2_records']])
            first = max(int(self.v2_seen[i]), count - V2_RING)
            for c in range(first, count):
                rows.append(self.v2_rows[i, c % V2_RING].copy())
                meta.append(self.v2_meta[i, c % V2_RING].copy())
                if with_h:
                    hs.append(self.v2_h[i, c % V2_RING].copy() if self.v2_h is not None else np.zeros(dim, np.float32))
            self.v2_seen[i] = count
        if not rows:
            out = np.zeros((0,), ROW), np.zeros((0, V2_META), np.float64)
            return out + (np.zeros((0, dim), np.float32),) if with_h else out
        out = np.stack(rows), np.stack(meta)
        return out + (np.stack(hs).astype(np.float32),) if with_h else out

    def _drop(self, c):
        if c in self.live:
            self.live.remove(c)
            i = self.index.get(c)
            if i is not None:   # reap an ended worker now: as a zombie it still answers kill(pid, 0), and its
                try:            # instance's ABP_WATCH_PID watcher would not see it gone (2026-10-05)
                    self.procs[i].join(0.5)
                except (AssertionError, ValueError, OSError):
                    pass
        try:
            self._poller.unregister(c.fileno())
        except (KeyError, ValueError, OSError):
            pass

    def poll_ready(self, idle=0.2, want=1, max_wait=0.0):
        """The workers whose records are ready: waits up to `idle` s for the first one, takes every other record that
        is already there, and while fewer than `want` are in hand keeps sweeping (without sleeping) for up to
        `max_wait` s after the first. want=1: no record is held back to fill a batch (the actor's call costs about
        the same for 1 or 16 records). (worker indices, row slots) as int64 arrays."""
        idx, slots = [], []
        if not self.live:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        timeout = max(0, int(idle * 1000))
        first = None
        while True:
            for fd, ev in self._poller.poll(timeout):
                if self._lanes and fd in self._lanes:   # 2026-10-10 (teacher v2): a lane request, not a record
                    self._lane_message(fd)
                    continue
                c = self._by_fd.get(fd)
                if c is None or c not in self.live:
                    continue
                # a worker's message is one Connection.send_bytes of one byte: a 4-byte length header and the byte,
                # written at once (atomic in a pipe); read it whole without Connection's two reads
                try:
                    msg = os.read(fd, 5)
                except OSError:
                    msg = b''
                if msg in (_MSG0, _MSG1):
                    i = self.index[c]
                    idx.append(i)
                    slots.append(2 * i + (msg == _MSG1))
                else:   # 'x' (the worker ends), EOF or anything unexpected
                    self._drop(c)
            if not idx:
                break
            if first is None:
                first = time.perf_counter()
            if len(idx) >= want or time.perf_counter() - first >= max_wait or not self.live:
                break
            timeout = 0
        return np.array(idx, np.int64), np.array(slots, np.int64)

    def totals(self):
        return dict(zip(STATS, self.stats.sum(axis=0).tolist()))

    def poll(self, want, max_wait, idle=0.2):
        """The workers whose records are ready and the records' rows: waits up to `idle` s for the first, then until
        `want` are there or max_wait s have passed. ([worker], [row])."""
        idx, slots, waiting, first = [], [], set(self.live), None
        while len(idx) < want and waiting:
            left = idle if first is None else max_wait - (time.perf_counter() - first)
            if left <= 0:
                break
            ready = connection.wait(list(waiting), timeout=left)
            if not ready:
                break
            for c in ready:
                waiting.discard(c)
                try:
                    msg = c.recv_bytes()
                    if msg in (b'0', b'1'):
                        idx.append(self.index[c])
                        slots.append(2 * self.index[c] + (msg == b'1'))
                    else:
                        self._drop(c)
                except (EOFError, OSError):
                    self._drop(c)
            if first is None and idx:
                first = time.perf_counter()
        return idx, slots

    def teacher_records(self):
        """The teacher records written since the last call: (rows, meta [n, TEACH_META]); None without a teacher."""
        if self.teach_rows is None:
            return None
        rows, meta = [], []
        for i in range(self.cfg.workers):
            count = int(self.stats[i, ST['teach']])
            first = max(int(self.teach_seen[i]), count - TEACH_RING)
            for c in range(first, count):
                rows.append(self.teach_rows[i, c % TEACH_RING].copy())
                meta.append(self.teach_meta[i, c % TEACH_RING].copy())
            self.teach_seen[i] = count
        if not rows:
            return np.zeros((0,), ROW), np.zeros((0, TEACH_META), np.int32)
        return np.stack(rows), np.stack(meta)

    def choice_records(self):
        """The choice records written since the last call: (rows, meta [n, CHOICE_META]); None without branches."""
        if self.choice_rows is None:
            return None
        from .tok_branch import CHOICE_META, CHOICE_RING
        rows, meta = [], []
        for i in range(self.cfg.workers):
            count = int(self.stats[i, ST['choices']])
            first = max(int(self.choice_seen[i]), count - CHOICE_RING)
            for c in range(first, count):
                rows.append(self.choice_rows[i, c % CHOICE_RING].copy())
                meta.append(self.choice_meta[i, c % CHOICE_RING].copy())
            self.choice_seen[i] = count
        if not rows:
            return np.zeros((0,), ROW), np.zeros((0, CHOICE_META), np.float64)
        return np.stack(rows), np.stack(meta)

    def lab_records(self):
        """The lab's result records written since the last call (tok_lab RES vectors, [n, RES_F]); None without the
        lab."""
        if self.lab_res is None:
            return None
        from .tok_lab import LAB_RING, RES_F
        out = []
        for i in range(self.cfg.workers):
            count = int(self.stats[i, ST['lab_results']])
            first = max(int(self.lab_seen[i]), count - LAB_RING)
            for c in range(first, count):
                out.append(self.lab_res[i, c % LAB_RING].copy())
            self.lab_seen[i] = count
        return np.stack(out) if out else np.zeros((0, RES_F), np.float64)

    def reply(self, idx):
        for i in idx:
            try:   # what Connection.send_bytes(b'a') writes (header and byte in one write), without its layers
                os.write(self._fds[i], _REPLY)
            except OSError:
                pass

    def close(self, timeout=60.0):
        self.control[0] = 1
        end = time.time() + timeout
        while time.time() < end and any(p.is_alive() for p in self.procs) and self.live:
            idx, _ = self.poll(len(self.live), 0.05)
            self.reply(idx)
        for p in self.procs:
            p.join(max(0.1, end - time.time()))
            if p.is_alive():
                p.terminate()
        self.rows = self.actions = self.stats = self.control = self.teach_rows = self.teach_meta = None
        self.choice_rows = self.choice_meta = self.aux = None
        self.lab_jobs = self.lab_res = None
        self.lane_rows = self.lane_io = self.lane_out = self.main_value = self.v2_rows = self.v2_meta = None
        self.main_h = self.lane_h = self.v2_h = None
        for b in self.attached:
            b.close()
        for b in self.blocks:
            b.close()
            b.unlink()
        if self.png_file is not None:
            if os.environ.get('ABP_PNG_CACHE_FILE') == self.png_file:
                del os.environ['ABP_PNG_CACHE_FILE']
            try:
                os.unlink(self.png_file)
            except OSError:
                pass
