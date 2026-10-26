# 计时链提交 ↔ 主线提交

长测和计时用的代码标签 `v3-<hash>` 指计时链（分支 exp/memory-chain-v3-timing，源自候选仓库的 a-chain）或 crn 基线（exp/memory-chain-v3-crnbase）上的提交；主线是 exp/memory-chain-v3（候选仓库 master）。按提交说明对应；"—" 表示该提交只在这一条链上（计时链从 d94f666 起分叉，跳过了主线上的 212388c 等，后来又补拣）。

| 计时链 | 主线 | 说明 |
|---|---|---|
| 64ae451 | 64ae451 | v2 candidate as received |
| 9388647 | 9388647 | Bounded source gradient horizon (--memory-horizon, default 2048) |
| 0c8be2e | 0c8be2e | Pin chunks consumed by live Writer graphs; add test_source_horizon |
| 265717c | 265717c | Add HorizonAttention timing bench (native SDPA autograd vs masked) |
| bfd3656 | bfd3656 | Bounded-retention test; workers report their own graph frames |
| 05e7cde | 05e7cde | Opt-in host-memory accounting (ISAAC_MEMORY_ACCOUNT=1) |
| 83cf114 | 83cf114 | Record retention: reference ancestor frames, release after consumption, byte budget |
| 104176b | 104176b | RecordLedger: keep only live histories as donors |
| 1fe66ab | 1fe66ab | Diagnostics: live-object census, actor/source accounting, game env strip |
| 8cae7ea | 8cae7ea | Rule 5: release raw bytes of encoded, consumed frames |
| 37d8a6e | 37d8a6e | Rule 5: release the learner's frames only before the latest decision |
| aa3d84e | aa3d84e | Rule 5 D2a: keep ended episodes' source values for a grace period |
| e5ce72e | e5ce72e | Rule 5: share_prefix skips released donor blocks' rows; value pass reads cached sources |
| 772addc | 772addc | Timing only: split u_recompute into rc_prep / rc_sources / recompute |
| d85fae3 | d85fae3 | H=0: episodes with no future loss keep values only; log grace ages |
| 8720882 | 8720882 | Actor profiler (ISAAC_ACTOR_TORCH_PROFILE, off by default) |
| f2a423e | f2a423e | Source-card memory breakdown (ISAAC_SOURCE_MEMORY_BREAKDOWN=1, off by default) |
| f4911b1 | f4911b1 | Actor profiler: one window per process, at the process's AT-th main call |
| 61a2ea3 | 61a2ea3 | test_actor_profile_cuda: a fresh process window per actor kind |
| d94f666 | d94f666 | Actor profiler: timers mode (perf_counter sections, no torch.profiler) |
| 805868e | e29bfba | Actor timers: thread CPU time per section, the process's threads |
| fe05db2 | dbf0dfe | Actor A1: temporal state writes inside the core graph; no zero K/V copies |
| 3c100d5 | 60ecccd | Actor A2: memory-block close as a captured Writer graph per closed-count bucket |
| 9cc883d | 04c6b60 | Actor A3: one owned copy of a call's records, history rows as its views |
| 0c4b059 | 2ef8ed8 | Actor A4: component payload keys through a void view's tolist (same bytes) |
| 2601f32 | 42a61e8 | Actor A1b: tdest rows through the pinned control block |
| 726c14a | 1b570e0 | Fix: common random numbers only for branch records in the rebuild actors |
| 14a7405 | 729f5cd | Revert "Actor A3: one owned copy of a call's records, history rows as its views" |
| 3b492fb | ce909b3 | Source release accounting: retain_histories skips while a VJP is pending, frames released, VJP-to-retain lag |
| 2b6bc9d | ee8a98c | Release source graphs while an interval's VJP is pending |
| da07983 | 212388c | Saved-tensor census by module and shape (ISAAC_SOURCE_SAVED_PROFILE=N), per-worker graph frames |
| 010a06b | 17ecfa5 | Fix: the pending VJP's chunks were not protected from the horizon detach |
| c037218 | 8357752 | Timeline lines (ISAAC_TIMELINE=1, measurement only): learner parts, source events, actors' collect |
| 7d9c227 | 3a1c189 | Opt-in Spatial layer checkpointing in the source workers (--source-layer-checkpoint 1, default 0) |
| 419fce9 | 8f8a0dc | Scheduling: --source-dispatch-at-close 1 dispatches a closed source VJP when the update closes it |
| b657026 | 6b5f8e2 | Accounting: why graph frames are held (source_hold_*), record waits (record_*_wait_*) |
| 208911e | 32263d4 | Retention by reachable frame intervals (option d); multi-label hold accounting |
| 1727f71 | fc0ad4f | Fix: a waiting record's own query frame stays reachable; record the first non-zero dropped gradient |
| 187ede3 | c8f356f | Fix: in-flight branches keep their ancestors' reachable graphs [length - H, shared) |
| 019dda7 | 7ff027e | Accounting: the first dropped gradient's learner-side context (ISAAC_LINEAGE_SOURCES=1) |
| 9086793 | 1c5c374 | Accounting: per-episode first/last in-flight report per holder, which retain call detached |
| 258f97a | 753a47a | Fix: merge_jobs keeps both actor processes' in-flight lineage |
| d345da5 | 8ce8a58 | Audit: count gradient lost through Writer blocks built on released chunks; grace-period late-record test |

crn 基线（exp/memory-chain-v3-crnbase）：

| 提交 | 说明 |
|---|---|
| 805868e | Actor timers: thread CPU time per section, the process's threads |
| afd843a | Fix (on the timing base): common random numbers only for branch records (= 1b570e0) |
