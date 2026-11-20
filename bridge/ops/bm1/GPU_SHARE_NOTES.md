# BM1_GPU_SHARE_WITH -- shelved (2026-11-20)

Archive branch only. Two attempts (d737651, 5111cee) at letting one job (seq2's evaluation) start on a GPU that one of
our own running containers (seq2's learner, GPU 2 on dgene) already uses. Neither was deployed; exp/seq-v1 reverted
both. The coordinator stopped the work after Codex round 28: a safety check on a shared host, still with high-risk
findings after one fix round, one of them a regression on every Docker job's launch (the guards' resumes go through
it), bought only a few hours' earlier evaluation. The evaluation moved to lane b's own GPUs after seq2n's end.

## Codex round 27 (on d737651) -- deploy-blocking
1. Ownership not proven: an `aisaac-bench-*` name is not ours; `grep -F` matched a substring anywhere in the cgroup
   file (short ids, ids in unrelated path components); production accepted `BM1_PROC`, so the evidence could be
   replaced by an ordinary directory; nothing checked that nvidia-smi's pids and /proc share the host's namespace.
2. No exclusivity between the check and the start (also in the original check): another tenant can take the card
   after the last query; occupancy that nvidia-smi does not list as compute processes is only bounded by memory.
3. List parsing failures could read as an empty list (`for ... in $(... | tr ...)`); pids squeezed (`10 1` -> `101`).
Lower: tests asserted exit codes, not reasons, and fed every card the same data; `docker inspect` had no timeout; a
log write failure could end the refusal before `exit 4`.

## Codex round 28 (on 5111cee) -- still "do not deploy"
1. The registry is not trusted evidence: if the registry and the cidfile directory sit inside the `/bench` mount that
   job containers write, a container can register itself; `mktemp -u` reserves nothing; "neither marker file exists"
   does not prove the host's PID namespace.
2. The memory reading's `sed` exit status is not checked.
3. Regression: every Docker job (the switch off too) gained `--cidfile` and a background registrar -- a normal resume
   of a training guard could fail on it.

## If this is ever done again
Start from: a static allow-list file of (container name, full id, GPU) in a host directory that no container mounts,
written by hand with a bm1.log line; run.sh untouched; the default path byte-for-byte unchanged; the host PID namespace
checked directly (e.g. /proc/self/ns/pid against PID 1's); every external call's exit status checked; a lease, if one
ever exists for every tenant of the host, covering check and start.
