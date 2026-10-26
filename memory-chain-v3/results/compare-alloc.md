| GiB per update | C旧保留 (9->20) | A (9->20) | C新保留_glibc (9->20) | C新保留_jemalloc (9->20) |
|---|---|---|---|---|
| games: archive (parked room entries) | +0.338 | +0.357 | +0.261 | +0.327 |
| games: other (unmapped, episodes, queues, root, template) | +0.384 | +0.272 | +0.362 | +0.415 |
| python: trainer (= actor0) | +0.207 | +0.053 | +0.096 | +0.135 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.168 | -0.016 | +0.044 | +0.151 |
| python: other spawned (3 source workers, ...) | +0.334 | +0.110 | +0.173 | +0.242 |
| compile + other | +0.000 | +0.000 | +0.000 | +0.000 |
| Pss sum | +1.430 | +0.776 | +0.936 | +1.271 |
| cgroup anon | +1.067 | +0.686 | +0.795 | +0.904 |
| cgroup shmem | +0.322 | +0.110 | +0.166 | +0.225 |
| coverage (Pss / anon+shmem) | 103% | 97% | 97% | 112% |
| archive procs | 166->319 | 172->331 | 173->290 | 145->287 |
| level at end (cgroup anon+shmem GiB) | 67.6 | 60.1 | 64.9 | 65.8 |
