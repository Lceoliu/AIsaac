| GiB per update | 旧架构 (1->7) | C_规则1-4 (9->20) | C_规则5 (9->20) | A (9->20) |
|---|---|---|---|---|
| games: archive (parked room entries) | +0.268 | +0.261 | +0.363 | +0.357 |
| games: other (unmapped, episodes, queues, root, template) | +0.839 | +0.362 | +0.077 | +0.272 |
| python: trainer (= actor0) | -0.004 | +0.096 | +0.018 | +0.053 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.003 | +0.044 | +0.104 | -0.016 |
| python: other spawned (3 source workers, ...) | -0.017 | +0.173 | +0.195 | +0.110 |
| compile + other | +0.000 | +0.000 | +0.000 | +0.000 |
| Pss sum | +1.089 | +0.936 | +0.757 | +0.776 |
| cgroup anon | +1.478 | +0.795 | +0.785 | +0.686 |
| cgroup shmem | +0.024 | +0.166 | +0.200 | +0.110 |
| coverage (Pss / anon+shmem) | 72% | 97% | 77% | 97% |
| archive procs | 215->278 | 173->290 | 173->318 | 172->331 |
| level at end (cgroup anon+shmem GiB) | 49.8 | 64.9 | 61.9 | 60.1 |
