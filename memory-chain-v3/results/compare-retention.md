| GiB per update | 新C旧保留 (9->17) | 新A (9->17) | 新C新保留 (9->17) |
|---|---|---|---|
| games: archive (parked room entries) | +0.383 | +0.353 | +0.405 |
| games: other (unmapped, episodes, queues, root, template) | +0.262 | +0.263 | +0.083 |
| python: trainer (= actor0) | +0.148 | -0.017 | +0.091 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.218 | -0.019 | +0.337 |
| python: other spawned (3 source workers, ...) | +0.284 | +0.147 | +0.312 |
| compile + other | +0.000 | +0.000 | +0.000 |
| Pss sum | +1.296 | +0.727 | +1.228 |
| cgroup anon | +1.344 | +0.793 | +0.969 |
| cgroup shmem | +0.271 | +0.146 | +0.311 |
| coverage (Pss / anon+shmem) | 80% | 77% | 96% |
| archive procs | 166->294 | 172->289 | 159->292 |
| level at end (cgroup anon+shmem GiB) | 65.2 | 58.8 | 62.8 |
