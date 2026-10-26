| GiB per update | 旧架构 (1->7) | 新C (1->7) | 新A (1->7) | 新C (9->20) | 新A (9->20) |
|---|---|---|---|---|---|
| games: archive (parked room entries) | +0.268 | +0.666 | +0.566 | +0.338 | +0.357 |
| games: other (unmapped, episodes, queues, root, template) | +0.839 | -0.269 | -0.090 | +0.384 | +0.272 |
| python: trainer (= actor0) | -0.004 | +0.097 | +0.104 | +0.207 | +0.053 |
| python: spawned by trainer (learner, actor1, 64 game bridges) | +0.003 | +0.457 | +0.403 | +0.168 | -0.016 |
| python: other spawned (3 source workers, ...) | -0.017 | +0.078 | +0.169 | +0.334 | +0.110 |
| compile + other | +0.000 | +0.000 | -0.000 | +0.000 | +0.000 |
| Pss sum | +1.089 | +1.030 | +1.152 | +1.430 | +0.776 |
| cgroup anon | +1.478 | +1.216 | +1.310 | +1.067 | +0.686 |
| cgroup shmem | +0.024 | +0.034 | +0.097 | +0.322 | +0.110 |
| coverage (Pss / anon+shmem) | 72% | 82% | 82% | 103% | 97% |
| archive procs | 215->278 | 24->136 | 38->136 | 166->319 | 172->331 |
| level at end (cgroup anon+shmem GiB) | 49.8 | 50.4 | 50.6 | 67.6 | 60.1 |
