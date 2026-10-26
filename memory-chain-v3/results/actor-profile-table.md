| | new rebuild pid 693 | new rebuild pid 87 | old graph pid 686 | old graph pid 70 |
|---|---|---|---|---|
| owner calls in window / all main calls of the process | 200 / 400 | 200 / 400 | 200 / 398 | 200 / 398 |
| other calls in window (lane, ...) | {} | {} | {} | {} |
| **latency per call (ms, median)** | 10.50 | 11.23 | 6.05 | 6.14 |
| latency quartiles | 9.68 / 10.50 / 14.94 | 9.64 / 11.23 / 13.80 | 5.45 / 6.05 / 6.64 | 5.62 / 6.14 / 6.76 |
| **host time in submit (ms, median)** | 4.34 | 4.60 | 2.43 | 2.56 |
| wait in result (ms, median) | 0.009 | 0.008 | 0.010 | 0.010 |
| **owner GPU time (ms/call, sum of h.* device)** | 2.71 | 3.05 | 2.80 | 2.90 |
| all kernels of the process per main call (ms) | 2.97 | 3.36 | 2.96 | 3.01 |
| h.component_keys CPU / GPU ms | 0.561 / 0.000 | 0.559 / 0.000 |  |  |
| h.component_replay CPU / GPU ms | 0.000 / 0.152 | 0.000 / 0.179 |  |  |
| h.core_replay CPU / GPU ms | 0.000 / 1.310 | 0.000 / 1.630 | 0.000 / 2.718 | 0.000 / 2.820 |
| h.d2h CPU / GPU ms | 0.000 / 0.002 | 0.000 / 0.002 | 0.000 / 0.002 | 0.000 / 0.002 |
| h.grow_refresh CPU / GPU ms | 0.156 / 0.000 | 0.147 / 0.000 |  |  |
| h.h2d_uniform CPU / GPU ms | 0.000 / 0.253 | 0.000 / 0.252 | 0.000 / 0.062 | 0.000 / 0.066 |
| h.rawhistory_append CPU / GPU ms | 0.747 / 0.000 | 0.683 / 0.000 |  |  |
| h.reset_prefixes CPU / GPU ms | 0.246 / 0.000 | 0.228 / 0.000 |  |  |
| h.stage_pack CPU / GPU ms | 0.195 / 0.000 | 0.207 / 0.000 | 0.284 / 0.000 | 0.292 / 0.000 |
| h.state_tstate CPU / GPU ms | 0.000 / 0.056 | 0.000 / 0.043 |  |  |
| h.tdest CPU / GPU ms | 0.000 / 0.245 | 0.000 / 0.225 | 0.000 / 0.014 | 0.000 / 0.015 |
| h.writer_close CPU / GPU ms | 0.000 / 0.689 | 0.000 / 0.715 |  |  |
| rows per call (mean) | 15.2 | 16.5 | 14.2 | 16.0 |
| row buckets r | [16, 32] | [16, 32] | [16, 32] | [16, 32] |
| entity buckets e (calls) | 16:192, 32:8 | 16:166, 32:34 | 16:57, 32:80, 40:57, 64:6 | 16:27, 32:162, 40:11 |
| graphs before / after window | 15 / 15 | 13 / 13 | 8 / 8 | 8 / 8 |
| dynamo counter changes | none | none | none | none |
| note: capacity (mean) | 256.00 | 512.00 |  |  |
| note: closed (mean) | 0.20 | 0.23 |  |  |
| note: component_buckets (mean) | 2.23 | 2.58 |  |  |
| note: component_rows (mean) | 40.40 | 50.00 |  |  |
| note: core_capture (mean) | 0.00 | 0.00 |  |  |
| note: len_max (mean) | 1908.81 | 4472.94 |  |  |
| note: len_mean (mean) | 734.58 | 1081.46 |  |  |
| note: stale (mean) | 0.00 | 0.00 |  |  |
| kernels other (ms per owner call, all streams) | 2.745 | 3.083 | 3.006 | 3.118 |
| kernels gemm (ms per owner call, all streams) | 1.633 | 1.789 | 1.591 | 1.562 |
| kernels triton fused (ms per owner call, all streams) | 1.012 | 1.286 | 0.336 | 0.319 |
| kernels copy/memset (ms per owner call, all streams) | 0.306 | 0.311 | 0.269 | 0.281 |
| kernels attention/softmax (ms per owner call, all streams) | 0.123 | 0.126 | 0.061 | 0.051 |
| kernels conv (ms per owner call, all streams) | 0.069 | 0.076 | 0.177 | 0.178 |
| kernels index/gather/scatter (ms per owner call, all streams) | 0.043 | 0.042 | 0.387 | 0.415 |
| kernels reduce/norm (ms per owner call, all streams) | 0.001 | 0.001 | 0.062 | 0.063 |
| **eager network, GPU ms per call (uncompiled, no graph)** | 22.92 | 19.27 | 31.40 | 31.82 |
| eager: Spatial CNN (patch, room, map, player) | 0.00 (0%) | 0.00 (0%) | 2.92 (9%) | 3.06 (10%) |
| eager: Spatial tokens (embeddings, entity, door, blocks, items, global) | 0.63 (3%) | 0.50 (3%) | 3.06 (10%) | 3.20 (10%) |
| eager: Temporal attention (old: tatt) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| eager: Reader input / layers / Writer | 0.02 (0%) | 0.02 (0%) | 0.00 (0%) | 0.00 (0%) |
| eager: c_t fusion and heads | 1.13 (5%) | 0.98 (5%) | 0.14 (0%) | 0.14 (0%) |
| eager: remainder (functional ops) | 21.15 (92%) | 17.77 (92%) | 25.28 (81%) | 25.42 (80%) |
