| | ck cuda:3 | ck cuda:0 | ck cuda:2 | full cuda:0 | full cuda:3 | full cuda:2 |
|---|---|---|---|---|---|---|
| VJP wall s | 12.12 | 16.73 | 17.68 | 12.83 | 12.74 | 16.31 |
| frames (encode chunks) | 11453 | 17679 | 21813 | 16287 | 17878 | 18824 |
| packets / encode chunks / writes | 209 / 94 / 115 | 308 / 142 / 166 | 313 / 148 / 165 | 288 / 120 / 168 | 294 / 136 / 158 | 347 / 149 / 198 |
| backward calls (episode groups) | 56 | 62 | 84 | 57 | 66 | 83 |
| frames per chunk min / median / mean / max | 1 / 92 / 122 / 264 | 1 / 62 / 124 / 267 | 1 / 200 / 147 / 264 | 1 / 168 / 136 / 266 | 1 / 169 / 131 / 270 | 1 / 108 / 126 / 275 |
| frames per backward call | 205 | 285 | 260 | 286 | 271 | 227 |
| kernels | 91755 | 128302 | 137711 | 112236 | 112080 | 134368 |
| kernels per 1k frames | 8011 | 7257 | 6313 | 6891 | 6269 | 7138 |
| kernel us mean / median / p90 | 5.0 / 2.3 / 9.1 | 5.3 / 2.4 / 10.3 | 5.2 / 2.3 / 10.2 | 5.1 / 2.3 / 9.2 | 4.9 / 2.2 / 8.4 | 4.9 / 2.2 / 8.7 |
| kernel s total | 0.46 | 0.68 | 0.72 | 0.57 | 0.55 | 0.65 |
| copies (n / s) | 7591 / 0.01 | 11382 / 0.02 | 12310 / 0.02 | 10534 / 0.02 | 10906 / 0.01 | 15175 / 0.12 |
| GPU window s / busy s | 7.01 / 0.47 | 9.89 / 0.70 | 10.59 / 0.74 | 6.85 / 0.59 | 6.65 / 0.57 | 8.31 / 0.77 |
| GPU idle share of the window | 93.3% | 92.9% | 93.0% | 91.4% | 91.5% | 90.7% |
| autograd nodes (per 1k frames) | 42202 (3685) | 59270 (3353) | 63935 (2931) | 56030 (3440) | 57753 (3230) | 71094 (3777) |

**ck cuda:3: top kernels (count, s)**

- 19713 × 0.037 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 679 × 0.037 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 8213 × 0.026 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 564 × 0.024 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 2608 × 0.023 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 2303 × 0.015 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 564 × 0.013 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 492 × 0.012 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 8977 × 0.012 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 564 × 0.012 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 10753, evaluate_function: ImmutableSourceCastBackward 7356, evaluate_function: SelectBackward0 4023, evaluate_function: EmbeddingBackward0 2679, evaluate_function: ViewBackward0 2216, evaluate_function: ToCopyBackward0 1602, evaluate_function: AddBackward0 1478, evaluate_function: IndexSelectBackward0 1454, evaluate_function: CatBackward0 1410, evaluate_function: SliceBackward0 1410, evaluate_function: CompiledFunctionBackward 1198, evaluate_function: TBackward0 1050

**ck cuda:0: top kernels (count, s)**

- 942 × 0.059 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 27793 × 0.054 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 776 × 0.037 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 11328 × 0.037 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 3555 × 0.032 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3186 × 0.023 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 776 × 0.021 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 804 × 0.020 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 12811 × 0.018 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 764 × 0.017 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 15057, evaluate_function: ImmutableSourceCastBackward 10110, evaluate_function: SelectBackward0 6051, evaluate_function: EmbeddingBackward0 3686, evaluate_function: ViewBackward0 3100, evaluate_function: ToCopyBackward0 2248, evaluate_function: AddBackward0 2078, evaluate_function: IndexSelectBackward0 2025, evaluate_function: CatBackward0 1940, evaluate_function: SliceBackward0 1940, evaluate_function: CompiledFunctionBackward 1644, evaluate_function: TBackward0 1468

**ck cuda:2: top kernels (count, s)**

- 1001 × 0.063 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 30040 × 0.056 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 836 × 0.040 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 12093 × 0.039 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 3840 × 0.034 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3429 × 0.026 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 836 × 0.025 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 924 × 0.020 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 14197 × 0.019 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 784 × 0.018 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 16653, evaluate_function: ImmutableSourceCastBackward 10850, evaluate_function: SelectBackward0 6887, evaluate_function: EmbeddingBackward0 3971, evaluate_function: ViewBackward0 3108, evaluate_function: ToCopyBackward0 2339, evaluate_function: AddBackward0 2230, evaluate_function: IndexSelectBackward0 2127, evaluate_function: CatBackward0 2090, evaluate_function: SliceBackward0 2090, evaluate_function: CompiledFunctionBackward 1769, evaluate_function: TBackward0 1540

**full cuda:0: top kernels (count, s)**

- 932 × 0.056 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 27206 × 0.054 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 11014 × 0.037 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 3537 × 0.035 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 1146 × 0.030 s  `triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_6`
- 3133 × 0.024 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 12560 × 0.018 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 832 × 0.018 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 5442 × 0.013 s  `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_co`
- 468 × 0.013 s  `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 13182, evaluate_function: ImmutableSourceCastBackward 9792, evaluate_function: SelectBackward0 5901, evaluate_function: EmbeddingBackward0 3629, evaluate_function: ViewBackward0 3154, evaluate_function: ToCopyBackward0 2219, evaluate_function: AddBackward0 2036, evaluate_function: IndexSelectBackward0 1989, evaluate_function: CatBackward0 1910, evaluate_function: SliceBackward0 1910, evaluate_function: TBackward0 1459, evaluate_function: AddmmBackward0 1287

**full cuda:3: top kernels (count, s)**

- 914 × 0.058 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 26963 × 0.050 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 3478 × 0.037 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 11006 × 0.034 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 1134 × 0.021 s  `triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_6`
- 3104 × 0.020 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 772 × 0.019 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 12632 × 0.017 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 596 × 0.016 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`
- 440 × 0.013 s  `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 15320, evaluate_function: ImmutableSourceCastBackward 9838, evaluate_function: SelectBackward0 6033, evaluate_function: EmbeddingBackward0 3591, evaluate_function: ViewBackward0 2946, evaluate_function: ToCopyBackward0 2168, evaluate_function: AddBackward0 2028, evaluate_function: IndexSelectBackward0 1942, evaluate_function: CatBackward0 1890, evaluate_function: SliceBackward0 1890, evaluate_function: TBackward0 1419, evaluate_function: AddmmBackward0 1219

**full cuda:2: top kernels (count, s)**

- 1070 × 0.066 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 32391 × 0.059 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 4039 × 0.041 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 13310 × 0.041 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 1308 × 0.033 s  `triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_6`
- 3601 × 0.025 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 1020 × 0.022 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 15487 × 0.020 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 572 × 0.015 s  `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nt`
- 628 × 0.014 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 19752, evaluate_function: ImmutableSourceCastBackward 11884, evaluate_function: SelectBackward0 7930, evaluate_function: EmbeddingBackward0 4142, evaluate_function: ViewBackward0 3616, evaluate_function: ToCopyBackward0 2667, evaluate_function: AddBackward0 2372, evaluate_function: IndexSelectBackward0 2255, evaluate_function: CatBackward0 2180, evaluate_function: SliceBackward0 2180, evaluate_function: TBackward0 1684, evaluate_function: AddmmBackward0 1452
