| | base cuda:2 | base cuda:0 | base cuda:3 | merge cuda:2 | merge cuda:0 | merge cuda:3 | fused cuda:2 | fused cuda:0 | fused cuda:3 | both cuda:2 | both cuda:3 | both cuda:0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VJP wall s | 13.95 | 16.23 | 17.04 | 13.14 | 15.57 | 16.47 | 17.56 | 18.82 | 16.92 | 12.78 | 16.21 | 17.34 |
| frames (encode chunks) | 13830 | 16753 | 21910 | 19121 | 16860 | 15583 | 14897 | 17024 | 18131 | 13098 | 14925 | 22207 |
| packets / encode chunks / writes | 274 / 106 / 168 | 342 / 151 / 191 | 324 / 148 / 176 | 295 / 145 / 150 | 311 / 138 / 173 | 315 / 124 / 191 | 327 / 123 / 204 | 346 / 130 / 216 | 327 / 141 / 186 | 283 / 127 / 156 | 308 / 131 / 177 | 376 / 203 / 173 |
| backward calls (episode groups) | 71 | 76 | 79 | 71 | 79 | 77 | 65 | 70 | 76 | 64 | 71 | 81 |
| frames per chunk min / median / mean / max | 1 / 128 / 130 / 273 | 1 / 62 / 111 / 262 | 1 / 202 / 148 / 272 | 1 / 158 / 132 / 272 | 1 / 88 / 122 / 269 | 1 / 121 / 126 / 268 | 1 / 88 / 121 / 268 | 1 / 110 / 131 / 267 | 1 / 134 / 129 / 267 | 1 / 53 / 103 / 270 | 1 / 58 / 114 / 269 | 1 / 57 / 109 / 269 |
| frames per backward call | 195 | 220 | 277 | 269 | 213 | 202 | 229 | 243 | 239 | 205 | 210 | 274 |
| kernels | 127250 | 143410 | 134053 | 109639 | 132884 | 142030 | 146302 | 153306 | 143879 | 109686 | 133351 | 143450 |
| kernels per 1k frames | 9201 | 8560 | 6118 | 5734 | 7882 | 9114 | 9821 | 9005 | 7936 | 8374 | 8935 | 6460 |
| kernel us mean / median / p90 | 4.9 / 2.3 / 9.6 | 5.1 / 2.4 / 10.0 | 5.4 / 2.3 / 10.0 | 5.4 / 2.4 / 10.6 | 5.3 / 2.4 / 10.5 | 5.1 / 2.3 / 9.6 | 5.4 / 2.2 / 11.0 | 5.7 / 2.3 / 11.7 | 5.4 / 2.2 / 10.5 | 5.2 / 2.3 / 10.1 | 5.3 / 2.2 / 10.3 | 5.1 / 2.2 / 10.0 |
| kernel s total | 0.62 | 0.73 | 0.72 | 0.59 | 0.70 | 0.73 | 0.79 | 0.87 | 0.77 | 0.57 | 0.71 | 0.72 |
| copies (n / s) | 10330 / 0.02 | 12231 / 0.02 | 11976 / 0.02 | 9369 / 0.01 | 11486 / 0.02 | 12222 / 0.02 | 13636 / 0.02 | 14494 / 0.02 | 13176 / 0.02 | 9948 / 0.01 | 12118 / 0.02 | 12434 / 0.02 |
| GPU window s / busy s | 8.98 / 0.64 | 9.77 / 0.75 | 9.87 / 0.74 | 8.14 / 0.60 | 9.29 / 0.72 | 9.28 / 0.74 | 10.00 / 0.81 | 10.80 / 0.89 | 9.78 / 0.79 | 7.87 / 0.58 | 9.62 / 0.73 | 10.04 / 0.74 |
| GPU idle share of the window | 92.9% | 92.3% | 92.5% | 92.6% | 92.2% | 92.0% | 91.9% | 91.7% | 91.9% | 92.6% | 92.4% | 92.6% |
| autograd nodes (per 1k frames) | 57887 (4186) | 68098 (4065) | 61775 (2819) | 38720 (2025) | 46424 (2753) | 49882 (3201) | 59714 (4008) | 63158 (3710) | 58003 (3199) | 31984 (2442) | 38733 (2595) | 41253 (1858) |

**base cuda:2: top kernels (count, s)**

- 27710 × 0.052 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 956 × 0.051 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 11446 × 0.036 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 788 × 0.032 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 3675 × 0.032 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3219 × 0.022 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 788 × 0.021 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 904 × 0.017 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 12456 × 0.017 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 644 × 0.015 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 13710, evaluate_function: ImmutableSourceCastBackward 10212, evaluate_function: SelectBackward0 5593, evaluate_function: EmbeddingBackward0 3743, evaluate_function: ViewBackward0 3186, evaluate_function: ToCopyBackward0 2277, evaluate_function: AddBackward0 2086, evaluate_function: IndexSelectBackward0 2033, evaluate_function: CatBackward0 1970, evaluate_function: SliceBackward0 1970, evaluate_function: CompiledFunctionBackward 1663, evaluate_function: TBackward0 1489

**base cuda:0: top kernels (count, s)**

- 1059 × 0.061 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 30966 × 0.060 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 12953 × 0.042 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 4032 × 0.040 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 868 × 0.038 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 3559 × 0.026 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 868 × 0.021 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 1032 × 0.020 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 14231 × 0.020 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 800 × 0.019 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 18242, evaluate_function: ImmutableSourceCastBackward 11564, evaluate_function: SelectBackward0 6711, evaluate_function: EmbeddingBackward0 4123, evaluate_function: ViewBackward0 3600, evaluate_function: ToCopyBackward0 2579, evaluate_function: AddBackward0 2306, evaluate_function: IndexSelectBackward0 2241, evaluate_function: CatBackward0 2170, evaluate_function: SliceBackward0 2170, evaluate_function: CompiledFunctionBackward 1875, evaluate_function: TBackward0 1658

**base cuda:3: top kernels (count, s)**

- 984 × 0.065 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 29447 × 0.055 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 808 × 0.042 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 3727 × 0.039 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 11770 × 0.037 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 3338 × 0.023 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 908 × 0.022 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 808 × 0.022 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 872 × 0.021 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 13783 × 0.019 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 15157, evaluate_function: ImmutableSourceCastBackward 10486, evaluate_function: SelectBackward0 6765, evaluate_function: EmbeddingBackward0 3838, evaluate_function: ViewBackward0 3212, evaluate_function: ToCopyBackward0 2355, evaluate_function: AddBackward0 2200, evaluate_function: IndexSelectBackward0 2084, evaluate_function: CatBackward0 2020, evaluate_function: SliceBackward0 2020, evaluate_function: CompiledFunctionBackward 1705, evaluate_function: TBackward0 1538

**merge cuda:2: top kernels (count, s)**

- 814 × 0.049 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 24552 × 0.047 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 664 × 0.032 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 9750 × 0.032 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 3075 × 0.030 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 2731 × 0.021 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 664 × 0.020 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 696 × 0.016 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 10993 × 0.015 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 728 × 0.015 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: ImmutableSourceCastBackward 8668, evaluate_function: SelectBackward0 5201, evaluate_function: EmbeddingBackward0 3154, evaluate_function: ViewBackward0 2764, evaluate_function: ToCopyBackward0 1973, evaluate_function: AddBackward0 1794, evaluate_function: IndexSelectBackward0 1720, evaluate_function: CatBackward0 1660, evaluate_function: SliceBackward0 1660, evaluate_function: CompiledFunctionBackward 1407, evaluate_function: TBackward0 1280, evaluate_function: AddmmBackward0 1114

**merge cuda:0: top kernels (count, s)**

- 981 × 0.060 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 29757 × 0.057 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 11855 × 0.039 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 808 × 0.038 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 3736 × 0.036 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3315 × 0.026 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 808 × 0.021 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 968 × 0.020 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 13343 × 0.019 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 780 × 0.018 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`

autograd node kinds: evaluate_function: ImmutableSourceCastBackward 10586, evaluate_function: SelectBackward0 6316, evaluate_function: EmbeddingBackward0 3838, evaluate_function: ViewBackward0 3258, evaluate_function: ToCopyBackward0 2350, evaluate_function: AddBackward0 2150, evaluate_function: IndexSelectBackward0 2071, evaluate_function: CatBackward0 2020, evaluate_function: SliceBackward0 2020, evaluate_function: CompiledFunctionBackward 1720, evaluate_function: TBackward0 1529, evaluate_function: AddmmBackward0 1341

**merge cuda:3: top kernels (count, s)**

- 1051 × 0.062 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 31724 × 0.059 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 860 × 0.040 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 12713 × 0.039 s  `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorB`
- 3983 × 0.038 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3536 × 0.023 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 860 × 0.022 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 864 × 0.021 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 936 × 0.020 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 14234 × 0.019 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `

autograd node kinds: evaluate_function: ImmutableSourceCastBackward 11328, evaluate_function: SelectBackward0 6780, evaluate_function: EmbeddingBackward0 4085, evaluate_function: ViewBackward0 3552, evaluate_function: ToCopyBackward0 2549, evaluate_function: AddBackward0 2308, evaluate_function: IndexSelectBackward0 2228, evaluate_function: CatBackward0 2150, evaluate_function: SliceBackward0 2150, evaluate_function: CompiledFunctionBackward 1837, evaluate_function: TBackward0 1648, evaluate_function: AddmmBackward0 1442

**fused cuda:2: top kernels (count, s)**

- 35298 × 0.070 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 1160 × 0.069 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 956 × 0.044 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 4424 × 0.040 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 956 × 0.028 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 3921 × 0.028 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 936 × 0.022 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 1052 × 0.022 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 15851 × 0.021 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 688 × 0.017 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 17273, evaluate_function: SelectBackward0 7560, evaluate_function: EmbeddingBackward0 4541, evaluate_function: ViewBackward0 3822, evaluate_function: ToCopyBackward0 2750, evaluate_function: AddBackward0 2554, evaluate_function: IndexSelectBackward0 2473, evaluate_function: CatBackward0 2390, evaluate_function: SliceBackward0 2390, evaluate_function: CompiledFunctionBackward 2010, evaluate_function: TBackward0 1807, evaluate_function: AddmmBackward0 1573

**fused cuda:0: top kernels (count, s)**

- 1208 × 0.076 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 36914 × 0.075 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 992 × 0.048 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 4576 × 0.046 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 4079 × 0.030 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 992 × 0.027 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 1048 × 0.027 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 1012 × 0.023 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 16647 × 0.023 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 660 × 0.017 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 18631, evaluate_function: SelectBackward0 8056, evaluate_function: EmbeddingBackward0 4712, evaluate_function: ViewBackward0 4004, evaluate_function: ToCopyBackward0 2888, evaluate_function: AddBackward0 2670, evaluate_function: IndexSelectBackward0 2585, evaluate_function: CatBackward0 2480, evaluate_function: SliceBackward0 2480, evaluate_function: CompiledFunctionBackward 2096, evaluate_function: TBackward0 1888, evaluate_function: AddmmBackward0 1634

**fused cuda:3: top kernels (count, s)**

- 34786 × 0.069 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 1146 × 0.068 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 960 × 0.045 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 4445 × 0.041 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3918 × 0.025 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 960 × 0.024 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 940 × 0.024 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 884 × 0.021 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 15682 × 0.021 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 564 × 0.016 s  `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 16703, evaluate_function: SelectBackward0 7312, evaluate_function: EmbeddingBackward0 4560, evaluate_function: ViewBackward0 3572, evaluate_function: ToCopyBackward0 2595, evaluate_function: AddBackward0 2528, evaluate_function: IndexSelectBackward0 2445, evaluate_function: CatBackward0 2400, evaluate_function: SliceBackward0 2400, evaluate_function: CompiledFunctionBackward 1967, evaluate_function: TBackward0 1758, evaluate_function: AddmmBackward0 1522

**both cuda:2: top kernels (count, s)**

- 26072 × 0.053 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 868 × 0.046 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 712 × 0.030 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 3298 × 0.027 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 2916 × 0.020 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 712 × 0.018 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 11560 × 0.016 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 768 × 0.014 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 612 × 0.014 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 5010 × 0.011 s  `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_co`

autograd node kinds: evaluate_function: SelectBackward0 5356, evaluate_function: EmbeddingBackward0 3382, evaluate_function: ViewBackward0 2940, evaluate_function: ToCopyBackward0 2111, evaluate_function: AddBackward0 1892, evaluate_function: IndexSelectBackward0 1848, evaluate_function: CatBackward0 1780, evaluate_function: SliceBackward0 1780, evaluate_function: CompiledFunctionBackward 1535, evaluate_function: TBackward0 1358, evaluate_function: AddmmBackward0 1202, evaluate_function: TransposeBackward0 1092

**both cuda:3: top kernels (count, s)**

- 32104 × 0.064 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 1065 × 0.063 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 888 × 0.041 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 4108 × 0.038 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3626 × 0.023 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 888 × 0.022 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 852 × 0.022 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 864 × 0.020 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 14372 × 0.019 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 496 × 0.014 s  `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nt`

autograd node kinds: evaluate_function: SelectBackward0 6626, evaluate_function: EmbeddingBackward0 4218, evaluate_function: ViewBackward0 3382, evaluate_function: ToCopyBackward0 2441, evaluate_function: AddBackward0 2344, evaluate_function: IndexSelectBackward0 2279, evaluate_function: CatBackward0 2220, evaluate_function: SliceBackward0 2220, evaluate_function: CompiledFunctionBackward 1827, evaluate_function: TBackward0 1641, evaluate_function: AddmmBackward0 1427, evaluate_function: UnsqueezeBackward0 1242

**both cuda:0: top kernels (count, s)**

- 34701 × 0.069 s  `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<cha`
- 1157 × 0.063 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 984 × 0.040 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 4584 × 0.037 s  `void at::native::(anonymous namespace)::embedding_backward_feature_kernel<float, float, long>(long c`
- 3983 × 0.027 s  `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float,`
- 15568 × 0.021 s  `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, `
- 984 × 0.021 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_8`
- 900 × 0.019 s  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_nt`
- 820 × 0.018 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_stages_32x1_tn`
- 6344 × 0.014 s  `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_co`

autograd node kinds: evaluate_function: SelectBackward0 6949, evaluate_function: EmbeddingBackward0 4674, evaluate_function: ViewBackward0 3482, evaluate_function: AddBackward0 2522, evaluate_function: ToCopyBackward0 2507, evaluate_function: IndexSelectBackward0 2480, evaluate_function: CatBackward0 2460, evaluate_function: SliceBackward0 2460, evaluate_function: CompiledFunctionBackward 1961, evaluate_function: TBackward0 1749, evaluate_function: AddmmBackward0 1541, evaluate_function: UnsqueezeBackward0 1330
