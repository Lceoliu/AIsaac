| | replay cuda:3 | replay cuda:2 | replay cuda:0 |
|---|---|---|---|
| VJP wall s | 5.25 | 9.13 | 12.19 |
| frames (encode chunks) | 14755 | 13622 | 31731 |
| packets / encode chunks / writes | 180 / 99 / 81 | 261 / 98 / 163 | 461 / 198 / 263 |
| backward calls (episode groups) | 64 | 57 | 91 |
| frames per chunk min / median / mean / max | 1 / 158 / 149 / 513 | 1 / 104 / 139 / 512 | 1 / 138 / 160 / 524 |
| frames per backward call | 231 | 239 | 349 |
| kernels | 3754 | 8414 | 11565 |
| kernels per 1k frames | 254 | 618 | 364 |
| kernel us mean / median / p90 | 48.9 / 3.0 / 64.6 | 44.8 / 2.7 / 47.7 | 58.7 / 2.9 / 69.3 |
| kernel s total | 0.18 | 0.38 | 0.68 |
| copies (n / s) | 1392 / 0.08 | 2620 / 0.15 | 4263 / 0.27 |
| GPU window s / busy s | 4.79 / 0.26 | 8.50 / 0.53 | 11.42 / 0.95 |
| GPU idle share of the window | 94.6% | 93.7% | 91.7% |
| autograd nodes (per 1k frames) | 907 (61) | 2195 (161) | 2843 (90) |

**replay cuda:3: top kernels (count, s)**

- 14 × 0.029 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 14 × 0.016 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 18 × 0.014 s  `triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_6`
- 15 × 0.010 s  `void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true, at::native::`
- 8 × 0.007 s  `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro`
- 21 × 0.007 s  `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_nn`
- 12 × 0.006 s  `triton_poi_fused_gelu_gelu_backward_0`
- 14 × 0.006 s  `void pytorch_flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4,`
- 12 × 0.006 s  `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_nt`
- 8 × 0.005 s  `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nt`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 337, evaluate_function: ImmutableSourceCastBackward 166, evaluate_function: EmbeddingBackward0 57, evaluate_function: ViewBackward0 38, evaluate_function: AddBackward0 32, evaluate_function: CatBackward0 30, evaluate_function: SliceBackward0 30, evaluate_function: ToCopyBackward0 29, evaluate_function: IndexSelectBackward0 27, evaluate_function: TBackward0 21, evaluate_function: SelectBackward0 19, evaluate_function: CompiledFunctionBackward 18

**replay cuda:2: top kernels (count, s)**

- 34 × 0.055 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 42 × 0.040 s  `triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_6`
- 34 × 0.031 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 35 × 0.021 s  `void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true, at::native::`
- 28 × 0.013 s  `triton_poi_fused_gelu_gelu_backward_0`
- 20 × 0.012 s  `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro`
- 34 × 0.011 s  `void pytorch_flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4,`
- 7 × 0.011 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13`
- 34 × 0.010 s  `void pytorch_flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4, tru`
- 30 × 0.009 s  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_relu_f2f_tn`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 797, evaluate_function: ImmutableSourceCastBackward 390, evaluate_function: EmbeddingBackward0 133, evaluate_function: ViewBackward0 102, evaluate_function: AddBackward0 80, evaluate_function: ToCopyBackward0 77, evaluate_function: CatBackward0 70, evaluate_function: SliceBackward0 70, evaluate_function: IndexSelectBackward0 63, evaluate_function: TBackward0 53, evaluate_function: SelectBackward0 47, evaluate_function: TransposeBackward0 42

**replay cuda:0: top kernels (count, s)**

- 44 × 0.105 s  `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<64, 64, 128`
- 54 × 0.067 s  `triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_6`
- 44 × 0.061 s  `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass:`
- 45 × 0.040 s  `void at::native::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true, at::native::`
- 32 × 0.024 s  `void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro`
- 36 × 0.023 s  `triton_poi_fused_gelu_gelu_backward_0`
- 44 × 0.020 s  `void pytorch_flash::flash_bwd_dot_do_o_kernel<true, Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4,`
- 9 × 0.019 s  `triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_13`
- 32 × 0.018 s  `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nt`
- 44 × 0.017 s  `void pytorch_flash::flash_bwd_convert_dq_kernel<Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4, tru`

autograd node kinds: evaluate_function: torch::autograd::AccumulateGrad 1027, evaluate_function: ImmutableSourceCastBackward 502, evaluate_function: EmbeddingBackward0 171, evaluate_function: ViewBackward0 134, evaluate_function: AddBackward0 104, evaluate_function: ToCopyBackward0 101, evaluate_function: CatBackward0 90, evaluate_function: SliceBackward0 90, evaluate_function: IndexSelectBackward0 81, evaluate_function: TBackward0 69, evaluate_function: SelectBackward0 61, evaluate_function: TransposeBackward0 56
