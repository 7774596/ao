// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// Row-wise abs-max fused with * inv_max (FP32), for FP8 scale before
// sparse24_sm90_sparsify. Matches:
//   x.abs().amax(dim=1, keepdim=True).float() * inv_max
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace torchao {

namespace {

constexpr int kThreads = 256;

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <typename T>
__global__ void rowwise_abs_max_mul_kernel(
    const T* __restrict__ input,
    int64_t n_rows,
    int64_t n_cols,
    int64_t stride0,
    int64_t stride1,
    float inv_max,
    float* __restrict__ out_scale) {
  const int64_t row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  const T* row_ptr = input + row * stride0;
  float local = 0.f;
  for (int64_t c = threadIdx.x; c < n_cols; c += blockDim.x) {
    float v = fabsf(to_float(row_ptr[c * stride1]));
    local = fmaxf(local, v);
  }
  __shared__ float smem[kThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_scale[row] = smem[0] * inv_max;
  }
}

template <typename T>
void launch_rowwise_abs_max_mul(
    const at::Tensor& input_2d,
    float inv_max,
    at::Tensor& out_m1) {
  const int64_t n_rows = input_2d.size(0);
  const int64_t n_cols = input_2d.size(1);
  const int64_t stride0 = input_2d.stride(0);
  const int64_t stride1 = input_2d.stride(1);
  const T* input_ptr = input_2d.data_ptr<T>();
  float* out_ptr = out_m1.data_ptr<float>();
  rowwise_abs_max_mul_kernel<T><<<n_rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      input_ptr, n_rows, n_cols, stride0, stride1, inv_max, out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

at::Tensor rowwise_abs_max_fp32_scale(
    const at::Tensor& input_2d,
    const at::Tensor& inv_max_scalar) {
  TORCH_CHECK(input_2d.dim() == 2, "rowwise_abs_max_fp32_scale: expected 2D input");
  TORCH_CHECK(
      input_2d.device().is_cuda(),
      "rowwise_abs_max_fp32_scale: expected CUDA tensor");
  TORCH_CHECK(
      inv_max_scalar.numel() == 1,
      "rowwise_abs_max_fp32_scale: inv_max must be a scalar tensor");
  TORCH_CHECK(
      inv_max_scalar.scalar_type() == at::ScalarType::Float,
      "rowwise_abs_max_fp32_scale: inv_max must be float32");
  const float inv_max = inv_max_scalar.item<float>();
  at::Tensor input_contig = input_2d.contiguous();
  const at::cuda::OptionalCUDAGuard device_guard(input_contig.device());
  at::Tensor out = at::empty(
      {input_contig.size(0), 1},
      input_contig.options().dtype(at::kFloat));
  if (input_contig.scalar_type() == at::ScalarType::BFloat16) {
    launch_rowwise_abs_max_mul<c10::BFloat16>(input_contig, inv_max, out);
  } else if (input_contig.scalar_type() == at::ScalarType::Half) {
    launch_rowwise_abs_max_mul<c10::Half>(input_contig, inv_max, out);
  } else if (input_contig.scalar_type() == at::ScalarType::Float) {
    launch_rowwise_abs_max_mul<float>(input_contig, inv_max, out);
  } else {
    TORCH_CHECK(
        false,
        "rowwise_abs_max_fp32_scale: unsupported dtype ",
        input_contig.scalar_type(),
        " (use bf16/fp16/fp32)");
  }
  return out;
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchao::rowwise_abs_max_fp32_scale"),
      TORCH_FN(rowwise_abs_max_fp32_scale));
}

} // namespace torchao
