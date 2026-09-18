#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdint>

namespace {
constexpr int threads = 256;

// One thread owns one output byte. Tail bits are zero; no atomics, scratch
// buffers, full-size integer conversion, or device synchronization are needed.
__global__ void pack_bool_kernel(const bool* input, uint8_t* output, int64_t n) {
  for (int64_t byte = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       byte < (n + 7) / 8; byte += int64_t(blockDim.x) * gridDim.x) {
    uint8_t packed = 0;
    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int64_t i = byte * 8 + bit;
      if (i < n && input[i]) packed |= uint8_t(1u << bit);
    }
    output[byte] = packed;
  }
}

// Fuse ReLU/ReLU6 and gate packing: the full-width boolean mask never exists.
// Comparisons preserve NaNs in the output and match the existing strict gate
// (x > 0, and x < 6 for ReLU6), including the two boundary values.
template <typename scalar_t>
__global__ void activate_pack_kernel(
    scalar_t* input, uint8_t* output, int64_t n, int activation) {
  for (int64_t byte = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       byte < (n + 7) / 8; byte += int64_t(blockDim.x) * gridDim.x) {
    uint8_t packed = 0;
    #pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int64_t i = byte * 8 + bit;
      if (i < n) {
        const scalar_t value = input[i];
        const bool positive = value > scalar_t(0);
        if (positive && (activation == 1 || value < scalar_t(6)))
          packed |= uint8_t(1u << bit);
        scalar_t result = value < scalar_t(0) ? scalar_t(0) : value;
        if (activation == 2 && value > scalar_t(6)) result = scalar_t(6);
        input[i] = result;
      }
    }
    output[byte] = packed;
  }
}

int blocks(int64_t n) {
  return int(std::min<int64_t>(((n + 7) / 8 + threads - 1) / threads, 4096));
}
}  // namespace

at::Tensor pack_bool_cuda(const at::Tensor& mask) {
  TORCH_CHECK(mask.is_cuda() && mask.is_contiguous(), "expected contiguous CUDA mask");
  TORCH_CHECK(mask.scalar_type() == at::kBool, "expected boolean mask");
  const c10::cuda::CUDAGuard guard(mask.device());
  auto output = at::empty({(mask.numel() + 7) / 8}, mask.options().dtype(at::kByte));
  if (mask.numel()) {
    pack_bool_kernel<<<blocks(mask.numel()), threads, 0,
        c10::cuda::getCurrentCUDAStream(mask.get_device()).stream()>>>(
        mask.data_ptr<bool>(), output.data_ptr<uint8_t>(), mask.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor activate_pack_cuda(at::Tensor input, int64_t activation) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "expected contiguous CUDA input");
  TORCH_CHECK(activation == 1 || activation == 2, "expected ReLU or ReLU6");
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty({(input.numel() + 7) / 8}, input.options().dtype(at::kByte));
  if (input.numel()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
        "memflora_activate_pack", [&] {
          activate_pack_kernel<scalar_t><<<blocks(input.numel()), threads, 0,
              c10::cuda::getCurrentCUDAStream(input.get_device()).stream()>>>(
              input.data_ptr<scalar_t>(), output.data_ptr<uint8_t>(),
              input.numel(), int(activation));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
