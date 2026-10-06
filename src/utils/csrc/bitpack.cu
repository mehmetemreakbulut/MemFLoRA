#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
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
// One thread per output element: coalesced reads/writes, no unpacked bool mask.
template <typename scalar_t>
__global__ void masked_scaled_grad_kernel(
    const scalar_t* grad, const uint8_t* packed, const scalar_t* scale,
    scalar_t* output, int64_t n, int64_t channels, int64_t spatial) {
  using opmath_t = at::opmath_type<scalar_t>;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) {
    const bool pass = (packed[i / 8] & uint8_t(1u << (i % 8))) != 0;
    // Preserve where(mask, grad, 0) * scale semantics, including 0 * inf.
    const opmath_t gated = pass ? opmath_t(grad[i]) : opmath_t(0);
    output[i] = scalar_t(gated * opmath_t(scale[(i / spatial) % channels]));
  }
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

at::Tensor masked_scaled_grad_cuda(const at::Tensor& grad, const at::Tensor& packed,
                                 const at::Tensor& scale) {
  TORCH_CHECK(grad.is_cuda() && grad.is_contiguous() && grad.dim() == 4,
              "expected contiguous CUDA NCHW gradient");
  TORCH_CHECK(packed.device() == grad.device() && packed.is_contiguous() &&
              packed.scalar_type() == at::kByte, "expected same-device packed uint8 mask");
  TORCH_CHECK(scale.device() == grad.device() && scale.is_contiguous() &&
              scale.dim() == 1 && scale.numel() == grad.size(1) &&
              scale.scalar_type() == grad.scalar_type(), "invalid channel scale");
  TORCH_CHECK(packed.numel() >= (grad.numel() + 7) / 8, "packed mask too short");
  const c10::cuda::CUDAGuard guard(grad.device());
  auto output = at::empty(grad.sizes(), grad.options());
  if (grad.numel()) {
    const int grid = int(std::min<int64_t>((grad.numel() + threads - 1) / threads, 4096));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad.scalar_type(),
        "memflora_masked_scaled_grad", [&] {
          masked_scaled_grad_kernel<scalar_t><<<grid, threads, 0,
              c10::cuda::getCurrentCUDAStream(grad.get_device()).stream()>>>(
              grad.data_ptr<scalar_t>(), packed.data_ptr<uint8_t>(),
              scale.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
              grad.numel(), grad.size(1), grad.size(2) * grad.size(3));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
