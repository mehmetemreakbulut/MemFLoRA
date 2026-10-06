#include <torch/extension.h>

at::Tensor pack_bool_cuda(const at::Tensor& mask);
at::Tensor activate_pack_cuda(at::Tensor input, int64_t activation);
at::Tensor masked_scaled_grad_cuda(const at::Tensor& grad, const at::Tensor& packed,
                                 const at::Tensor& scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_bool", &pack_bool_cuda);
  m.def("activate_pack_", &activate_pack_cuda);
  m.def("masked_scaled_grad", &masked_scaled_grad_cuda);
}
