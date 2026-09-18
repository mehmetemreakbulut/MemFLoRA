#include <torch/extension.h>

at::Tensor pack_bool_cuda(const at::Tensor& mask);
at::Tensor activate_pack_cuda(at::Tensor input, int64_t activation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_bool", &pack_bool_cuda);
  m.def("activate_pack_", &activate_pack_cuda);
}
