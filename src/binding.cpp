#include <torch/extension.h>

torch::Tensor rmsnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps,
    int64_t block_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward_cuda, "Fused RMSNorm forward CUDA with configurable block size");
}
