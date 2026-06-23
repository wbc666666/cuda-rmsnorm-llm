import torch
import rmsnorm_cuda_ext


def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    block_size: int = 256,
):
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA tensor")
    if block_size not in [128, 256, 512]:
        raise ValueError("block_size must be one of 128, 256, or 512")

    x = x.contiguous()
    weight = weight.contiguous()

    return rmsnorm_cuda_ext.rmsnorm_forward(x, weight, eps, block_size)
