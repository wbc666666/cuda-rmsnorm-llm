import torch
import torch.nn as nn

from src.rmsnorm_cuda import rmsnorm_cuda


class CudaQwenRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6, block_size: int = 128):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.variance_epsilon = eps
        self.block_size = block_size

    def forward(self, hidden_states: torch.Tensor):
        # 尽量避免无意义拷贝；只有非 contiguous 时才转 contiguous
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        weight = self.weight
        if not weight.is_contiguous():
            weight = weight.contiguous()

        # 模型加载时已经 .to(dtype=torch.float16)，所以这里一般不需要每次 .to(dtype)
        return rmsnorm_cuda(
            hidden_states,
            weight,
            self.variance_epsilon,
            block_size=self.block_size,
        )


def replace_qwen_rmsnorm_with_cuda(model: nn.Module, block_size: int = 128):
    replaced = 0

    for name, child in list(model.named_children()):
        class_name = child.__class__.__name__

        is_rmsnorm = (
            class_name in ["Qwen2RMSNorm", "Qwen3RMSNorm"]
            or class_name.endswith("RMSNorm")
        )

        if is_rmsnorm and hasattr(child, "weight"):
            eps = getattr(child, "variance_epsilon", None)
            if eps is None:
                eps = getattr(child, "eps", 1e-6)

            new_layer = CudaQwenRMSNorm(
                weight=child.weight.data,
                eps=float(eps),
                block_size=block_size,
            ).to(device=child.weight.device, dtype=child.weight.dtype)

            setattr(model, name, new_layer)
            replaced += 1
        else:
            replaced += replace_qwen_rmsnorm_with_cuda(child, block_size=block_size)

    return replaced


def count_rmsnorm_layers(model: nn.Module):
    names = []
    for name, module in model.named_modules():
        if "RMSNorm" in module.__class__.__name__:
            names.append((name, module.__class__.__name__))
    return len(names), names
