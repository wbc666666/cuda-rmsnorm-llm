import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    PyTorch RMSNorm baseline.

    输入:
        x: [batch, seq_len, hidden_size]
        weight: [hidden_size]

    计算:
        y = x / sqrt(mean(x^2) + eps) * weight

    这里使用 FP32 累加，提高数值稳定性。
    输出 dtype 与输入 x 保持一致。
    """
    x_float = x.float()
    weight_float = weight.float()

    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    y = x_norm * weight_float

    return y.to(dtype=x.dtype)
