import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rmsnorm import rmsnorm_ref
from src.rmsnorm_cuda import rmsnorm_cuda


def test_one(dtype, batch, seq_len, hidden_size):
    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    y_ref = rmsnorm_ref(x, weight)
    y_cuda = rmsnorm_cuda(x, weight)

    max_abs_err = (y_ref.float() - y_cuda.float()).abs().max().item()
    mean_abs_err = (y_ref.float() - y_cuda.float()).abs().mean().item()

    print(
        f"dtype={str(dtype).replace('torch.', ''):<8} "
        f"shape=({batch},{seq_len},{hidden_size}) "
        f"max_abs_err={max_abs_err:.6e} "
        f"mean_abs_err={mean_abs_err:.6e}"
    )

    if dtype == torch.float32:
        assert max_abs_err < 1e-5
    elif dtype == torch.float16:
        assert max_abs_err < 5e-3
    elif dtype == torch.bfloat16:
        assert max_abs_err < 5e-2


def main():
    shapes = [
        (1, 128, 1024),
        (1, 128, 2048),
        (1, 128, 4096),
        (1, 512, 4096),
        (4, 128, 4096),
    ]

    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    for shape in shapes:
        for dtype in dtypes:
            test_one(dtype, *shape)

    print("All correctness tests passed.")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available."
    main()