import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rmsnorm import rmsnorm_ref
from src.rmsnorm_cuda import rmsnorm_cuda


def run_pytorch(x, weight, repeats=50):
    for _ in range(repeats):
        y = rmsnorm_ref(x, weight)
    return y


def run_cuda(x, weight, repeats=50):
    for _ in range(repeats):
        y = rmsnorm_cuda(x, weight, block_size=128)
    return y


def profile_one(name, fn, x, weight, output_dir):
    # 先 warmup，避免第一次 CUDA 初始化影响 profiler
    with torch.no_grad():
        for _ in range(20):
            _ = fn(x, weight, repeats=1)
    torch.cuda.synchronize()

    trace_path = output_dir / f"{name}_trace.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            with record_function(name):
                _ = fn(x, weight, repeats=50)

    torch.cuda.synchronize()

    print(f"\n===== {name} =====")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
    ))

    prof.export_chrome_trace(str(trace_path))
    print(f"Saved trace to: {trace_path}")


def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    output_dir = PROJECT_ROOT / "results" / "profiler"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    batch = 1
    seq_len = 512
    hidden_size = 4096
    dtype = torch.float16
    device = "cuda"

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)

    print(f"Input shape: {tuple(x.shape)}, dtype={dtype}")

    profile_one(
        name="pytorch_rmsnorm",
        fn=run_pytorch,
        x=x,
        weight=weight,
        output_dir=output_dir,
    )

    profile_one(
        name="cuda_fused_rmsnorm",
        fn=run_cuda,
        x=x,
        weight=weight,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
