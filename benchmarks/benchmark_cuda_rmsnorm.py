import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rmsnorm import rmsnorm_ref
from src.rmsnorm_cuda import rmsnorm_cuda


def time_func(func, x, weight, warmup=50, iters=200):
    for _ in range(warmup):
        _ = func(x, weight)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = func(x, weight)
    end.record()

    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / iters
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return avg_ms, peak_mem_mb, y


def benchmark_one(dtype, batch, seq_len, hidden_size):
    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    pytorch_ms, pytorch_mem, y_ref = time_func(rmsnorm_ref, x, weight)
    cuda_ms, cuda_mem, y_cuda = time_func(rmsnorm_cuda, x, weight)

    max_abs_err = (y_ref.float() - y_cuda.float()).abs().max().item()
    mean_abs_err = (y_ref.float() - y_cuda.float()).abs().mean().item()

    num_tokens = batch * seq_len
    pytorch_tokens_s = num_tokens / (pytorch_ms / 1000.0)
    cuda_tokens_s = num_tokens / (cuda_ms / 1000.0)
    speedup = pytorch_ms / cuda_ms

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "pytorch_latency_ms": pytorch_ms,
        "cuda_latency_ms": cuda_ms,
        "speedup": speedup,
        "pytorch_tokens_per_sec": pytorch_tokens_s,
        "cuda_tokens_per_sec": cuda_tokens_s,
        "pytorch_peak_memory_mb": pytorch_mem,
        "cuda_peak_memory_mb": cuda_mem,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
    }


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

    results = []

    print("Running PyTorch baseline vs custom CUDA RMSNorm benchmark...")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print()

    for batch, seq_len, hidden_size in shapes:
        for dtype in dtypes:
            r = benchmark_one(dtype, batch, seq_len, hidden_size)
            results.append(r)

            print(
                f"dtype={r['dtype']:<8} "
                f"shape=({batch},{seq_len},{hidden_size}) "
                f"pytorch={r['pytorch_latency_ms']:.4f} ms "
                f"cuda={r['cuda_latency_ms']:.4f} ms "
                f"speedup={r['speedup']:.2f}x "
                f"max_err={r['max_abs_err']:.6e}"
            )

    output_path = PROJECT_ROOT / "results" / "rmsnorm_cuda_benchmark.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available."
    main()