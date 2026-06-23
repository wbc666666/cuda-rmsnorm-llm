import csv
import sys
import statistics
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rmsnorm import rmsnorm_ref
from src.rmsnorm_cuda import rmsnorm_cuda


def measure(func, x, weight, warmup=100, iters=500, repeats=5):
    times = []

    for _ in range(repeats):
        for _ in range(warmup):
            _ = func(x, weight)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = func(x, weight)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_one(dtype, batch, seq_len, hidden_size):
    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    pytorch_time = measure(rmsnorm_ref, x, weight)
    cuda_time = measure(rmsnorm_cuda, x, weight)

    y_ref = rmsnorm_ref(x, weight)
    y_cuda = rmsnorm_cuda(x, weight)

    max_abs_err = (y_ref.float() - y_cuda.float()).abs().max().item()
    mean_abs_err = (y_ref.float() - y_cuda.float()).abs().mean().item()

    num_tokens = batch * seq_len

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "seq_len": seq_len,
        "hidden_size": hidden_size,

        "pytorch_median_ms": pytorch_time["median_ms"],
        "pytorch_mean_ms": pytorch_time["mean_ms"],
        "pytorch_min_ms": pytorch_time["min_ms"],
        "pytorch_max_ms": pytorch_time["max_ms"],

        "cuda_median_ms": cuda_time["median_ms"],
        "cuda_mean_ms": cuda_time["mean_ms"],
        "cuda_min_ms": cuda_time["min_ms"],
        "cuda_max_ms": cuda_time["max_ms"],

        "speedup_median": pytorch_time["median_ms"] / cuda_time["median_ms"],
        "pytorch_tokens_per_sec": num_tokens / (pytorch_time["median_ms"] / 1000.0),
        "cuda_tokens_per_sec": num_tokens / (cuda_time["median_ms"] / 1000.0),

        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
    }


def main():
    torch.cuda.empty_cache()

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

    print("Stable benchmark: PyTorch RMSNorm vs custom CUDA RMSNorm")
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
                f"pytorch_median={r['pytorch_median_ms']:.4f} ms "
                f"cuda_median={r['cuda_median_ms']:.4f} ms "
                f"speedup={r['speedup_median']:.2f}x "
                f"max_err={r['max_abs_err']:.6e}"
            )

    output_path = PROJECT_ROOT / "results" / "rmsnorm_cuda_benchmark_stable.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available."
    main()