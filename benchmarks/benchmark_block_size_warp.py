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


def benchmark_one(dtype, batch, seq_len, hidden_size, block_size):
    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    cuda_func = lambda a, b: rmsnorm_cuda(a, b, block_size=block_size)

    cuda_time = measure(cuda_func, x, weight)

    y_ref = rmsnorm_ref(x, weight)
    y_cuda = rmsnorm_cuda(x, weight, block_size=block_size)

    max_abs_err = (y_ref.float() - y_cuda.float()).abs().max().item()
    mean_abs_err = (y_ref.float() - y_cuda.float()).abs().mean().item()

    num_tokens = batch * seq_len
    tokens_per_sec = num_tokens / (cuda_time["median_ms"] / 1000.0)

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "block_size": block_size,

        "cuda_median_ms": cuda_time["median_ms"],
        "cuda_mean_ms": cuda_time["mean_ms"],
        "cuda_min_ms": cuda_time["min_ms"],
        "cuda_max_ms": cuda_time["max_ms"],

        "cuda_tokens_per_sec": tokens_per_sec,
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

    block_sizes = [128, 256, 512]

    results = []

    print("Block size benchmark for warp-level CUDA RMSNorm")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print()

    for batch, seq_len, hidden_size in shapes:
        for dtype in dtypes:
            best = None

            for block_size in block_sizes:
                r = benchmark_one(dtype, batch, seq_len, hidden_size, block_size)
                results.append(r)

                print(
                    f"block={block_size:<3} "
                    f"dtype={r['dtype']:<8} "
                    f"shape=({batch},{seq_len},{hidden_size}) "
                    f"median={r['cuda_median_ms']:.4f} ms "
                    f"tokens/s={r['cuda_tokens_per_sec']:.2f} "
                    f"max_err={r['max_abs_err']:.6e}"
                )

                if best is None or r["cuda_median_ms"] < best["cuda_median_ms"]:
                    best = r

            print(
                f"BEST for dtype={str(dtype).replace('torch.', '')}, "
                f"shape=({batch},{seq_len},{hidden_size}): "
                f"block={best['block_size']}, "
                f"median={best['cuda_median_ms']:.4f} ms"
            )
            print("-" * 100)

    output_path = PROJECT_ROOT / "results" / "rmsnorm_block_size_benchmark_warp.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available."
    main()
