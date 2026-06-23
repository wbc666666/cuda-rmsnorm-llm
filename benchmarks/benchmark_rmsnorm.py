import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rmsnorm import rmsnorm_ref


def benchmark_one(dtype, batch, seq_len, hidden_size, warmup=50, iters=200):
    device = "cuda"

    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    # 预热
    for _ in range(warmup):
        _ = rmsnorm_ref(x, weight)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        y = rmsnorm_ref(x, weight)
    end_event.record()

    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / iters

    num_tokens = batch * seq_len
    tokens_per_sec = num_tokens / (avg_ms / 1000.0)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # 正确性参考：统一和 FP32 结果比较
    with torch.no_grad():
        y_fp32 = rmsnorm_ref(x.float(), weight.float())
        y_test = y.float()
        max_abs_err = (y_fp32 - y_test).abs().max().item()
        mean_abs_err = (y_fp32 - y_test).abs().mean().item()

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "batch": batch,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "avg_latency_ms": avg_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_mem_mb,
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

    print("Running PyTorch RMSNorm baseline benchmark...")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print()

    for batch, seq_len, hidden_size in shapes:
        for dtype in dtypes:
            result = benchmark_one(dtype, batch, seq_len, hidden_size)
            results.append(result)

            print(
                f"dtype={result['dtype']:<8} "
                f"shape=({batch},{seq_len},{hidden_size}) "
                f"latency={result['avg_latency_ms']:.4f} ms "
                f"tokens/s={result['tokens_per_sec']:.2f} "
                f"mem={result['peak_memory_mb']:.2f} MB "
                f"max_err={result['max_abs_err']:.6e}"
            )

    output_path = PROJECT_ROOT / "results" / "rmsnorm_pytorch_baseline.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available."
    main()
