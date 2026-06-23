import csv
import sys
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm.qwen_rmsnorm_patch import replace_qwen_rmsnorm_with_cuda


MODEL_PATH = "/mnt/new_4tdisk/wbc/models/Qwen2.5-0.5B-Instruct"


def measure_forward(model, input_ids, attention_mask=None, warmup=5, repeats=20):
    times = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        torch.cuda.synchronize()

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def load_model(dtype=torch.float16, patch=False):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda").eval()

    replaced = 0
    if patch:
        replaced = replace_qwen_rmsnorm_with_cuda(model, block_size=128)

    return model, replaced


def make_inputs(tokenizer, seq_len):
    base_text = (
        "异构计算是利用 CPU、GPU、FPGA 等不同计算单元协同完成任务的一种计算方式。"
        "在大语言模型推理中，不同算子具有不同的计算和访存特征，因此需要针对硬件进行优化。"
    )
    text = base_text * 200

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding=False,
    ).to("cuda")

    return inputs


def benchmark_dtype(dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"\n===== dtype={dtype} =====")

    print("Loading original Qwen...")
    model_ref, _ = load_model(dtype=dtype, patch=False)

    print("Loading CUDA-RMSNorm patched Qwen...")
    model_cuda, replaced = load_model(dtype=dtype, patch=True)
    print(f"Replaced RMSNorm layers: {replaced}")

    results = []

    for seq_len in [32, 128, 512, 1024]:
        inputs = make_inputs(tokenizer, seq_len)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        actual_seq_len = input_ids.shape[1]

        ref_time = measure_forward(model_ref, input_ids, attention_mask)
        cuda_time = measure_forward(model_cuda, input_ids, attention_mask)

        speedup = ref_time["median_ms"] / cuda_time["median_ms"]

        print(
            f"seq_len={actual_seq_len:<4} "
            f"original={ref_time['median_ms']:.4f} ms "
            f"patched={cuda_time['median_ms']:.4f} ms "
            f"speedup={speedup:.4f}x"
        )

        results.append({
            "dtype": str(dtype).replace("torch.", ""),
            "seq_len": actual_seq_len,
            "replaced_rmsnorm_layers": replaced,
            "original_median_ms": ref_time["median_ms"],
            "patched_median_ms": cuda_time["median_ms"],
            "speedup": speedup,
            "original_mean_ms": ref_time["mean_ms"],
            "patched_mean_ms": cuda_time["mean_ms"],
            "original_min_ms": ref_time["min_ms"],
            "patched_min_ms": cuda_time["min_ms"],
            "original_max_ms": ref_time["max_ms"],
            "patched_max_ms": cuda_time["max_ms"],
        })

    del model_ref
    del model_cuda
    torch.cuda.empty_cache()

    return results


def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    all_results = []

    all_results.extend(benchmark_dtype(torch.float16))

    if torch.cuda.is_bf16_supported():
        all_results.extend(benchmark_dtype(torch.bfloat16))

    output_path = PROJECT_ROOT / "results" / "qwen_rmsnorm_patch_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
