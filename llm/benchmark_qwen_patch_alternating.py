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


def run_once(model, input_ids, attention_mask=None):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)


def warmup_model(model, input_ids, attention_mask=None, warmup=8):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    torch.cuda.synchronize()


def benchmark_one_config(model_ref, model_cuda, input_ids, attention_mask, rounds=10):
    original_times = []
    patched_times = []
    raw_records = []

    # 两个模型都先 warmup，避免首轮初始化影响
    warmup_model(model_ref, input_ids, attention_mask)
    warmup_model(model_cuda, input_ids, attention_mask)

    for r in range(rounds):
        if r % 2 == 0:
            order = ["original", "patched"]
        else:
            order = ["patched", "original"]

        for item in order:
            if item == "original":
                t = run_once(model_ref, input_ids, attention_mask)
                original_times.append(t)
            else:
                t = run_once(model_cuda, input_ids, attention_mask)
                patched_times.append(t)

            raw_records.append({
                "round": r,
                "order": "->".join(order),
                "model": item,
                "latency_ms": t,
            })

    original_median = statistics.median(original_times)
    patched_median = statistics.median(patched_times)
    speedup = original_median / patched_median

    summary = {
        "original_median_ms": original_median,
        "patched_median_ms": patched_median,
        "speedup": speedup,
        "original_mean_ms": statistics.mean(original_times),
        "patched_mean_ms": statistics.mean(patched_times),
        "original_min_ms": min(original_times),
        "patched_min_ms": min(patched_times),
        "original_max_ms": max(original_times),
        "patched_max_ms": max(patched_times),
    }

    return summary, raw_records


def benchmark_dtype(dtype):
    print(f"\n===== dtype={dtype} =====")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Loading original Qwen...")
    model_ref, _ = load_model(dtype=dtype, patch=False)

    print("Loading CUDA-RMSNorm patched Qwen...")
    model_cuda, replaced = load_model(dtype=dtype, patch=True)

    print(f"Replaced RMSNorm layers: {replaced}")

    summary_results = []
    raw_results = []

    for seq_len in [32, 128, 512, 1024]:
        inputs = make_inputs(tokenizer, seq_len)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        actual_seq_len = input_ids.shape[1]

        summary, raw = benchmark_one_config(
            model_ref=model_ref,
            model_cuda=model_cuda,
            input_ids=input_ids,
            attention_mask=attention_mask,
            rounds=10,
        )

        row = {
            "dtype": str(dtype).replace("torch.", ""),
            "seq_len": actual_seq_len,
            "replaced_rmsnorm_layers": replaced,
            **summary,
        }

        summary_results.append(row)

        for record in raw:
            raw_results.append({
                "dtype": str(dtype).replace("torch.", ""),
                "seq_len": actual_seq_len,
                **record,
            })

        print(
            f"seq_len={actual_seq_len:<4} "
            f"original={summary['original_median_ms']:.4f} ms "
            f"patched={summary['patched_median_ms']:.4f} ms "
            f"speedup={summary['speedup']:.4f}x"
        )

    del model_ref
    del model_cuda
    torch.cuda.empty_cache()

    return summary_results, raw_results


def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.backends.cuda.matmul.allow_tf32 = True

    all_summary = []
    all_raw = []

    summary, raw = benchmark_dtype(torch.float16)
    all_summary.extend(summary)
    all_raw.extend(raw)

    if torch.cuda.is_bf16_supported():
        summary, raw = benchmark_dtype(torch.bfloat16)
        all_summary.extend(summary)
        all_raw.extend(raw)

    summary_path = PROJECT_ROOT / "results" / "qwen_rmsnorm_patch_alternating_summary.csv"
    raw_path = PROJECT_ROOT / "results" / "qwen_rmsnorm_patch_alternating_raw.csv"

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summary[0].keys())
        writer.writeheader()
        writer.writerows(all_summary)

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_raw[0].keys())
        writer.writeheader()
        writer.writerows(all_raw)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw records to: {raw_path}")


if __name__ == "__main__":
    main()
