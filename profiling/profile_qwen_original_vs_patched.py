import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function
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
    text = base_text * 300

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding=False,
    ).to("cuda")

    return inputs


def warmup(model, input_ids, attention_mask, warmup_steps=10):
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    torch.cuda.synchronize()


def profile_model(name, model, input_ids, attention_mask, output_dir, repeats=10):
    trace_path = output_dir / f"{name}_trace.json"

    warmup(model, input_ids, attention_mask, warmup_steps=10)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            with record_function(name):
                for _ in range(repeats):
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )

    torch.cuda.synchronize()

    print(f"\n===== {name} =====")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
    ))

    prof.export_chrome_trace(str(trace_path))
    print(f"Saved trace to: {trace_path}")


def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = PROJECT_ROOT / "results" / "profiler"
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16
    seq_len = 512

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    inputs = make_inputs(tokenizer, seq_len)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    print(f"seq_len={input_ids.shape[1]}, dtype={dtype}")

    print("Loading original Qwen...")
    model_original, _ = load_model(dtype=dtype, patch=False)

    print("Loading patched Qwen...")
    model_patched, replaced = load_model(dtype=dtype, patch=True)
    print(f"Replaced RMSNorm layers: {replaced}")

    profile_model(
        name="qwen_original_fp16_seq512",
        model=model_original,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_dir=output_dir,
        repeats=10,
    )

    profile_model(
        name="qwen_patched_fp16_seq512",
        model=model_patched,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_dir=output_dir,
        repeats=10,
    )

    print("\nProfiler finished.")


if __name__ == "__main__":
    main()
