import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm.qwen_rmsnorm_patch import (
    replace_qwen_rmsnorm_with_cuda,
    count_rmsnorm_layers,
)

MODEL_PATH = "/mnt/new_4tdisk/wbc/models/Qwen2.5-0.5B-Instruct"


def load_model(dtype=torch.float16):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda").eval()


def main():
    assert torch.cuda.is_available(), "CUDA is not available."

    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Loading original model...")
    model_ref = load_model(dtype=dtype)

    print("Loading CUDA-RMSNorm patched model...")
    model_cuda = load_model(dtype=dtype)

    before_count, before_names = count_rmsnorm_layers(model_cuda)
    print(f"RMSNorm-like layers before patch: {before_count}")
    for n, c in before_names[:10]:
        print("  ", n, c)
    if before_count > 10:
        print("  ...")

    replaced = replace_qwen_rmsnorm_with_cuda(model_cuda, block_size=128)
    after_count, after_names = count_rmsnorm_layers(model_cuda)

    print(f"Replaced RMSNorm layers: {replaced}")
    print(f"RMSNorm-like layers after patch: {after_count}")
    for n, c in after_names[:10]:
        print("  ", n, c)
    if after_count > 10:
        print("  ...")

    prompt = "请用一句话介绍异构计算。"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        logits_ref = model_ref(**inputs).logits
        logits_cuda = model_cuda(**inputs).logits

    max_abs_err = (logits_ref.float() - logits_cuda.float()).abs().max().item()
    mean_abs_err = (logits_ref.float() - logits_cuda.float()).abs().mean().item()

    print(f"logits max_abs_err={max_abs_err:.6e}")
    print(f"logits mean_abs_err={mean_abs_err:.6e}")

    print("Qwen RMSNorm patch test finished.")


if __name__ == "__main__":
    main()
