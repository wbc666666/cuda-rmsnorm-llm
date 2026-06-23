import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/new_4tdisk/wbc/models/Qwen2.5-0.5B-Instruct"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda").eval()

    prompt = "请用一句话介绍异构计算。"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("Generating...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
