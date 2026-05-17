import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grpo/checkpoint-300"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3

SYSTEM_PROMPT = (
    "You are a radiologist. Given the clinical indication and CT findings, "
    "write the impression section of the radiology report. "
    "Include: the primary diagnosis or conclusion, key supporting findings, "
    "any clinically significant incidental findings, and appropriate follow-up recommendations. "
    "Be concise and clinically accurate."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

with open("resources/test.jsonl") as f:
    test_data = json.load(f)

# sample a few positive cases
samples = [r for r in test_data if r["diagnosis"] != "healthy"][:N]

for i, row in enumerate(samples):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["report"]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True, enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"[{i+1}/{N}] {row['diagnosis']}")
    print(f"\n--- GENERATED ---\n{generated}")
    print(f"\n--- EXPECTED ---\n{row['impression']}")
