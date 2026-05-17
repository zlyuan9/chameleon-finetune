import csv
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = sys.argv[1]   # e.g. resources/baseline_results_4bjudge_final.csv
OUTPUT_FILE = sys.argv[2]  # e.g. resources/baseline_classified.csv

MODEL_ID = "Qwen/Qwen3-4B"

SYSTEM_PROMPT = (
    "You are a radiology expert. Given a known diagnosis and a generated radiology impression, "
    "determine whether the impression correctly identifies the diagnosis.\n"
    "Answer with a single digit: 1 if the impression correctly identifies the diagnosis, 0 if it does not.\n"
    "Output only the digit, nothing else."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

with open(INPUT_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

results = []
for i, row in enumerate(rows):
    diagnosis = row["diagnosis"]
    generated = row["generated"][:2000]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Known diagnosis: {diagnosis}\n\n"
            f"Generated impression:\n{generated}\n\n"
            "Does this impression correctly identify the diagnosis? Answer 1 or 0."
        )}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True, enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    correct = 1 if response.startswith("1") else 0

    print(f"[{i+1}/{len(rows)}] {diagnosis} -> {response} -> {correct}", flush=True)
    results.append({**row, "correct": correct})

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["correct"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. Saved to {OUTPUT_FILE}")
