# baseline evaluation harness
# for each row in test set: generate impression with Qwen, score with judge, save results to csv

import argparse
import csv
import json
import math
import os
import random
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from judge import judge

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
TEST_FILE = "resources/test.jsonl"
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "resources/baseline_results.csv")

SYSTEM_PROMPT = (
    "You are a radiologist. Given the clinical indication and CT findings, "
    "write the impression section of the radiology report. "
    "Include: the primary diagnosis or conclusion, key supporting findings, "
    "any clinically significant incidental findings, and appropriate follow-up recommendations. "
    "Be concise and clinically accurate."
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_impression(model, tokenizer, report):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of rows to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="skip first N rows")
    parser.add_argument("--resume", action="store_true", help="resume from existing CSV, skipping already completed rows")
    parser.add_argument("--sample-positive", type=int, default=None, help="sample N positive cases evenly across pathologies")
    parser.add_argument("--sample-healthy", type=int, default=None, help="sample N healthy cases")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(TEST_FILE) as f:
        all_data = json.load(f)

    if args.sample_positive or args.sample_healthy:
        random.seed(args.seed)
        healthy = [r for r in all_data if r["diagnosis"] == "healthy"]
        positive = [r for r in all_data if r["diagnosis"] != "healthy"]

        sampled = []
        if args.sample_healthy:
            sampled += random.sample(healthy, min(args.sample_healthy, len(healthy)))

        if args.sample_positive:
            from collections import defaultdict
            by_pathology = defaultdict(list)
            for r in positive:
                by_pathology[r["diagnosis"]].append(r)
            n_per = math.ceil(args.sample_positive / len(by_pathology))
            for records in by_pathology.values():
                sampled += random.sample(records, min(n_per, len(records)))
            # trim to exact count in case ceil overshot
            positive_sampled = [r for r in sampled if r["diagnosis"] != "healthy"]
            sampled = [r for r in sampled if r["diagnosis"] == "healthy"] + positive_sampled[:args.sample_positive]

        random.shuffle(sampled)
        test_data = sampled
    else:
        test_data = all_data[args.offset:]
        if args.limit:
            test_data = test_data[:args.limit]

    # count already completed rows to skip
    completed = 0
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, newline="") as f:
            completed = sum(1 for _ in csv.reader(f)) - 1  # subtract header
        print(f"Resuming from row {completed + 1}")
        test_data = test_data[completed:]

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    model, tokenizer = load_model()

    write_mode = "a" if args.resume and completed > 0 else "w"
    with open(OUTPUT_FILE, write_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["diagnosis", "generated", "expected", "score"])
        if write_mode == "w":
            writer.writeheader()

        for i, row in enumerate(test_data):
            print(f"[{completed + i + 1}/{completed + len(test_data)}] {row['diagnosis']}")

            generated = generate_impression(model, tokenizer, row["report"])
            score = judge(generated, row["impression"], session_id=session_id)

            writer.writerow({
                "diagnosis": row["diagnosis"],
                "generated": generated,
                "expected": row["impression"],
                "score": score,
            })
            csvfile.flush()

    print(f"Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
