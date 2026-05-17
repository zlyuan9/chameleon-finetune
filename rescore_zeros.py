import csv
import json
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
INPUT_FILE = "resources/grpo_results.csv"
OUTPUT_FILE = "resources/grpo_results_fixed.csv"

SYSTEM_PROMPT = (
    "You are an expert radiologist. Compare the GENERATED impression to the EXPECTED (reference) impression. "
    "Your job is not to invent new criteria—measure how well the generated text aligns with what the reference states.\n\n"
    "Compare systematically:\n"
    "- Diagnosis / impression: does generated match the reference's main conclusion(s)? What's missing or different?\n"
    "- Findings: which reference findings appear in generated; which are omitted or altered?\n"
    "- Contradictions / extras: does generated add or deny anything the reference does not support?\n"
    "- Follow-up / recommendations: consistent with the reference where applicable?\n"
    "- Tone and terminology: clinically reasonable relative to the reference?\n\n"
    "Then assign one overall score from 0 (poor match to expected) to 10 (strong match; omissions/contradictions minor or none).\n\n"
    "Respond with a single integer score and a brief one sentence summarizing the comparison (what matched vs diverged).\n"
    "Format: SCORE: <number>\nREASON: <sentence>"
)


def rescore(actual, expected):
    actual = actual[:3000]
    expected = expected[:3000]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"EXPECTED (reference) impression:\n{expected}\n\n"
            f"ACTUAL (generated) impression:\n{actual}\n\n"
            "Compare actual to expected and output SCORE and REASON as instructed."
        )}
    ]
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        data=json.dumps({
            "model": "anthropic/claude-haiku-4-5",
            "messages": messages,
            "max_tokens": 100,
        }),
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"] or ""
    match = re.search(r'SCORE:\s*(\d+)', content, re.IGNORECASE)
    score = int(match.group(1)) if match else 0
    print(f"  rescored: {score}")
    return score


with open(INPUT_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

zero_count = sum(1 for r in rows if r["score"] == "0")
print(f"Found {zero_count} zero-score rows to fix out of {len(rows)} total")

fixed = 0
for i, row in enumerate(rows):
    if row["score"] == "0":
        print(f"[{i+1}/{len(rows)}] {row['diagnosis']} — rescoring...")
        try:
            new_score = rescore(row["generated"], row["expected"])
            row["score"] = str(new_score)
            fixed += 1
        except Exception as e:
            print(f"  error: {e}")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["diagnosis", "generated", "expected", "score"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. Fixed {fixed} rows. Saved to {OUTPUT_FILE}")
