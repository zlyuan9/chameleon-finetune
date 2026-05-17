# judge server - runs Qwen3-8B locally and exposes a /judge endpoint
# run with: uvicorn judge_server:app --host 0.0.0.0 --port 8000

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = "Qwen/Qwen3-4B"

app = FastAPI()
model = None
tokenizer = None

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


@app.on_event("startup")
def load_model():
    global model, tokenizer
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model ready")


class JudgeRequest(BaseModel):
    actual: str
    expected: str


@app.post("/judge")
def judge(req: JudgeRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"EXPECTED (reference) impression:\n{req.expected}\n\n"
                f"ACTUAL (generated) impression:\n{req.actual}\n\n"
                "Compare actual to expected and output SCORE and REASON as instructed."
            )
        }
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
            max_new_tokens=256,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    score_match = re.search(r'SCORE:\s*\*{0,2}(\d+)\*{0,2}', response_text, re.IGNORECASE)
    if not score_match:
        # fallback: find any standalone integer 0-10
        score_match = re.search(r'\b([0-9]|10)\b', response_text)
    score = int(score_match.group(1)) if score_match else 0
    reason_match = re.search(r'REASON:\s*(.+)', response_text, re.IGNORECASE)
    reason = reason_match.group(1).strip() if reason_match else response_text.strip()

    return {"score": score, "reason": reason}


@app.get("/health")
def health():
    return {"status": "ok"}
