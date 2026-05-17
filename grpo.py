import os
import torch
import requests
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

SFT_CHECKPOINT = "checkpoints/sft/final"
OUTPUT_DIR = "checkpoints/grpo"
JUDGE_SERVER_URL = os.environ.get("JUDGE_SERVER_URL", "http://localhost:8000")

SYSTEM_PROMPT = (
    "You are a radiologist. Given the clinical indication and CT findings, "
    "write the impression section of the radiology report. "
    "Include: the primary diagnosis or conclusion, key supporting findings, "
    "any clinically significant incidental findings, and appropriate follow-up recommendations. "
    "Be concise and clinically accurate."
)

tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    SFT_CHECKPOINT,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def format_dataset(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["report"]},
        ],
        "reference": example["impression"],
    }


dataset = load_dataset("json", data_files="resources/train.jsonl", split="train")
dataset = dataset.map(format_dataset, remove_columns=["report", "impression", "diagnosis"])


def reward_fn(completions, prompts, reference, **kwargs):
    scores = []
    # reference is repeated G times by TRL to match number of completions
    # completions are plain decoded strings
    for i, (completion, ref) in enumerate(zip(completions, reference)):
        try:
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
            elif isinstance(completion, dict):
                text = completion.get("content", str(completion))
            else:
                text = str(completion)
            if isinstance(ref, list):
                ref = ref[0] if ref else ""
            resp = requests.post(
                f"{JUDGE_SERVER_URL}/judge",
                json={"actual": text, "expected": str(ref)},
                timeout=120,
            )
            resp.raise_for_status()
            score = float(resp.json()["score"])
        except Exception as e:
            print(f"Judge error [{i}]: {e}", flush=True)
            score = 0.0
        scores.append(score)
    return scores


config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=500,
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,       # effective batch = 8
    num_generations=4,                   # G=4 completions per prompt
    max_completion_length=512,
    generation_kwargs={"temperature": 0.8, "do_sample": True, "max_new_tokens": 512},
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=dataset,
)

resume = os.environ.get("RESUME_FROM_CHECKPOINT")
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(OUTPUT_DIR + "/final")
