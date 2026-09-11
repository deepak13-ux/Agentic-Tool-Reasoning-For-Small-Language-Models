"""
V1 — SFT with Curriculum Learning for Qwen 3B BM25 Agent.

Trains in 3 stages (easy -> hard):
  Stage 1: Single-step tool-call + no-tool examples (2 epochs)
  Stage 2: + Multi-step + clarification examples (2 epochs)
  Stage 3: Full dataset incl. adversarial + error-recovery (1 epoch)
"""

import os
import gc
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "train_data_3b.jsonl"
OUTPUT_DIR = "./qwen-3b-bm25-sft-v1"
MAX_SEQ_LEN = 1024

# ---------------------------------------------------------------------------
# Tokenizer & Model
# ---------------------------------------------------------------------------

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model {MODEL_ID} in BFloat16...")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
)
model.gradient_checkpointing_enable()
for param in model.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

# ---------------------------------------------------------------------------
# Dataset loading & classification
# ---------------------------------------------------------------------------

def classify_trajectory(record):
    """Classify a trajectory into curriculum stage."""
    msgs = record["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]

    # Check for error-recovery (has 2+ tool calls, second search is a retry)
    if len(tool_msgs) >= 2:
        for a in assistant_msgs:
            if "refine" in a["content"].lower() or "retry" in a["content"].lower():
                return "stage3"

    # Check for adversarial (search + "not available" type answer)
    for a in assistant_msgs:
        content = a["content"].lower()
        if "not available" in content or "could not find" in content or "unable to find" in content:
            return "stage3"

    # Multi-step (2+ tool calls)
    if len(tool_msgs) >= 2:
        return "stage2"

    # Clarification (no tool call, asks for more info)
    if len(tool_msgs) == 0:
        for a in assistant_msgs:
            if "clarify" in a["content"].lower() or "specify" in a["content"].lower():
                return "stage2"

    # Single-step or no-tool = stage 1
    return "stage1"


def preprocess_function(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    tokenized = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


print(f"Loading dataset from {DATASET_PATH}...")
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Total dataset size: {len(raw_dataset)} examples")

# Classify into stages
stage1_records, stage2_records, stage3_records = [], [], []
for i in range(len(raw_dataset)):
    record = raw_dataset[i]
    stage = classify_trajectory(record)
    if stage == "stage1":
        stage1_records.append(record)
    elif stage == "stage2":
        stage2_records.append(record)
    else:
        stage3_records.append(record)

print(f"  Stage 1 (easy):   {len(stage1_records)} examples")
print(f"  Stage 2 (medium): {len(stage2_records)} examples")
print(f"  Stage 3 (hard):   {len(stage3_records)} examples")

# ---------------------------------------------------------------------------
# Curriculum Training
# ---------------------------------------------------------------------------

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

def train_stage(stage_name, records, num_epochs, lr, resume_from=None):
    """Train one curriculum stage."""
    if not records:
        print(f"  [{stage_name}] No data — skipping.")
        return

    ds = Dataset.from_list(records)
    tokenized_ds = ds.map(preprocess_function, remove_columns=ds.column_names, desc=f"Tokenizing {stage_name}")

    stage_output = f"{OUTPUT_DIR}/{stage_name}"
    args = TrainingArguments(
        output_dir=stage_output,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized_ds, data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(stage_output)
    tokenizer.save_pretrained(stage_output)
    print(f"  [{stage_name}] Saved to {stage_output}")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("V1 — Curriculum SFT Training")
    print(f"{'='*70}")

    # Stage 1: Easy — single-step + no-tool (2 epochs, higher LR)
    print(f"\n--- Stage 1: Single-step + No-tool ({len(stage1_records)} examples, 2 epochs) ---")
    train_stage("stage1", stage1_records, num_epochs=2, lr=2e-5)

    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2: Medium — add multi-step + clarification (2 epochs, lower LR)
    print(f"\n--- Stage 2: + Multi-step + Clarification ({len(stage1_records) + len(stage2_records)} examples, 2 epochs) ---")
    train_stage("stage2", stage1_records + stage2_records, num_epochs=2, lr=1e-5)

    gc.collect()
    torch.cuda.empty_cache()

    # Stage 3: Hard — full dataset (1 epoch, lowest LR)
    all_records = stage1_records + stage2_records + stage3_records
    print(f"\n--- Stage 3: Full dataset ({len(all_records)} examples, 1 epoch) ---")
    train_stage("stage3", all_records, num_epochs=1, lr=5e-6)

    # Save final model
    print(f"\nSaving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Curriculum SFT training finished!")
