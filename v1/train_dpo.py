"""
V1 — DPO (Direct Preference Optimization) Training.

After SFT, this script:
1. Generates paired preference data (chosen vs rejected) using the SFT model
2. Trains with DPO using trl library for sample-efficient preference learning

Requires: pip install trl
"""

import os
import gc
import json
import re
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpus import CORPUS, SYSTEM_PROMPT, bm25_search

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SFT_MODEL_PATH = "./qwen-3b-bm25-sft-v1"
OUTPUT_DIR = "./qwen-3b-bm25-dpo-v1"
PREFERENCE_DATA_FILE = "dpo_pairs.jsonl"

# Task pool for generating preference pairs
TASK_POOL = [
    {"query": "Find the total revenue in Q1 and Q2.", "target_doc": "doc_fin_01", "required_keywords": ["$450,000", "$620,000"]},
    {"query": "How many server nodes are in the US-East data center?", "target_doc": "doc_infra_02", "required_keywords": ["128", "nodes"]},
    {"query": "What is the payload capacity of the logistics fleet trucks?", "target_doc": "doc_logistics_03", "required_keywords": ["18,000", "payload"]},
    {"query": "Check the warehouse inventory for product Alpha.", "target_doc": "doc_inventory_04", "required_keywords": ["850", "product alpha"]},
    {"query": "How many backend engineers does the company employ?", "target_doc": "doc_hr_05", "required_keywords": ["96", "backend"]},
    {"query": "What is the password rotation policy period?", "target_doc": "doc_sec_06", "required_keywords": ["90-day", "password"]},
    {"query": "What was the root cause of the Q3 API incident?", "target_doc": "doc_incident_07", "required_keywords": ["connection pool", "unindexed"]},
    {"query": "How many H100 GPUs are in the AI training cluster?", "target_doc": "doc_ai_08", "required_keywords": ["32", "H100"]},
    {"query": "What was Q4 revenue?", "target_doc": "doc_fin_09", "required_keywords": ["$820,000", "q4"]},
    {"query": "What were the SOC 2 Type II audit findings?", "target_doc": "doc_compliance_10", "required_keywords": ["zero critical", "SOC 2"]},
    {"query": "What percentage of IT budget goes to infrastructure?", "target_doc": "doc_budget_11", "required_keywords": ["35%", "$2.94M"]},
    {"query": "What is the API p99 latency?", "target_doc": "doc_perf_12", "required_keywords": ["320ms", "p99"]},
    {"query": "What is the engineering attrition rate?", "target_doc": "doc_hr_05", "required_keywords": ["8.5%", "attrition"]},
    {"query": "What is the InfiniBand network speed?", "target_doc": "doc_ai_08", "required_keywords": ["800 Gbps", "InfiniBand"]},
    {"query": "What is the full-year net profit margin?", "target_doc": "doc_fin_09", "required_keywords": ["22%", "net profit"]},
    {"query": "What is the CDN cache hit ratio?", "target_doc": "doc_perf_12", "required_keywords": ["94%", "cache"]},
    {"query": "What is the data center PUE ratio?", "target_doc": "doc_infra_02", "required_keywords": ["1.15", "PUE"]},
    {"query": "How many delivery vans operate at Hub Beta?", "target_doc": "doc_logistics_03", "required_keywords": ["8", "hub beta"]},
    {"query": "What is the monthly inventory turnover rate?", "target_doc": "doc_inventory_04", "required_keywords": ["4.2", "turnover"]},
    {"query": "What is the DSAR average response time?", "target_doc": "doc_compliance_10", "required_keywords": ["18 business days", "DSAR"]},
]

# ---------------------------------------------------------------------------
# Reward computation (same improved version)
# ---------------------------------------------------------------------------

def compute_reward(task, docs, resp_1, resp_2):
    reward = 0.0
    retrieved_ids = [d["id"] for d in docs]
    if task["target_doc"] in retrieved_ids:
        reward += 0.25 if retrieved_ids[0] == task["target_doc"] else 0.15
    required = task.get("required_keywords", [])
    if required:
        matches = sum(1 for kw in required if kw.lower() in resp_2.lower())
        reward += 0.25 * (matches / len(required))
    if "<thought>" in resp_1 and "</thought>" in resp_1:
        reward += 0.1
    if "<thought>" in resp_2 and "</thought>" in resp_2:
        reward += 0.1
    if '"name": "bm25_search"' in resp_1 and '"arguments"' in resp_1:
        reward += 0.15
    # Faithfulness: check answer references doc content
    if docs:
        doc_words = set(docs[0]["text"].lower().split())
        ans_words = set(resp_2.lower().split())
        overlap = len(doc_words & ans_words) / max(len(doc_words), 1)
        reward += 0.15 * min(overlap * 3, 1.0)
    return reward

# ---------------------------------------------------------------------------
# Generate rollout
# ---------------------------------------------------------------------------

def generate_rollout(model, tokenizer, task, temperature=0.7):
    """Generate a single on-policy rollout, return (messages, reward) or (None, 0)."""
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["query"]},
    ]
    # Step 1: generate thought + tool call
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=True,
            temperature=temperature, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    resp_1 = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": resp_1})

    # Extract tool call
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_1, re.DOTALL)
    if not match:
        return None, 0.0
    try:
        call = json.loads(match.group(1))
        if call.get("name") != "bm25_search":
            return None, 0.0
        search_query = call.get("arguments", {}).get("query", "")
        top_k = call.get("arguments", {}).get("top_k", 2)
    except (json.JSONDecodeError, AttributeError):
        return None, 0.0
    if not search_query:
        return None, 0.0

    docs = bm25_search(search_query, top_k=min(top_k, 3))
    if not docs:
        return None, 0.0
    obs = json.dumps({"results": docs}, indent=2)
    messages.append({"role": "tool", "content": obs})

    # Step 2: generate synthesis
    prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs2 = tokenizer([prompt2], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out2 = model.generate(
            **inputs2, max_new_tokens=350, do_sample=True,
            temperature=temperature, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    resp_2 = tokenizer.decode(out2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": resp_2})

    reward = compute_reward(task, docs, resp_1, resp_2)
    return messages, reward

# ---------------------------------------------------------------------------
# Generate preference pairs
# ---------------------------------------------------------------------------

def generate_preference_pairs(model, tokenizer, n_samples=8):
    """For each task, generate N rollouts and create chosen/rejected pairs."""
    pairs = []
    for task in TASK_POOL:
        rollouts = []
        for _ in range(n_samples):
            traj, reward = generate_rollout(model, tokenizer, task, temperature=0.8)
            if traj:
                rollouts.append((traj, reward))
            gc.collect()
            torch.cuda.empty_cache()

        if len(rollouts) < 2:
            continue

        rollouts.sort(key=lambda x: x[1], reverse=True)
        best_traj, best_reward = rollouts[0]
        worst_traj, worst_reward = rollouts[-1]

        if best_reward - worst_reward < 0.1:
            continue  # Not enough contrast

        # Format as DPO pair
        chosen_text = tokenizer.apply_chat_template(best_traj, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(worst_traj, tokenize=False)

        pairs.append({
            "prompt": task["query"],
            "chosen": chosen_text,
            "rejected": rejected_text,
            "chosen_reward": best_reward,
            "rejected_reward": worst_reward,
        })
        print(f"  [{task['query'][:40]}...] best={best_reward:.2f} worst={worst_reward:.2f}")

    return pairs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("V1 — DPO Preference Training")
    print("=" * 70)

    # Load SFT model
    print(f"\nLoading SFT model from {SFT_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )

    # Step 1: Generate or load preference pairs
    if os.path.exists(PREFERENCE_DATA_FILE):
        print(f"\n[1/2] Loading existing preference pairs from {PREFERENCE_DATA_FILE}...")
        pairs = []
        with open(PREFERENCE_DATA_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        print(f"  -> Loaded {len(pairs)} preference pairs")
    else:
        print("\n[1/2] Generating preference pairs (8 rollouts per task)...")
        pairs = generate_preference_pairs(model, tokenizer, n_samples=8)
        print(f"  -> Generated {len(pairs)} preference pairs")
        with open(PREFERENCE_DATA_FILE, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"  -> Saved to {PREFERENCE_DATA_FILE}")

    if len(pairs) < 5:
        print("WARNING: Too few preference pairs. Consider more tasks or rollouts.")

    # Step 2: DPO Training
    print("\n[2/2] Running DPO training...")
    try:
        from trl import DPOTrainer, DPOConfig

        # Load reference model (frozen copy of SFT)
        ref_model = AutoModelForCausalLM.from_pretrained(
            SFT_MODEL_PATH, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )

        # Prepare dataset
        dpo_dataset = Dataset.from_list([
            {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in pairs
        ])

        dpo_config = DPOConfig(
            output_dir=OUTPUT_DIR,
            beta=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-7,
            num_train_epochs=1,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            report_to="none",
            save_strategy="epoch",
            logging_steps=5,
            max_length=1024,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=dpo_dataset,
            processing_class=tokenizer,
        )
        trainer.train()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"DPO model saved to {OUTPUT_DIR}")

    except ImportError:
        print("ERROR: trl not installed. Run: pip install trl")
        print("Preference pairs are saved — you can train DPO later.")

    print("\nDPO training finished!")
