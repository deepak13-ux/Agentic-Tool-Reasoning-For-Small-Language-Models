"""
V1 — GRPO (Group Relative Policy Optimization) Training.

After SFT (or DPO), further improves the model through group-relative RL:
1. For each task, sample G=4 rollouts
2. Compute reward for each
3. Compute advantage = (reward - mean) / std within the group
4. Apply policy gradient weighted by advantage

Inspired by DeepSeek-R1's approach.
"""

import os
import gc
import re
import json
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpus import CORPUS, SYSTEM_PROMPT, bm25_search

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Try DPO model first, fall back to SFT
MODEL_PATH = "./qwen-3b-bm25-dpo-v1"
_config_path = os.path.join(MODEL_PATH, "config.json")
if not os.path.exists(_config_path):
    print(f"WARNING: {MODEL_PATH} has no valid config.json, falling back to SFT model.")
    MODEL_PATH = "./qwen-3b-bm25-sft-v1"
OUTPUT_DIR = "./qwen-3b-bm25-grpo-v1"

GROUP_SIZE = 4           # Number of rollouts per task
NUM_ITERATIONS = 8
TASKS_PER_ITER = 15
MIN_GROUP_STD = 0.05     # Skip groups with no reward variance
LR = 3e-6
MAX_GRAD_NORM = 1.0
KL_COEFF = 0.01          # KL penalty coefficient

# ---------------------------------------------------------------------------
# Task pool with ground-truth
# ---------------------------------------------------------------------------

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
# Model setup
# ---------------------------------------------------------------------------

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
)

# ---------------------------------------------------------------------------
# Improved reward function
# ---------------------------------------------------------------------------

def compute_reward(task, docs, resp_1, resp_2):
    """Enhanced reward with faithfulness and query quality scoring."""
    reward = 0.0
    retrieved_ids = [d["id"] for d in docs]

    # 1. Document retrieval (0.25) — bonus for rank-1
    if task["target_doc"] in retrieved_ids:
        reward += 0.25 if retrieved_ids[0] == task["target_doc"] else 0.15

    # 2. Keyword coverage (0.25)
    required = task.get("required_keywords", [])
    if required:
        matches = sum(1 for kw in required if kw.lower() in resp_2.lower())
        reward += 0.25 * (matches / len(required))

    # 3. Thought tag presence (0.15)
    if "<thought>" in resp_1 and "</thought>" in resp_1:
        reward += 0.075
    if "<thought>" in resp_2 and "</thought>" in resp_2:
        reward += 0.075

    # 4. Tool call format (0.15)
    if '"name": "bm25_search"' in resp_1 and '"arguments"' in resp_1:
        reward += 0.15

    # 5. Faithfulness — answer uses doc content (0.20)
    if docs:
        doc_words = set(docs[0]["text"].lower().split())
        ans_words = set(resp_2.lower().split())
        overlap = len(doc_words & ans_words) / max(len(doc_words), 1)
        reward += 0.20 * min(overlap * 3, 1.0)

    return round(reward, 4)

# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def rollout(task, temperature=0.8):
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["query"]},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=True,
            temperature=temperature, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    resp_1 = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": resp_1})

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_1, re.DOTALL)
    if not match:
        return None, 0.0
    try:
        call = json.loads(match.group(1))
        if call.get("name") != "bm25_search":
            return None, 0.0
        sq = call.get("arguments", {}).get("query", "")
        tk = call.get("arguments", {}).get("top_k", 2)
    except (json.JSONDecodeError, AttributeError):
        return None, 0.0
    if not sq:
        return None, 0.0

    docs = bm25_search(sq, top_k=min(tk, 3))
    if not docs:
        return None, 0.0
    obs = json.dumps({"results": docs}, indent=2)
    messages.append({"role": "tool", "content": obs})

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
# GRPO Training Loop
# ---------------------------------------------------------------------------

def run_grpo():
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_updates = 0

    print("=" * 70)
    print("V1 — GRPO Self-Improvement Loop")
    print(f"  Iterations: {NUM_ITERATIONS}, Tasks/iter: {TASKS_PER_ITER}")
    print(f"  Group size: {GROUP_SIZE}, LR: {LR}")
    print("=" * 70)

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\n{'─'*50}")
        print(f"[Iteration {iteration}/{NUM_ITERATIONS}]")

        tasks = random.choices(TASK_POOL, k=TASKS_PER_ITER)
        iter_updates = 0
        iter_rewards = []

        for task in tasks:
            # Generate G rollouts per task
            group = []
            for _ in range(GROUP_SIZE):
                traj, reward = rollout(task, temperature=0.8)
                if traj:
                    group.append((traj, reward))
                    iter_rewards.append(reward)

            if len(group) < 2:
                continue

            # Compute group-relative advantage
            rewards = np.array([r for _, r in group])
            mean_r, std_r = rewards.mean(), rewards.std()
            if std_r < MIN_GROUP_STD:
                continue  # No signal in this group

            advantages = (rewards - mean_r) / (std_r + 1e-8)

            # Policy update with advantage weighting
            model.train()
            for (traj, _), adv in zip(group, advantages):
                text = tokenizer.apply_chat_template(traj, tokenize=False)
                tokens = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
                tokens["labels"] = tokens["input_ids"].clone()

                optimizer.zero_grad()
                outputs = model(**tokens)
                # Advantage-weighted loss (negative advantage = push away)
                loss = -adv * outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                iter_updates += 1

            gc.collect()
            torch.cuda.empty_cache()

        total_updates += iter_updates
        avg_r = np.mean(iter_rewards) if iter_rewards else 0
        print(f"  Avg reward: {avg_r:.3f}, Updates: {iter_updates}, Total: {total_updates}")

        # Save checkpoint every 2 iterations
        if iteration % 2 == 0:
            ckpt = f"{OUTPUT_DIR}/checkpoint-iter-{iteration}"
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Saved checkpoint to {ckpt}")

    # Save final
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nGRPO model saved to '{OUTPUT_DIR}'")
    print(f"Total gradient updates: {total_updates}")


if __name__ == "__main__":
    run_grpo()
