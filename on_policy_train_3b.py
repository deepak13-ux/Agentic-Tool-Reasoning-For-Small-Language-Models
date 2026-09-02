"""
On-Policy RL Training for Qwen 3B BM25 Tool-Use Agent.

After SFT, this script further improves the model through self-play:
1. Model generates its own trajectories (rollouts)
2. A rule-based reward verifier scores each trajectory
3. Only high-reward trajectories are used for gradient updates

Designed for 48GB GPU.
"""

import os
import re
import gc
import json
import random
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpus import CORPUS, SYSTEM_PROMPT, bm25_search

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# 1. TASK POOL — Diverse queries with verifiable ground-truth
# ---------------------------------------------------------------------------

TASK_POOL = [
    # Single-step tasks
    {"query": "Find the total revenue in Q1 and Q2 from the financial report.", "target_doc": "doc_fin_01", "required_keywords": ["$450,000", "$620,000"]},
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
    {"query": "What is the DSAR average response time?", "target_doc": "doc_compliance_10", "required_keywords": ["18 business days", "DSAR"]},
    {"query": "What is the CDN cache hit ratio?", "target_doc": "doc_perf_12", "required_keywords": ["94%", "cache"]},
    {"query": "What is the data center PUE ratio?", "target_doc": "doc_infra_02", "required_keywords": ["1.15", "PUE"]},
    {"query": "How many delivery vans operate at Hub Beta?", "target_doc": "doc_logistics_03", "required_keywords": ["8", "hub beta"]},
    {"query": "What is the monthly inventory turnover rate?", "target_doc": "doc_inventory_04", "required_keywords": ["4.2", "turnover"]},
]

# ---------------------------------------------------------------------------
# 2. MODEL SETUP
# ---------------------------------------------------------------------------

SFT_MODEL_PATH = "./qwen-3b-bm25-sft"
OUTPUT_DIR = "./qwen-3b-bm25-on-policy"

print(f"Loading SFT model from {SFT_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

# ---------------------------------------------------------------------------
# 3. ON-POLICY ROLLOUT GENERATION
# ---------------------------------------------------------------------------

def rollout_trajectory(task: dict, temperature: float = 0.7):
    """
    Samples an on-policy trajectory from the current model π_θ.
    Returns (messages, reward) or (None, 0.0) on failure.
    """
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["query"]},
    ]

    # --- Step 1: Model generates thought + tool call ---
    prompt_1 = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs_1 = tokenizer([prompt_1], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_1 = model.generate(
            **inputs_1,
            max_new_tokens=256,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp_1 = tokenizer.decode(
        out_1[0][inputs_1.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    messages.append({"role": "assistant", "content": resp_1})

    # --- Extract Tool Call ---
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_1, re.DOTALL)
    if not match:
        return None, 0.0  # Failed to produce tool call format

    try:
        call = json.loads(match.group(1))
        search_query = call.get("arguments", {}).get("query", "")
        top_k = call.get("arguments", {}).get("top_k", 2)
        # Verify tool name
        if call.get("name") != "bm25_search":
            return None, 0.0
    except (json.JSONDecodeError, AttributeError):
        return None, 0.0

    if not search_query:
        return None, 0.0

    # --- Environment Step: Execute BM25 search ---
    docs = bm25_search(search_query, top_k=min(top_k, 3))
    if not docs:
        return None, 0.0

    obs = json.dumps({"results": docs}, indent=2)
    messages.append({"role": "tool", "content": obs})

    # --- Step 2: Model generates thought + final synthesis ---
    prompt_2 = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs_2 = tokenizer([prompt_2], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_2 = model.generate(
            **inputs_2,
            max_new_tokens=350,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp_2 = tokenizer.decode(
        out_2[0][inputs_2.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    messages.append({"role": "assistant", "content": resp_2})

    # --- Reward Computation ---
    reward = compute_reward(task, docs, resp_1, resp_2)
    return messages, reward


def compute_reward(task: dict, docs: list, resp_1: str, resp_2: str) -> float:
    """
    Rule-based reward verifier with 4 components:

    1. Document retrieval accuracy  (0.0 - 0.3)
    2. Keyword coverage in answer   (0.0 - 0.3)
    3. Thought tag presence          (0.0 - 0.2)
    4. Tool call format correctness  (0.0 - 0.2)
    """
    reward = 0.0

    # 1. Did it retrieve the target document?
    retrieved_ids = [d["id"] for d in docs]
    if task["target_doc"] in retrieved_ids:
        reward += 0.3

    # 2. Keyword coverage in final response
    required = task.get("required_keywords", [])
    if required:
        matches = sum(1 for kw in required if kw.lower() in resp_2.lower())
        keyword_ratio = matches / len(required)
        reward += 0.3 * keyword_ratio

    # 3. Thought tags present in both responses
    if "<thought>" in resp_1 and "</thought>" in resp_1:
        reward += 0.1
    if "<thought>" in resp_2 and "</thought>" in resp_2:
        reward += 0.1

    # 4. Correct tool call format
    if '"name": "bm25_search"' in resp_1 and '"arguments"' in resp_1:
        reward += 0.2

    return reward


# ---------------------------------------------------------------------------
# 4. ON-POLICY TRAINING LOOP
# ---------------------------------------------------------------------------

def run_on_policy_training(
    num_iterations: int = 8,
    rollouts_per_iter: int = 20,
    min_reward: float = 0.7,
    lr: float = 5e-6,
    max_grad_norm: float = 1.0,
):
    """Main on-policy self-improvement loop."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("=" * 70)
    print("On-Policy Self-Improvement Loop for Qwen 3B")
    print(f"  Iterations: {num_iterations}")
    print(f"  Rollouts per iteration: {rollouts_per_iter}")
    print(f"  Min reward threshold: {min_reward}")
    print(f"  Learning rate: {lr}")
    print("=" * 70)

    total_updates = 0

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'─'*50}")
        print(f"[Iteration {iteration}/{num_iterations}] Collecting On-Policy Rollouts...")

        on_policy_buffer = []
        total_reward = 0.0
        attempted = 0

        # Sample tasks randomly for diversity
        tasks_this_iter = random.choices(TASK_POOL, k=rollouts_per_iter)

        for task in tasks_this_iter:
            attempted += 1
            traj, reward = rollout_trajectory(task, temperature=0.7)
            total_reward += reward

            if traj and reward >= min_reward:
                on_policy_buffer.append((traj, reward))

            # Memory cleanup every few rollouts
            if attempted % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_reward = total_reward / max(attempted, 1)
        print(f"  -> Attempted: {attempted}, Accepted: {len(on_policy_buffer)}")
        print(f"  -> Average reward: {avg_reward:.3f}")

        if not on_policy_buffer:
            print("  [Warning]: No rollouts met threshold. Skipping update.")
            continue

        # --- Policy Update Step ---
        model.train()
        total_loss = 0.0

        for traj, reward in on_policy_buffer:
            text = tokenizer.apply_chat_template(traj, tokenize=False)
            tokens = tokenizer(
                text, return_tensors="pt", max_length=1024, truncation=True
            ).to(model.device)
            tokens["labels"] = tokens["input_ids"].clone()

            optimizer.zero_grad()
            outputs = model(**tokens)
            # Weight loss by reward — higher reward = stronger gradient signal
            loss = outputs.loss * reward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_updates += 1

        avg_loss = total_loss / len(on_policy_buffer)
        print(f"  -> Policy updated. Mean weighted loss: {avg_loss:.4f}")
        print(f"  -> Total gradient updates so far: {total_updates}")

        # Save checkpoint every 2 iterations
        if iteration % 2 == 0:
            ckpt_dir = f"{OUTPUT_DIR}/checkpoint-iter-{iteration}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  -> Saved checkpoint to {ckpt_dir}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    print(f"\n{'='*70}")
    print("On-Policy Self-Training completed!")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved final on-policy model to '{OUTPUT_DIR}'")
    print(f"Total gradient updates: {total_updates}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_on_policy_training()
