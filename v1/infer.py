"""
V1 — Enhanced Inference for Qwen 3B BM25 Agent.

New features:
- Best-of-N sampling (generate N candidates, pick highest reward)
- Self-consistency / majority voting mode
- Standard single-pass mode (default)

Usage:
    python infer.py                                    # default (GRPO model)
    python infer.py --model ./qwen-3b-bm25-sft-v1     # SFT model
    python infer.py --best-of-n 4                      # best-of-4 sampling
"""

import argparse
import json
import re
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpus import SYSTEM_PROMPT, bm25_search

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="V1 Qwen 3B BM25 Agent Inference")
    p.add_argument("--model", type=str, default="./qwen-3b-bm25-grpo-v1")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tool-rounds", type=int, default=3)
    p.add_argument("--best-of-n", type=int, default=1,
                   help="Generate N candidates and pick the best (1 = standard)")
    p.add_argument("--self-consistency", type=int, default=0,
                   help="Majority voting over N samples (0 = disabled)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Tool call extraction & execution
# ---------------------------------------------------------------------------

def extract_tool_call(response):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
        if call.get("name") == "bm25_search":
            return call
    except json.JSONDecodeError:
        pass
    return None


def execute_tool(call):
    args = call.get("arguments", {})
    query = args.get("query", "")
    top_k = args.get("top_k", 2)
    if not query:
        return json.dumps({"error": "Empty search query"})
    results = bm25_search(query, top_k=min(top_k, 5))
    return json.dumps({"results": results}, indent=2)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, messages, max_tokens, temperature):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 0.01),
            top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(model, tokenizer, query, args, verbose=True, temperature_override=None):
    """Run the agent loop. Returns (messages, final_answer)."""
    temp = temperature_override if temperature_override is not None else args.temperature
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    final_answer = ""

    for round_num in range(1, args.max_tool_rounds + 1):
        response = generate_response(model, tokenizer, messages, args.max_tokens, temp)
        messages.append({"role": "assistant", "content": response})

        if verbose:
            print(f"\n{'─'*50}")
            print(f"🤖 Agent (round {round_num}):")
            print(response)

        tool_call = extract_tool_call(response)
        if tool_call is None:
            final_answer = response
            break

        search_query = tool_call["arguments"]["query"]
        if verbose:
            print(f"\n🔧 Executing: bm25_search(query=\"{search_query}\")")
        observation = execute_tool(tool_call)
        messages.append({"role": "tool", "content": observation})

        if verbose:
            results = json.loads(observation).get("results", [])
            print(f"   📄 Found {len(results)} document(s):")
            for doc in results:
                print(f"      - [{doc['id']}] {doc['title']} (score: {doc['score']})")

        final_answer = response

    return messages, final_answer

# ---------------------------------------------------------------------------
# Reward for best-of-N scoring
# ---------------------------------------------------------------------------

def score_trajectory(messages):
    """Simple reward score for ranking candidates."""
    score = 0.0
    for msg in messages:
        if msg["role"] == "assistant":
            c = msg["content"]
            if "<thought>" in c and "</thought>" in c:
                score += 0.2
            if '"name": "bm25_search"' in c:
                score += 0.2
        if msg["role"] == "tool":
            try:
                data = json.loads(msg["content"])
                n_results = len(data.get("results", []))
                score += 0.1 * min(n_results, 3)
            except json.JSONDecodeError:
                pass
    # Bonus for longer final answers (more detailed)
    final_msgs = [m for m in messages if m["role"] == "assistant"]
    if final_msgs:
        last = final_msgs[-1]["content"]
        if len(last) > 100:
            score += 0.2
    return score

# ---------------------------------------------------------------------------
# Best-of-N
# ---------------------------------------------------------------------------

def best_of_n(model, tokenizer, query, args):
    """Generate N candidates and return the best one."""
    n = args.best_of_n
    print(f"\n🎲 Best-of-{n} sampling...")
    candidates = []
    for i in range(n):
        print(f"  Candidate {i+1}/{n}...", end=" ")
        msgs, answer = run_agent(model, tokenizer, query, args, verbose=False, temperature_override=0.7)
        reward = score_trajectory(msgs)
        candidates.append((msgs, answer, reward))
        print(f"score={reward:.2f}")

    # Pick best
    best = max(candidates, key=lambda x: x[2])
    print(f"\n✅ Selected candidate with score {best[2]:.2f}")
    print(f"\n{'─'*50}")
    print("🤖 Agent (best-of-N):")
    print(best[1])
    return best[0]

# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------

def self_consistency(model, tokenizer, query, args):
    """Generate N answers and pick the most common one."""
    n = args.self_consistency
    print(f"\n🗳️ Self-consistency voting (N={n})...")
    answers = []
    for i in range(n):
        print(f"  Sample {i+1}/{n}...", end=" ")
        _, answer = run_agent(model, tokenizer, query, args, verbose=False, temperature_override=0.7)
        # Extract key facts (numbers, percentages)
        facts = set(re.findall(r'\$[\d,]+|\d+\.?\d*%|\d[\d,]+', answer))
        answers.append((answer, frozenset(facts)))
        print(f"facts={facts}")

    # Vote on fact sets
    fact_counter = Counter(fs for _, fs in answers)
    most_common_facts = fact_counter.most_common(1)[0][0]

    # Return the answer whose facts match the majority
    for answer, facts in answers:
        if facts == most_common_facts:
            print(f"\n✅ Majority answer (appeared {fact_counter[most_common_facts]}x):")
            print(f"\n{'─'*50}")
            print("🤖 Agent (self-consistency):")
            print(answer)
            return answer
    return answers[0][0]

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)

    mode = "standard"
    if args.best_of_n > 1:
        mode = f"best-of-{args.best_of_n}"
    elif args.self_consistency > 1:
        mode = f"self-consistency-{args.self_consistency}"

    print(f"\n{'='*60}")
    print(f"  V1 — Qwen 3B BM25 Agent (mode: {mode})")
    print(f"  Type your query and press Enter.")
    print(f"  Type 'quit' to stop, 'clear' to reset.")
    print(f"{'='*60}")

    while True:
        try:
            query = input("\n❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "clear":
            print("Conversation cleared.")
            continue

        if args.best_of_n > 1:
            best_of_n(model, tokenizer, query, args)
        elif args.self_consistency > 1:
            self_consistency(model, tokenizer, query, args)
        else:
            run_agent(model, tokenizer, query, args)


if __name__ == "__main__":
    main()
