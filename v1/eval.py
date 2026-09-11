"""
V1 — Evaluation Framework for the Qwen 3B BM25 Agent.

Runs a held-out test set and measures:
- Retrieval Precision@1 (did it find the right document?)
- Keyword F1 (did the answer contain required facts?)
- Format compliance (proper thought tags + JSON tool calls)
- Overall reward score

Outputs results to eval_results.json for experiment tracking.
"""

import json
import re
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpus import SYSTEM_PROMPT, bm25_search

# ---------------------------------------------------------------------------
# Held-out test set (distinct from training tasks)
# ---------------------------------------------------------------------------

TEST_SET = [
    {"query": "What is the operating margin from the H1 financial report?",
     "target_doc": "doc_fin_01", "required_keywords": ["28%", "operating margin"]},
    {"query": "What is the average server uptime?",
     "target_doc": "doc_infra_02", "required_keywords": ["99.97%", "uptime"]},
    {"query": "How many transport trucks operate at Hub Alpha?",
     "target_doc": "doc_logistics_03", "required_keywords": ["14", "trucks"]},
    {"query": "How many units of product Beta are in Warehouse A?",
     "target_doc": "doc_inventory_04", "required_keywords": ["1,250", "beta"]},
    {"query": "How many frontend engineers does the company have?",
     "target_doc": "doc_hr_05", "required_keywords": ["48", "frontend"]},
    {"query": "How long must security audit logs be stored?",
     "target_doc": "doc_sec_06", "required_keywords": ["365", "days"]},
    {"query": "How many users were impacted by the Q3 downtime?",
     "target_doc": "doc_incident_07", "required_keywords": ["12,000", "users"]},
    {"query": "What is the total NVMe storage capacity in the AI cluster?",
     "target_doc": "doc_ai_08", "required_keywords": ["1.2 PB", "NVMe"]},
    {"query": "What was Q3 revenue?",
     "target_doc": "doc_fin_09", "required_keywords": ["$710,000", "q3"]},
    {"query": "Was the ISO 27001 certification renewed?",
     "target_doc": "doc_compliance_10", "required_keywords": ["ISO 27001", "renewed"]},
    {"query": "How much of the IT budget goes to security?",
     "target_doc": "doc_budget_11", "required_keywords": ["10%", "$840K"]},
    {"query": "What is the API p50 latency?",
     "target_doc": "doc_perf_12", "required_keywords": ["145ms", "p50"]},
    {"query": "What is the R&D spending as a percentage of revenue?",
     "target_doc": "doc_fin_01", "required_keywords": ["18%", "R&D"]},
    {"query": "How many DevOps specialists are in engineering?",
     "target_doc": "doc_hr_05", "required_keywords": ["24", "DevOps"]},
    {"query": "What version resolved the Q3 API incident?",
     "target_doc": "doc_incident_07", "required_keywords": ["v2.4.1"]},
    {"query": "How many data racks are in the US-East data center?",
     "target_doc": "doc_infra_02", "required_keywords": ["8", "racks"]},
    {"query": "What is the peak API traffic?",
     "target_doc": "doc_perf_12", "required_keywords": ["8,500", "requests"]},
    {"query": "What is the fuel consumption per 100 km for the fleet?",
     "target_doc": "doc_logistics_03", "required_keywords": ["32", "liters"]},
    {"query": "What is the budget allocation for cloud services?",
     "target_doc": "doc_budget_11", "required_keywords": ["15%", "$1.26M"]},
    {"query": "How many concurrent jobs can the AI cluster handle?",
     "target_doc": "doc_ai_08", "required_keywords": ["64", "concurrent"]},
]

# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def run_agent_silent(model, tokenizer, query, max_tokens=512, temperature=0.1):
    """Run agent loop silently, return (messages, extracted_docs, responses)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    all_docs = []
    responses = []

    for _ in range(3):  # max 3 rounds
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 0.01), top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": resp})
        responses.append(resp)

        # Check for tool call
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp, re.DOTALL)
        if not match:
            break
        try:
            call = json.loads(match.group(1))
            if call.get("name") != "bm25_search":
                break
            sq = call.get("arguments", {}).get("query", "")
            tk = call.get("arguments", {}).get("top_k", 2)
        except (json.JSONDecodeError, AttributeError):
            break
        if not sq:
            break
        docs = bm25_search(sq, top_k=min(tk, 3))
        all_docs.extend(docs)
        obs = json.dumps({"results": docs}, indent=2)
        messages.append({"role": "tool", "content": obs})

    return messages, all_docs, responses


def evaluate_single(model, tokenizer, task):
    """Evaluate a single test case. Returns a dict of metrics."""
    messages, docs, responses = run_agent_silent(model, tokenizer, task["query"])

    result = {
        "query": task["query"],
        "target_doc": task["target_doc"],
        "retrieval_p1": False,
        "retrieval_hit": False,
        "keyword_hits": 0,
        "keyword_total": len(task["required_keywords"]),
        "keyword_f1": 0.0,
        "has_thought": False,
        "has_tool_call": False,
        "format_ok": False,
    }

    # Retrieval precision
    retrieved_ids = [d["id"] for d in docs]
    if retrieved_ids:
        result["retrieval_p1"] = (retrieved_ids[0] == task["target_doc"])
        result["retrieval_hit"] = (task["target_doc"] in retrieved_ids)

    # Keyword coverage
    full_response = " ".join(responses).lower()
    hits = sum(1 for kw in task["required_keywords"] if kw.lower() in full_response)
    result["keyword_hits"] = hits
    result["keyword_f1"] = hits / max(result["keyword_total"], 1)

    # Format compliance
    if responses:
        result["has_thought"] = any("<thought>" in r and "</thought>" in r for r in responses)
        result["has_tool_call"] = any('"name": "bm25_search"' in r for r in responses)
        result["format_ok"] = result["has_thought"] and result["has_tool_call"]

    return result


def run_evaluation(model, tokenizer):
    """Run full evaluation suite."""
    results = []
    for i, task in enumerate(TEST_SET):
        print(f"  [{i+1}/{len(TEST_SET)}] {task['query'][:50]}...", end=" ")
        result = evaluate_single(model, tokenizer, task)
        results.append(result)
        status = "✓" if result["retrieval_hit"] and result["keyword_f1"] > 0.5 else "✗"
        print(f"{status} (P@1={result['retrieval_p1']}, KW={result['keyword_f1']:.0%})")

    # Aggregate metrics
    n = len(results)
    metrics = {
        "total_tests": n,
        "retrieval_p1": sum(r["retrieval_p1"] for r in results) / n,
        "retrieval_hit": sum(r["retrieval_hit"] for r in results) / n,
        "keyword_f1": sum(r["keyword_f1"] for r in results) / n,
        "format_compliance": sum(r["format_ok"] for r in results) / n,
        "thought_rate": sum(r["has_thought"] for r in results) / n,
        "tool_call_rate": sum(r["has_tool_call"] for r in results) / n,
    }
    return metrics, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen 3B BM25 Agent")
    parser.add_argument("--model", type=str, default="./qwen-3b-bm25-grpo-v1",
                        help="Path to model directory")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    print("=" * 60)
    print(f"V1 — Agent Evaluation")
    print(f"Model: {args.model}")
    print("=" * 60)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    print(f"\nRunning {len(TEST_SET)} test cases...\n")
    metrics, results = run_evaluation(model, tokenizer)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Retrieval P@1:    {metrics['retrieval_p1']:.1%}")
    print(f"  Retrieval Hit:    {metrics['retrieval_hit']:.1%}")
    print(f"  Keyword F1:       {metrics['keyword_f1']:.1%}")
    print(f"  Format Compliance:{metrics['format_compliance']:.1%}")
    print(f"  Thought Rate:     {metrics['thought_rate']:.1%}")
    print(f"  Tool Call Rate:   {metrics['tool_call_rate']:.1%}")

    # Save results
    output = {"metrics": metrics, "details": results, "model": args.model}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
