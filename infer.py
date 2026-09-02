"""
Interactive Inference for the Qwen 3B BM25 Tool-Use Agent.

Runs a REPL loop where the user types queries and the agent:
1. Thinks (reasoning in <thought> tags)
2. Calls bm25_search if needed
3. Receives tool results
4. Synthesizes a final answer

Usage:
    python infer.py                          # Uses on-policy model
    python infer.py --model ./qwen-3b-bm25-sft  # Uses SFT model
"""

import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from corpus import SYSTEM_PROMPT, bm25_search

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen 3B BM25 Agent Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="./qwen-3b-bm25-on-policy",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per step",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=3,
        help="Maximum tool call rounds before forcing final answer",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tool call extraction and execution
# ---------------------------------------------------------------------------

def extract_tool_call(response: str):
    """Extract a JSON tool call from the model response."""
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


def execute_tool(call: dict) -> str:
    """Execute a tool call and return the observation string."""
    args = call.get("arguments", {})
    query = args.get("query", "")
    top_k = args.get("top_k", 2)

    if not query:
        return json.dumps({"error": "Empty search query"})

    results = bm25_search(query, top_k=min(top_k, 5))
    return json.dumps({"results": results}, indent=2)


# ---------------------------------------------------------------------------
# Generation step
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, messages, max_tokens, temperature):
    """Generate a single response from the model."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(model, tokenizer, query: str, args):
    """Run the full agent loop for a single query."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for round_num in range(1, args.max_tool_rounds + 1):
        # Generate model response
        response = generate_response(
            model, tokenizer, messages, args.max_tokens, args.temperature
        )
        messages.append({"role": "assistant", "content": response})

        # Display the response
        print(f"\n{'─'*50}")
        print(f"🤖 Agent (round {round_num}):")
        print(response)

        # Check for tool call
        tool_call = extract_tool_call(response)
        if tool_call is None:
            # No tool call = final answer
            break

        # Execute tool
        search_query = tool_call["arguments"]["query"]
        print(f"\n🔧 Executing: bm25_search(query=\"{search_query}\")")
        observation = execute_tool(tool_call)
        messages.append({"role": "tool", "content": observation})

        # Show observation summary
        results = json.loads(observation).get("results", [])
        print(f"   📄 Found {len(results)} document(s):")
        for doc in results:
            print(f"      - [{doc['id']}] {doc['title']} (score: {doc['score']})")

    return messages


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)

    print("\n" + "=" * 60)
    print("  Qwen 3B BM25 Tool-Use Agent")
    print("  Type your query and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'clear' to reset conversation.")
    print("=" * 60)

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

        run_agent(model, tokenizer, query, args)


if __name__ == "__main__":
    main()
