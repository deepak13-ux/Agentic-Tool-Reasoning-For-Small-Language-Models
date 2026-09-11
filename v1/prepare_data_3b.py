"""
V1 — Enhanced SFT data preparation for Qwen 3B BM25 tool-use agent.

Improvements:
- Targets 2,000+ trajectories (up from ~500)
- Analytical chain-of-thought (not generic)
- Error-recovery trajectories (bad search -> retry)
- Adversarial / out-of-corpus trajectories
- More aggressive query paraphrasing
"""

import json
import random
import hashlib
from typing import Any, Dict, List
from corpus import CORPUS, SYSTEM_PROMPT, bm25_search

random.seed(42)
OUTPUT_FILE = "train_data_3b.jsonl"

# ---------------------------------------------------------------------------
# Thought builders — V1: Specific, analytical reasoning
# ---------------------------------------------------------------------------

def _thought(text: str) -> str:
    return f"<thought>{text}</thought>"

def _tool_call(query: str, top_k: int = 2) -> str:
    call = {"name": "bm25_search", "arguments": {"query": query, "top_k": top_k}}
    return '```json\n' + json.dumps(call) + '\n```'

# Specific reasoning templates for different document types
REASONING_MAP = {
    "doc_fin": "This is a financial query. I should search for revenue, profit, or expense data in the financial reports using specific fiscal period keywords.",
    "doc_infra": "This relates to infrastructure. I'll search for data center specs, server counts, or uptime metrics in the infrastructure report.",
    "doc_logistics": "This is about logistics operations. I need fleet, transport, or delivery data from the logistics plan.",
    "doc_inventory": "This asks about inventory. I should search for warehouse stock levels, product quantities, or turnover data.",
    "doc_hr": "This is an HR question. I'll look for headcount, team sizes, or attrition data in the engineering HR report.",
    "doc_sec": "This concerns security policy. I need to find authentication, encryption, or data retention policies.",
    "doc_incident": "This asks about an incident. I should search the post-mortem for root cause, impact, and resolution details.",
    "doc_ai": "This relates to AI infrastructure. I'll search for GPU specs, cluster capacity, or utilization data.",
    "doc_compliance": "This is about compliance. I need audit results, certification status, or regulatory metrics.",
    "doc_budget": "This is a budget question. I should search for IT budget allocations, spending breakdowns, or utilization rates.",
    "doc_perf": "This asks about performance metrics. I'll search for latency, error rates, or throughput data.",
}

def _get_reasoning(doc_id_hint: str, query: str) -> str:
    for prefix, reasoning in REASONING_MAP.items():
        if doc_id_hint.startswith(prefix):
            return reasoning
    return f"The user is asking about '{query}'. I need to search the internal documents for relevant information."

def _get_synthesis_reasoning(doc_title: str, query: str) -> str:
    return (
        f"The search returned relevant data from '{doc_title}'. "
        f"I can now extract the specific information the user asked about and present it clearly."
    )

# ---------------------------------------------------------------------------
# Single-step templates
# ---------------------------------------------------------------------------

SINGLE_STEP_TEMPLATES = [
    ("What is the total revenue in {period} from the financial report?", "total revenue {period} financial report", "doc_fin"),
    ("How many server nodes are in the {dc} data center?", "server nodes {dc} data center", "doc_infra"),
    ("What is the maximum payload capacity of the logistics fleet trucks?", "logistics fleet truck payload capacity", "doc_logistics"),
    ("How many units of product {prod} are in Warehouse {wh}?", "warehouse {wh} product {prod} inventory units", "doc_inventory"),
    ("What is the total engineering headcount?", "engineering headcount total employees", "doc_hr"),
    ("What is the password rotation policy?", "password rotation policy security", "doc_sec"),
    ("What caused the Q3 API incident?", "Q3 API incident downtime root cause", "doc_incident"),
    ("How many GPUs does the AI training cluster have?", "AI cluster GPU count specifications", "doc_ai"),
    ("What was Q4 revenue?", "Q4 revenue financial report", "doc_fin"),
    ("What were the SOC 2 audit results?", "SOC 2 audit compliance results", "doc_compliance"),
    ("What is the total IT budget for the fiscal year?", "IT budget total fiscal year allocation", "doc_budget"),
    ("What is the API p99 latency?", "API p99 latency performance metrics", "doc_perf"),
    ("How many regional offices does engineering have?", "engineering regional offices locations", "doc_hr"),
    ("What encryption standard is used for data at rest?", "encryption data at rest standard security", "doc_sec"),
    ("What is the CDN cache hit ratio?", "CDN cache hit ratio performance", "doc_perf"),
    ("What is the InfiniBand speed in the AI cluster?", "InfiniBand speed AI cluster network", "doc_ai"),
    ("How many delivery vans operate at Hub Beta?", "delivery vans Hub Beta logistics fleet", "doc_logistics"),
    ("What is the PUE ratio of the data center?", "PUE power usage effectiveness data center", "doc_infra"),
    ("What is the standby generator capacity?", "standby generator capacity data center", "doc_infra"),
    ("What is the monthly fleet maintenance cost?", "fleet maintenance cost monthly logistics", "doc_logistics"),
    ("What is the total inventory value across warehouses?", "total inventory value warehouses", "doc_inventory"),
    ("How many users were affected by the Q3 incident?", "Q3 incident affected users API", "doc_incident"),
    ("What is the annual attrition rate for engineering?", "attrition rate engineering annual", "doc_hr"),
    ("What is the GDPR compliance status?", "GDPR compliance review findings", "doc_compliance"),
    ("What percentage of the IT budget goes to cloud services?", "IT budget cloud services percentage", "doc_budget"),
    ("What is the average GPU utilization in the AI cluster?", "GPU utilization average AI cluster", "doc_ai"),
    ("What is the full-year revenue figure?", "full year revenue total annual", "doc_fin"),
    ("What is the DSAR average response time?", "DSAR response time average days", "doc_compliance"),
    ("What is the database query p95 latency?", "database query p95 latency", "doc_perf"),
    ("What is the peak traffic in requests per second?", "peak traffic requests per second", "doc_perf"),
    ("What is the capital expenditure total?", "capital expenditure total capex", "doc_fin"),
    ("How many concurrent training jobs does the AI cluster support?", "concurrent training jobs AI cluster", "doc_ai"),
    ("What is the error rate across all API endpoints?", "error rate endpoints API performance", "doc_perf"),
    ("What is the VPN authentication requirement?", "VPN authentication remote access requirement", "doc_sec"),
    ("How long must security audit logs be retained?", "security audit logs retention period", "doc_sec"),
    ("What release fixed the Q3 API incident?", "release fix Q3 API incident version", "doc_incident"),
    ("What is the operating margin?", "operating margin percentage financial", "doc_fin"),
    ("How much does R&D spending account for?", "R&D spending percentage revenue", "doc_fin"),
    ("What is the net profit for H1?", "net profit H1 first half", "doc_fin"),
    ("What is the fuel consumption of the logistics fleet?", "fuel consumption liters logistics fleet", "doc_logistics"),
]

# ---------------------------------------------------------------------------
# Multi-step templates
# ---------------------------------------------------------------------------

MULTI_STEP_TEMPLATES = [
    {"query": "Compare the revenue in Q1 versus Q4.",
     "searches": [("Q1 revenue financial report", "doc_fin_01"), ("Q4 revenue financial report", "doc_fin_09")]},
    {"query": "Compare the number of server nodes to the number of GPUs in the AI cluster.",
     "searches": [("server nodes data center total", "doc_infra_02"), ("AI cluster GPU count", "doc_ai_08")]},
    {"query": "What is the engineering headcount relative to the IT budget for salaries?",
     "searches": [("engineering headcount total", "doc_hr_05"), ("IT budget engineering salaries allocation", "doc_budget_11")]},
    {"query": "Compare H1 net profit with the full-year net profit margin.",
     "searches": [("H1 net profit revenue", "doc_fin_01"), ("full year net profit margin", "doc_fin_09")]},
    {"query": "How does the API latency compare to the database query latency?",
     "searches": [("API p50 p99 latency performance", "doc_perf_12"), ("database query p95 latency", "doc_perf_12")]},
    {"query": "Compare the security audit log retention period with the GDPR DSAR response requirement.",
     "searches": [("security audit logs retention days", "doc_sec_06"), ("GDPR DSAR response time days", "doc_compliance_10")]},
    {"query": "What is the relationship between fleet size and warehouse inventory value?",
     "searches": [("logistics fleet trucks vans total", "doc_logistics_03"), ("warehouse inventory total value", "doc_inventory_04")]},
    {"query": "How does the IT infrastructure budget allocation compare with actual server infrastructure costs?",
     "searches": [("IT budget infrastructure allocation percentage", "doc_budget_11"), ("server data center power watts", "doc_infra_02")]},
    {"query": "Compare the Q3 incident impact with the overall API performance metrics.",
     "searches": [("Q3 incident downtime affected users", "doc_incident_07"), ("API error rate latency performance", "doc_perf_12")]},
    {"query": "What fraction of the total budget is spent on the AI cluster versus cloud services?",
     "searches": [("AI cluster specifications GPU", "doc_ai_08"), ("IT budget cloud services percentage allocation", "doc_budget_11")]},
]

# ---------------------------------------------------------------------------
# No-tool templates (general knowledge)
# ---------------------------------------------------------------------------

NO_TOOL_TEMPLATES = [
    {"query": "Explain the difference between a hash table and a binary search tree.", "answer": "A hash table stores key-value pairs using a hash function for O(1) average lookup, insert, and delete. A binary search tree (BST) organizes data hierarchically: each node's left children are smaller and right children are larger, giving O(log n) average operations. Hash tables are faster for lookups but don't maintain order, while BSTs support ordered traversal and range queries efficiently."},
    {"query": "What are the time complexities of quicksort and mergesort?", "answer": "Quicksort has O(n log n) average time complexity but O(n^2) worst case. Mergesort guarantees O(n log n) in all cases but requires O(n) extra space. Quicksort is typically faster in practice due to better cache locality."},
    {"query": "Describe how a blockchain achieves consensus.", "answer": "Blockchains use consensus mechanisms like Proof of Work (PoW) or Proof of Stake (PoS). In PoW, miners compete to solve cryptographic puzzles. In PoS, validators stake cryptocurrency as collateral. Both ensure all nodes agree on the transaction history without a central authority."},
    {"query": "Explain the concept of memoization in dynamic programming.", "answer": "Memoization is a top-down optimization where you cache results of expensive function calls and return cached results for repeated inputs. It converts exponential recursive solutions into polynomial time. For example, Fibonacci(n) goes from O(2^n) to O(n)."},
    {"query": "What is the CAP theorem?", "answer": "The CAP theorem states that a distributed system can guarantee at most two of: Consistency (all nodes see the same data), Availability (every request gets a response), and Partition tolerance (system works despite network failures). Since partitions are inevitable, real systems choose between CP (like HBase) and AP (like Cassandra)."},
    {"query": "How does backpropagation work in neural networks?", "answer": "Backpropagation computes the gradient of the loss with respect to each weight using the chain rule, layer by layer from output to input. A forward pass computes predictions, then the backward pass propagates error gradients. These gradients are used by an optimizer like Adam to update weights."},
    {"query": "What is the difference between TCP and UDP?", "answer": "TCP is connection-oriented with reliable, ordered delivery and error checking. UDP is connectionless, offering faster but unreliable delivery. TCP is used for web/email/files; UDP for streaming, gaming, and DNS."},
    {"query": "What is the difference between a process and a thread?", "answer": "A process is an independent execution unit with its own memory space. A thread is a lightweight unit within a process that shares memory with other threads. Thread context switching is faster but shared memory makes threads prone to race conditions."},
]

# ---------------------------------------------------------------------------
# Missing-info / clarification templates
# ---------------------------------------------------------------------------

MISSING_INFO_TEMPLATES = [
    {"query": "What is the latest security policy?", "response": "I'd be happy to help, but could you clarify which specific security policy you're asking about? For example:\n- Password rotation and authentication policies\n- Remote access / VPN policies\n- Data retention policies\n- Data encryption standards\n\nPlease specify so I can search for the right document."},
    {"query": "Show me the current headcount for the sales department.", "response": "I don't have specific data about the sales department headcount in our current document corpus. The available HR data covers the engineering division. Could you clarify:\n- Are you looking for engineering headcount instead?\n- Do you have a specific report or time period in mind?"},
    {"query": "Show me the latest incident reports for the server farm.", "response": "I need more specifics:\n- Which server farm or data center? (e.g., US-East)\n- Are you looking for a specific time period?\n- Is this about the core API incident post-mortem, or a different system?"},
    {"query": "What are the new policies regarding data retention?", "response": "I can search for data retention policies, but could you clarify:\n- Are you asking about security audit log retention?\n- Or data retention for a specific department?\n- Are you interested in regulatory compliance requirements (GDPR DSAR timelines)?"},
    {"query": "Give me the financial report for last quarter.", "response": "I'd be happy to pull financial data, but I need to know which quarter:\n- Q1 & Q2 data is in the H1 Financial Report\n- Q3 & Q4 data is in the H2 Financial Report\n\nCould you specify the exact quarter?"},
    {"query": "What is the performance of the new server model?", "response": "I need clarification:\n- Are you asking about server infrastructure performance (uptime, power)?\n- Or application/API performance metrics (latency, throughput)?\n- Is there a specific server model number you're interested in?"},
]

# ---------------------------------------------------------------------------
# NEW: Error-recovery templates (bad search -> realize -> retry)
# ---------------------------------------------------------------------------

ERROR_RECOVERY_TEMPLATES = [
    {
        "query": "How much money does the company make?",
        "bad_search": "company money income",
        "good_search": "total revenue financial report",
        "doc_hint": "doc_fin",
    },
    {
        "query": "Tell me about the computers in the AI lab.",
        "bad_search": "computers AI lab hardware",
        "good_search": "AI cluster GPU specifications training",
        "doc_hint": "doc_ai",
    },
    {
        "query": "What's the internet speed for the GPU machines?",
        "bad_search": "internet speed GPU machines",
        "good_search": "InfiniBand speed AI cluster network",
        "doc_hint": "doc_ai",
    },
    {
        "query": "How many people work in tech?",
        "bad_search": "people work tech department",
        "good_search": "engineering headcount total employees",
        "doc_hint": "doc_hr",
    },
    {
        "query": "What went wrong with the website?",
        "bad_search": "website problem error",
        "good_search": "API incident downtime root cause",
        "doc_hint": "doc_incident",
    },
]

# ---------------------------------------------------------------------------
# NEW: Adversarial / out-of-corpus templates
# ---------------------------------------------------------------------------

ADVERSARIAL_TEMPLATES = [
    {"query": "What is the marketing budget for Q3?",
     "response": "I searched the available documents but could not find specific marketing budget data. The internal corpus contains IT budget allocations (totaling $8.4M across infrastructure, engineering, cloud, and security), but marketing budget details are not available in the current document set. You may need to check with the marketing department directly."},
    {"query": "How many customers does the company have?",
     "response": "I was unable to find customer count data in the available internal documents. The corpus contains financial reports, infrastructure specs, HR data, and operational metrics, but does not include customer-specific statistics. Please check with the sales or customer success team for this information."},
    {"query": "What is the CEO's compensation?",
     "response": "Executive compensation details are not available in the current internal document corpus. The available HR data covers engineering headcount and attrition rates. For executive compensation information, please contact the HR or legal department."},
]

# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------

def build_single_step(query: str, search_query: str, doc_hint: str) -> Dict:
    results = bm25_search(search_query, top_k=2)
    if not results:
        return None
    doc = results[0]
    thought1 = _thought(_get_reasoning(doc_hint, query))
    tool_call = _tool_call(search_query, 2)
    observation = json.dumps({"results": results}, indent=2)
    thought2 = _thought(_get_synthesis_reasoning(doc['title'], query))
    answer = f"Based on the retrieved document **{doc['title']}**:\n\n{doc['text']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"{thought1}\n{tool_call}"},
        {"role": "tool", "content": observation},
        {"role": "assistant", "content": f"{thought2}\n{answer}"},
    ]
    return {"messages": messages}


def build_multi_step(template: Dict) -> Dict:
    query = template["query"]
    s1_query, _ = template["searches"][0]
    s2_query, _ = template["searches"][1]
    results1 = bm25_search(s1_query, top_k=2)
    if not results1:
        return None
    thought1 = _thought(f"This question requires comparing information from multiple sources. I'll start by searching for the first data point.")
    tool_call1 = _tool_call(s1_query, 2)
    obs1 = json.dumps({"results": results1}, indent=2)
    results2 = bm25_search(s2_query, top_k=2)
    if not results2:
        return None
    thought2 = _thought(f"Found data from '{results1[0]['title']}'. Now I need the second data point for comparison.")
    tool_call2 = _tool_call(s2_query, 2)
    obs2 = json.dumps({"results": results2}, indent=2)
    thought3 = _thought(f"I now have data from both '{results1[0]['title']}' and '{results2[0]['title']}'. I can synthesize a comparison.")
    answer = (
        f"Based on the retrieved documents:\n\n"
        f"**From {results1[0]['title']}:** {results1[0]['text']}\n\n"
        f"**From {results2[0]['title']}:** {results2[0]['text']}\n\n"
        f"This comparison shows the relationship between these two data points."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"{thought1}\n{tool_call1}"},
        {"role": "tool", "content": obs1},
        {"role": "assistant", "content": f"{thought2}\n{tool_call2}"},
        {"role": "tool", "content": obs2},
        {"role": "assistant", "content": f"{thought3}\n{answer}"},
    ]
    return {"messages": messages}


def build_no_tool(template: Dict) -> Dict:
    thought = _thought("This is a general knowledge question. I can answer directly without searching internal documents.")
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template["query"]},
        {"role": "assistant", "content": f"{thought}\n{template['answer']}"},
    ]}


def build_missing_info(template: Dict) -> Dict:
    thought = _thought("This query is too vague to search effectively. I need to ask for specific details before retrieving documents.")
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template["query"]},
        {"role": "assistant", "content": f"{thought}\n{template['response']}"},
    ]}


def build_error_recovery(template: Dict) -> Dict:
    """Build a trajectory where the model makes a poor search, then retries."""
    query = template["query"]
    bad_results = bm25_search(template["bad_search"], top_k=2)
    good_results = bm25_search(template["good_search"], top_k=2)
    if not good_results:
        return None

    thought1 = _thought(f"The user is asking about '{query}'. Let me search for relevant information.")
    tool_call1 = _tool_call(template["bad_search"], 2)
    obs1 = json.dumps({"results": bad_results}, indent=2)

    thought2 = _thought(
        "The search results don't seem very relevant to what the user is asking. "
        "Let me refine my search query with more specific technical keywords."
    )
    tool_call2 = _tool_call(template["good_search"], 2)
    obs2 = json.dumps({"results": good_results}, indent=2)

    doc = good_results[0]
    thought3 = _thought(f"This is much better. '{doc['title']}' contains the information I need.")
    answer = f"Based on **{doc['title']}**:\n\n{doc['text']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"{thought1}\n{tool_call1}"},
        {"role": "tool", "content": obs1},
        {"role": "assistant", "content": f"{thought2}\n{tool_call2}"},
        {"role": "tool", "content": obs2},
        {"role": "assistant", "content": f"{thought3}\n{answer}"},
    ]
    return {"messages": messages}


def build_adversarial(template: Dict) -> Dict:
    """Build a trajectory where the answer is NOT in the corpus."""
    thought1 = _thought("Let me search the internal documents for this information.")
    search_q = " ".join(template["query"].lower().split()[:5])
    tool_call = _tool_call(search_q, 2)
    results = bm25_search(search_q, top_k=2)
    obs = json.dumps({"results": results}, indent=2)
    thought2 = _thought("The retrieved documents don't contain the specific information the user is asking about. I should inform them honestly.")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template["query"]},
        {"role": "assistant", "content": f"{thought1}\n{tool_call}"},
        {"role": "tool", "content": obs},
        {"role": "assistant", "content": f"{thought2}\n{template['response']}"},
    ]
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Query variation helpers (MORE aggressive paraphrasing)
# ---------------------------------------------------------------------------

PERIOD_VARS = ["Q1", "Q2", "Q3", "Q4", "H1", "Q1 and Q2"]
DC_VARS = ["US-East", "primary", "main"]
PROD_VARS = ["Alpha", "Beta", "Gamma"]
WH_VARS = ["A", "B"]

QUERY_PREFIXES = [
    "", "Can you tell me ", "I need to know ", "Please find ",
    "What is ", "Look up ", "Search for ", "Help me find ",
    "I'd like to know ", "Could you check ", "Find out ",
    "Tell me ", "Show me ", "I'm looking for ", "Do you know ",
]

QUERY_SUFFIXES = [
    "", " from our records", " according to our documents",
    " based on the latest data", " please", " if available",
]


def generate_single_step_variations() -> List[Dict]:
    trajectories = []
    for tmpl_query, tmpl_search, doc_hint in SINGLE_STEP_TEMPLATES:
        for prefix in QUERY_PREFIXES:
            for suffix in random.sample(QUERY_SUFFIXES, min(3, len(QUERY_SUFFIXES))):
                q = tmpl_query
                s = tmpl_search
                if "{period}" in q:
                    period = random.choice(PERIOD_VARS)
                    q = q.replace("{period}", period)
                    s = s.replace("{period}", period)
                if "{dc}" in q:
                    dc = random.choice(DC_VARS)
                    q = q.replace("{dc}", dc)
                    s = s.replace("{dc}", dc)
                if "{prod}" in q:
                    prod = random.choice(PROD_VARS)
                    q = q.replace("{prod}", prod)
                    s = s.replace("{prod}", prod)
                if "{wh}" in q:
                    wh = random.choice(WH_VARS)
                    q = q.replace("{wh}", wh)
                    s = s.replace("{wh}", wh)
                if prefix and not q.startswith("What"):
                    q = prefix + q[0].lower() + q[1:]
                q = q + suffix
                traj = build_single_step(q, s, doc_hint)
                if traj:
                    trajectories.append(traj)
    return trajectories


def generate_multi_step_all() -> List[Dict]:
    trajectories = []
    prefixes = ["", "Can you ", "Please ", "I'd like you to ", "Could you ", "Help me "]
    for tmpl in MULTI_STEP_TEMPLATES:
        for prefix in prefixes:
            varied = dict(tmpl)
            if prefix:
                varied["query"] = prefix + tmpl["query"][0].lower() + tmpl["query"][1:]
            t = build_multi_step(varied)
            if t:
                trajectories.append(t)
    return trajectories


def generate_error_recovery_all() -> List[Dict]:
    trajectories = []
    for tmpl in ERROR_RECOVERY_TEMPLATES:
        traj = build_error_recovery(tmpl)
        if traj:
            trajectories.append(traj)
            # Variation
            varied = dict(tmpl)
            varied["query"] = "Can you tell me " + tmpl["query"][0].lower() + tmpl["query"][1:]
            t = build_error_recovery(varied)
            if t:
                trajectories.append(t)
    return trajectories


def generate_adversarial_all() -> List[Dict]:
    return [build_adversarial(t) for t in ADVERSARIAL_TEMPLATES]


def generate_no_tool_all() -> List[Dict]:
    return [build_no_tool(t) for t in NO_TOOL_TEMPLATES]


def generate_missing_info_all() -> List[Dict]:
    return [build_missing_info(t) for t in MISSING_INFO_TEMPLATES]


# ---------------------------------------------------------------------------
# Deduplication & main
# ---------------------------------------------------------------------------

def deduplicate(trajectories: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for t in trajectories:
        msgs = t.get("messages", [])
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        key = hashlib.md5("||".join(user_msgs).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def main():
    print("=" * 70)
    print("V1 — Preparing enhanced training data for Qwen 3B BM25 Agent")
    print("=" * 70)

    print("\n[1/6] Generating single-step trajectories...")
    single = generate_single_step_variations()
    print(f"  -> Generated {len(single)} single-step trajectories")

    print("\n[2/6] Generating multi-step trajectories...")
    multi = generate_multi_step_all()
    print(f"  -> Generated {len(multi)} multi-step trajectories")

    print("\n[3/6] Generating error-recovery trajectories...")
    recovery = generate_error_recovery_all()
    print(f"  -> Generated {len(recovery)} error-recovery trajectories")

    print("\n[4/6] Generating adversarial trajectories...")
    adversarial = generate_adversarial_all()
    print(f"  -> Generated {len(adversarial)} adversarial trajectories")

    print("\n[5/6] Generating no-tool and missing-info trajectories...")
    no_tool = generate_no_tool_all()
    missing = generate_missing_info_all()
    print(f"  -> Generated {len(no_tool)} no-tool + {len(missing)} missing-info")

    print("\n[6/6] Merging and deduplicating...")
    all_data = single + multi + recovery + adversarial + no_tool + missing
    all_data = deduplicate(all_data)
    random.shuffle(all_data)

    # Tag each trajectory with its type for curriculum learning
    # (type is inferred from message count and content)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_data)} trajectories saved to {OUTPUT_FILE}")
    print(f"  - Single-step: ~{len(single)}")
    print(f"  - Multi-step:  ~{len(multi)}")
    print(f"  - Error-recovery: ~{len(recovery)}")
    print(f"  - Adversarial: ~{len(adversarial)}")
    print(f"  - No-tool: ~{len(no_tool)}")
    print(f"  - Missing-info: ~{len(missing)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
