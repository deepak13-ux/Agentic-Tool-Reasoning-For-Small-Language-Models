"""
Prepare high-quality SFT training data for Qwen 3B BM25 tool-use agent.

This script:
1. Cleans existing train_data_clean.jsonl (fixes tool-call format issues)
2. Generates ~500 new synthetic trajectories programmatically (no API needed)
3. Outputs a merged, deduplicated dataset to train_data_3b.jsonl
"""

import json
import random
import hashlib
from typing import Any, Dict, List
from corpus import CORPUS, SYSTEM_PROMPT, bm25_search

random.seed(42)
OUTPUT_FILE = "train_data_3b.jsonl"

# ---------------------------------------------------------------------------
# 1. SINGLE-STEP trajectory templates
# ---------------------------------------------------------------------------

SINGLE_STEP_TEMPLATES = [
    # (query_template, search_query, doc_id_hint)
    ("What is the total revenue in {period} from the financial report?", "total revenue {period} financial report", "doc_fin"),
    ("How many server nodes are in the {dc} data center?", "server nodes {dc} data center", "doc_infra_02"),
    ("What is the maximum payload capacity of the logistics fleet trucks?", "logistics fleet truck payload capacity", "doc_logistics_03"),
    ("How many units of product {prod} are in Warehouse {wh}?", "warehouse {wh} product {prod} inventory units", "doc_inventory_04"),
    ("What is the total engineering headcount?", "engineering headcount total employees", "doc_hr_05"),
    ("What is the password rotation policy?", "password rotation policy security", "doc_sec_06"),
    ("What caused the Q3 API incident?", "Q3 API incident downtime root cause", "doc_incident_07"),
    ("How many GPUs does the AI training cluster have?", "AI cluster GPU count specifications", "doc_ai_08"),
    ("What was Q4 revenue?", "Q4 revenue financial report", "doc_fin_09"),
    ("What were the SOC 2 audit results?", "SOC 2 audit compliance results", "doc_compliance_10"),
    ("What is the total IT budget for the fiscal year?", "IT budget total fiscal year allocation", "doc_budget_11"),
    ("What is the API p99 latency?", "API p99 latency performance metrics", "doc_perf_12"),
    ("How many regional offices does engineering have?", "engineering regional offices locations", "doc_hr_05"),
    ("What encryption standard is used for data at rest?", "encryption data at rest standard security", "doc_sec_06"),
    ("What is the CDN cache hit ratio?", "CDN cache hit ratio performance", "doc_perf_12"),
    ("What is the InfiniBand speed in the AI cluster?", "InfiniBand speed AI cluster network", "doc_ai_08"),
    ("How many delivery vans operate at Hub Beta?", "delivery vans Hub Beta logistics fleet", "doc_logistics_03"),
    ("What is the PUE ratio of the data center?", "PUE power usage effectiveness data center", "doc_infra_02"),
    ("What is the standby generator capacity?", "standby generator capacity data center", "doc_infra_02"),
    ("What is the monthly fleet maintenance cost?", "fleet maintenance cost monthly logistics", "doc_logistics_03"),
    ("What is the total inventory value across warehouses?", "total inventory value warehouses", "doc_inventory_04"),
    ("How many users were affected by the Q3 incident?", "Q3 incident affected users API", "doc_incident_07"),
    ("What is the annual attrition rate for engineering?", "attrition rate engineering annual", "doc_hr_05"),
    ("What is the GDPR compliance status?", "GDPR compliance review findings", "doc_compliance_10"),
    ("What percentage of the IT budget goes to cloud services?", "IT budget cloud services percentage", "doc_budget_11"),
    ("What is the average GPU utilization in the AI cluster?", "GPU utilization average AI cluster", "doc_ai_08"),
    ("What is the full-year revenue figure?", "full year revenue total annual", "doc_fin_09"),
    ("What is the DSAR average response time?", "DSAR response time average days", "doc_compliance_10"),
    ("What is the database query p95 latency?", "database query p95 latency", "doc_perf_12"),
    ("What is the peak traffic in requests per second?", "peak traffic requests per second", "doc_perf_12"),
    ("What is the capital expenditure total?", "capital expenditure total capex", "doc_fin_09"),
    ("How many concurrent training jobs does the AI cluster support?", "concurrent training jobs AI cluster", "doc_ai_08"),
    ("What is the error rate across all API endpoints?", "error rate endpoints API performance", "doc_perf_12"),
    ("What is the VPN authentication requirement?", "VPN authentication remote access requirement", "doc_sec_06"),
    ("How long must security audit logs be retained?", "security audit logs retention period", "doc_sec_06"),
    ("What release fixed the Q3 API incident?", "release fix Q3 API incident version", "doc_incident_07"),
    ("What is the operating margin?", "operating margin percentage financial", "doc_fin_01"),
    ("How much does R&D spending account for?", "R&D spending percentage revenue", "doc_fin_01"),
    ("What is the net profit for H1?", "net profit H1 first half", "doc_fin_01"),
    ("What is the fuel consumption of the logistics fleet?", "fuel consumption liters logistics fleet", "doc_logistics_03"),
]

MULTI_STEP_TEMPLATES = [
    {
        "query": "Compare the revenue in Q1 versus Q4.",
        "searches": [
            ("Q1 revenue financial report", "doc_fin_01"),
            ("Q4 revenue financial report", "doc_fin_09"),
        ],
    },
    {
        "query": "Compare the number of server nodes to the number of GPUs in the AI cluster.",
        "searches": [
            ("server nodes data center total", "doc_infra_02"),
            ("AI cluster GPU count", "doc_ai_08"),
        ],
    },
    {
        "query": "What is the engineering headcount relative to the IT budget for salaries?",
        "searches": [
            ("engineering headcount total", "doc_hr_05"),
            ("IT budget engineering salaries allocation", "doc_budget_11"),
        ],
    },
    {
        "query": "Compare H1 net profit with the full-year net profit margin.",
        "searches": [
            ("H1 net profit revenue", "doc_fin_01"),
            ("full year net profit margin", "doc_fin_09"),
        ],
    },
    {
        "query": "How does the API latency compare to the database query latency?",
        "searches": [
            ("API p50 p99 latency performance", "doc_perf_12"),
            ("database query p95 latency", "doc_perf_12"),
        ],
    },
    {
        "query": "Compare the security audit log retention period with the GDPR DSAR response requirement.",
        "searches": [
            ("security audit logs retention days", "doc_sec_06"),
            ("GDPR DSAR response time days", "doc_compliance_10"),
        ],
    },
    {
        "query": "What is the relationship between fleet size and warehouse inventory value?",
        "searches": [
            ("logistics fleet trucks vans total", "doc_logistics_03"),
            ("warehouse inventory total value", "doc_inventory_04"),
        ],
    },
    {
        "query": "How does the IT infrastructure budget allocation compare with actual server infrastructure costs?",
        "searches": [
            ("IT budget infrastructure allocation percentage", "doc_budget_11"),
            ("server data center power watts", "doc_infra_02"),
        ],
    },
    {
        "query": "Compare the Q3 incident impact with the overall API performance metrics.",
        "searches": [
            ("Q3 incident downtime affected users", "doc_incident_07"),
            ("API error rate latency performance", "doc_perf_12"),
        ],
    },
    {
        "query": "What fraction of the total budget is spent on the AI cluster versus cloud services?",
        "searches": [
            ("AI cluster specifications GPU", "doc_ai_08"),
            ("IT budget cloud services percentage allocation", "doc_budget_11"),
        ],
    },
]

NO_TOOL_TEMPLATES = [
    {"query": "Explain the difference between a hash table and a binary search tree.", "answer": "A hash table stores key-value pairs using a hash function for O(1) average lookup, insert, and delete. A binary search tree (BST) organizes data hierarchically: each node's left children are smaller and right children are larger, giving O(log n) average operations. Hash tables are faster for lookups but don't maintain order, while BSTs support ordered traversal and range queries efficiently."},
    {"query": "What are the time complexities of quicksort and mergesort?", "answer": "Quicksort has O(n log n) average time complexity but O(n²) worst case (when the pivot is always the smallest or largest element). Mergesort guarantees O(n log n) in all cases but requires O(n) extra space. Quicksort is typically faster in practice due to better cache locality and lower constant factors."},
    {"query": "Describe how a blockchain achieves consensus.", "answer": "Blockchains use consensus mechanisms like Proof of Work (PoW) or Proof of Stake (PoS). In PoW, miners compete to solve cryptographic puzzles; the first to solve adds a block and is rewarded. In PoS, validators stake cryptocurrency as collateral and are selected to propose blocks based on stake size. Both ensure all nodes agree on the transaction history without a central authority."},
    {"query": "Explain the concept of memoization in dynamic programming.", "answer": "Memoization is a top-down optimization technique where you cache the results of expensive function calls and return the cached result when the same inputs occur again. It converts exponential recursive solutions into polynomial time by avoiding redundant computation. For example, computing Fibonacci(n) recursively is O(2^n), but with memoization it becomes O(n) by storing each Fibonacci(k) once."},
    {"query": "What is the principle behind the P versus NP problem?", "answer": "P is the class of problems solvable in polynomial time. NP is the class where solutions can be verified in polynomial time. The P vs NP question asks whether every problem whose solution can be quickly verified can also be quickly solved. It remains one of the most important unsolved problems in computer science. If P=NP, many currently intractable problems (like SAT, TSP) would have efficient solutions."},
    {"query": "How does backpropagation work in neural networks?", "answer": "Backpropagation computes the gradient of the loss function with respect to each weight by applying the chain rule layer by layer, from the output back to the input. First, a forward pass computes predictions. Then the loss is calculated, and the backward pass propagates error gradients through each layer. These gradients are used by an optimizer (like SGD or Adam) to update weights and minimize the loss."},
    {"query": "What is the difference between TCP and UDP?", "answer": "TCP (Transmission Control Protocol) is connection-oriented, providing reliable, ordered delivery with error checking and retransmission. UDP (User Datagram Protocol) is connectionless, offering faster but unreliable delivery without guarantees of order or completeness. TCP is used for web browsing, email, and file transfers. UDP is preferred for real-time applications like video streaming, gaming, and DNS lookups."},
    {"query": "Explain how a Bloom filter works.", "answer": "A Bloom filter is a space-efficient probabilistic data structure for set membership testing. It uses a bit array and multiple hash functions. To add an element, each hash function maps it to a bit position which is set to 1. To query, check all hash positions — if any is 0, the element is definitely not in the set. If all are 1, it probably is (false positives are possible, but false negatives are not)."},
    {"query": "What is the CAP theorem?", "answer": "The CAP theorem states that a distributed system can simultaneously guarantee at most two of three properties: Consistency (all nodes see the same data), Availability (every request gets a response), and Partition tolerance (the system works despite network failures). Since network partitions are inevitable, real systems must choose between consistency (CP systems like HBase) and availability (AP systems like Cassandra)."},
    {"query": "Describe how MapReduce works.", "answer": "MapReduce is a programming model for processing large datasets in parallel. The Map phase applies a function to each input record, producing key-value pairs. The Shuffle phase groups all values by key. The Reduce phase aggregates values for each key into a final result. This model allows distributed computation across thousands of machines with automatic fault tolerance, popularized by Google and implemented in Hadoop."},
    {"query": "What is the difference between a process and a thread?", "answer": "A process is an independent execution unit with its own memory space, file descriptors, and resources. A thread is a lightweight execution unit within a process that shares memory and resources with other threads in the same process. Context switching between threads is faster than between processes. However, shared memory makes threads prone to race conditions, requiring synchronization mechanisms like mutexes."},
    {"query": "Explain the concept of a semaphore in concurrent programming.", "answer": "A semaphore is a synchronization primitive that controls access to shared resources. It maintains a counter: acquire (wait/P) decrements it, and release (signal/V) increments it. A binary semaphore (values 0 or 1) acts like a mutex. A counting semaphore allows up to N concurrent accesses, useful for limiting connection pools or thread pools. Semaphores prevent race conditions in multi-threaded programs."},
]

MISSING_INFO_TEMPLATES = [
    {"query": "What is the latest security policy?", "response": "I'd be happy to help, but could you clarify which specific security policy you're asking about? For example, are you interested in:\n- Password rotation and authentication policies\n- Remote access / VPN policies\n- Data retention policies\n- Data encryption standards\n\nPlease specify so I can search for the right document."},
    {"query": "Show me the current headcount for the sales department.", "response": "I don't have specific data about the sales department headcount in our current document corpus. The available HR data covers the engineering division. Could you clarify:\n- Are you looking for engineering headcount instead?\n- Do you have a specific report or time period in mind?\n\nThis will help me search for the most relevant information."},
    {"query": "Show me the latest incident reports for the server farm.", "response": "I'd like to help, but I need more specifics:\n- Which server farm or data center are you referring to? (e.g., US-East)\n- Are you looking for a specific time period?\n- Is this about the core API incident post-mortem, or a different system?\n\nPlease provide additional details so I can retrieve the right report."},
    {"query": "What are the new policies regarding data retention?", "response": "I can search for data retention policies, but could you clarify:\n- Are you asking about security audit log retention?\n- Or data retention for a specific department (e.g., customer records, HR data)?\n- Are you interested in regulatory compliance requirements (GDPR DSAR timelines)?\n\nPlease specify the area so I can find the most relevant policy."},
    {"query": "Give me the financial report for last quarter.", "response": "I'd be happy to pull financial data, but I need to know which quarter you're referring to:\n- Q1 & Q2 data is available in the H1 Financial Report\n- Q3 & Q4 data is available in the H2 Financial Report\n\nCould you specify the exact quarter (Q1, Q2, Q3, or Q4) so I can retrieve the correct figures?"},
    {"query": "What is the performance of the new server model?", "response": "I need some clarification to search for the right information:\n- Are you asking about the server infrastructure performance (uptime, power)?\n- Or the application/API performance metrics (latency, throughput)?\n- Is there a specific server model number you're interested in?\n\nPlease provide more details so I can search the correct reports."},
]


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------

def _build_thought(reasoning: str) -> str:
    return f"<thought>{reasoning}</thought>"


def _build_tool_call(query: str, top_k: int = 2) -> str:
    call = {"name": "bm25_search", "arguments": {"query": query, "top_k": top_k}}
    return '```json\n' + json.dumps(call) + '\n```'


def build_single_step(query: str, search_query: str) -> Dict[str, Any]:
    """Build a single-step tool-use trajectory."""
    results = bm25_search(search_query, top_k=2)
    if not results:
        return None

    doc = results[0]
    thought1 = _build_thought(
        f"The user is asking about information that should be in our internal documents. "
        f"I need to search for relevant data using the BM25 search tool."
    )
    tool_call = _build_tool_call(search_query, 2)

    observation = json.dumps({"results": results}, indent=2)

    thought2 = _build_thought(
        f"I found relevant information in the document '{doc['title']}'. "
        f"Let me synthesize a clear answer from the retrieved data."
    )
    # Build answer from the document text
    answer = f"Based on the retrieved document **{doc['title']}**:\n\n{doc['text']}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"{thought1}\n{tool_call}"},
        {"role": "tool", "content": observation},
        {"role": "assistant", "content": f"{thought2}\n{answer}"},
    ]
    return {"messages": messages}


def build_multi_step(template: Dict) -> Dict[str, Any]:
    """Build a multi-step tool-use trajectory (2 searches)."""
    query = template["query"]
    s1_query, s1_doc = template["searches"][0]
    s2_query, s2_doc = template["searches"][1]

    # First search
    results1 = bm25_search(s1_query, top_k=2)
    if not results1:
        return None

    thought1 = _build_thought(
        f"This question requires information from multiple sources. "
        f"I'll start by searching for the first piece of information."
    )
    tool_call1 = _build_tool_call(s1_query, 2)
    obs1 = json.dumps({"results": results1}, indent=2)

    # Second search
    results2 = bm25_search(s2_query, top_k=2)
    if not results2:
        return None

    thought2 = _build_thought(
        f"I found the first piece of data from '{results1[0]['title']}'. "
        f"Now I need to search for additional information to complete the comparison."
    )
    tool_call2 = _build_tool_call(s2_query, 2)
    obs2 = json.dumps({"results": results2}, indent=2)

    thought3 = _build_thought(
        f"Now I have data from both '{results1[0]['title']}' and '{results2[0]['title']}'. "
        f"Let me synthesize a comprehensive comparison."
    )
    answer = (
        f"Based on the retrieved documents:\n\n"
        f"**From {results1[0]['title']}:** {results1[0]['text']}\n\n"
        f"**From {results2[0]['title']}:** {results2[0]['text']}\n\n"
        f"This comparison shows the relationship between these two data points from our internal records."
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


def build_no_tool(template: Dict) -> Dict[str, Any]:
    """Build a no-tool trajectory (direct knowledge answer)."""
    thought = _build_thought(
        "This is a general knowledge question about computer science concepts. "
        "I can answer this directly without searching any internal documents."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template["query"]},
        {"role": "assistant", "content": f"{thought}\n{template['answer']}"},
    ]
    return {"messages": messages}


def build_missing_info(template: Dict) -> Dict[str, Any]:
    """Build a missing-info trajectory (ask for clarification)."""
    thought = _build_thought(
        "This query is too vague to search effectively. I need to ask the user "
        "for more specific details before I can retrieve the right documents."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": template["query"]},
        {"role": "assistant", "content": f"{thought}\n{template['response']}"},
    ]
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Query variation helpers
# ---------------------------------------------------------------------------

PERIOD_VARS = ["Q1", "Q2", "Q3", "Q4", "H1", "Q1 and Q2"]
DC_VARS = ["US-East", "primary", "main"]
PROD_VARS = ["Alpha", "Beta", "Gamma"]
WH_VARS = ["A", "B"]

QUERY_PREFIXES = [
    "", "Can you tell me ", "I need to know ", "Please find ",
    "What is ", "Look up ", "Search for ", "Help me find ",
]


def generate_single_step_variations() -> List[Dict]:
    """Generate variations of single-step templates."""
    trajectories = []
    for tmpl_query, tmpl_search, _ in SINGLE_STEP_TEMPLATES:
        # Generate a few variations per template
        for prefix in QUERY_PREFIXES:
            q = tmpl_query
            s = tmpl_search
            # Fill in template variables if any
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

            traj = build_single_step(q, s)
            if traj:
                trajectories.append(traj)
    return trajectories


def generate_multi_step_all() -> List[Dict]:
    """Generate all multi-step trajectories."""
    trajectories = []
    for tmpl in MULTI_STEP_TEMPLATES:
        traj = build_multi_step(tmpl)
        if traj:
            # Generate a few with slightly different wordings
            trajectories.append(traj)
            # Variation with prefix
            for prefix in ["Can you ", "Please ", "I'd like you to "]:
                varied = dict(tmpl)
                varied["query"] = prefix + tmpl["query"][0].lower() + tmpl["query"][1:]
                t = build_multi_step(varied)
                if t:
                    trajectories.append(t)
    return trajectories


def generate_no_tool_all() -> List[Dict]:
    return [build_no_tool(t) for t in NO_TOOL_TEMPLATES]


def generate_missing_info_all() -> List[Dict]:
    return [build_missing_info(t) for t in MISSING_INFO_TEMPLATES]


# ---------------------------------------------------------------------------
# Clean existing data
# ---------------------------------------------------------------------------

def clean_existing_data(path: str = "train_data_clean.jsonl") -> List[Dict]:
    """Load existing data and fix tool call format issues."""
    cleaned = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    msgs = record.get("messages", [])
                    if len(msgs) < 3:
                        continue
                    # Ensure system prompt is correct
                    msgs[0] = {"role": "system", "content": SYSTEM_PROMPT}
                    # Check for proper tool call format in assistant messages
                    valid = True
                    for msg in msgs:
                        if msg["role"] == "assistant":
                            content = msg.get("content", "")
                            # Check if it has a tool call but in wrong format
                            if '"action"' in content and '"parameters"' in content:
                                valid = False  # Bad format, skip
                                break
                    if valid:
                        cleaned.append(record)
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping existing data.")
    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def deduplicate(trajectories: List[Dict]) -> List[Dict]:
    """Deduplicate based on user query content."""
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
    print("Preparing training data for Qwen 3B BM25 Agent")
    print("=" * 70)

    # Step 1: Clean existing data
    print("\n[1/5] Cleaning existing training data...")
    existing = clean_existing_data()
    print(f"  -> Kept {len(existing)} valid trajectories from existing data")

    # Step 2: Generate single-step
    print("\n[2/5] Generating single-step trajectories...")
    single = generate_single_step_variations()
    print(f"  -> Generated {len(single)} single-step trajectories")

    # Step 3: Generate multi-step
    print("\n[3/5] Generating multi-step trajectories...")
    multi = generate_multi_step_all()
    print(f"  -> Generated {len(multi)} multi-step trajectories")

    # Step 4: Generate no-tool + missing-info
    print("\n[4/5] Generating no-tool and missing-info trajectories...")
    no_tool = generate_no_tool_all()
    missing = generate_missing_info_all()
    print(f"  -> Generated {len(no_tool)} no-tool + {len(missing)} missing-info trajectories")

    # Step 5: Merge, deduplicate, shuffle
    print("\n[5/5] Merging and deduplicating...")
    all_data = existing + single + multi + no_tool + missing
    all_data = deduplicate(all_data)
    random.shuffle(all_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"Total: {len(all_data)} trajectories saved to {OUTPUT_FILE}")
    print(f"  - From existing data: ~{len(existing)}")
    print(f"  - New synthetic: ~{len(single) + len(multi) + len(no_tool) + len(missing)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
