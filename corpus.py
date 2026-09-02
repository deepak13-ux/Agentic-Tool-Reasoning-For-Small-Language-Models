"""
Shared corpus & BM25 search engine.
Import this module from any script that needs document retrieval.
"""

import re
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Corporate Document Corpus  (12 documents — expanded for richer training)
# ---------------------------------------------------------------------------

CORPUS: List[Dict[str, Any]] = [
    {
        "id": "doc_fin_01",
        "title": "Q1 & Q2 Financial Report",
        "text": (
            "The company recorded total revenue of $450,000 in Q1 and $620,000 in Q2. "
            "Total operating expenses were $180,000 in Q1 and $210,000 in Q2, yielding "
            "an operating margin of 28%. Net profit for H1 was $680,000. R&D spending "
            "accounted for 18% of total revenue."
        ),
        "keywords": ["$450,000", "$620,000", "revenue", "q1", "q2", "operating margin", "28%", "net profit", "$680,000"],
    },
    {
        "id": "doc_infra_02",
        "title": "US-East Server Infrastructure Report",
        "text": (
            "The US-East data center operates 128 total server nodes across 8 data racks "
            "(16 nodes per rack). Each rack draws 4,200 watts. Standby generator capacity "
            "is 50 kW. Average server uptime is 99.97% over the last 12 months. The PUE "
            "(Power Usage Effectiveness) ratio is 1.15."
        ),
        "keywords": ["128", "nodes", "racks", "4,200", "watts", "99.97%", "uptime", "PUE", "1.15"],
    },
    {
        "id": "doc_logistics_03",
        "title": "Fleet Capacity & Logistics Plan",
        "text": (
            "The logistics fleet operates 14 transport trucks at Hub Alpha and 8 delivery "
            "vans at Hub Beta. Each truck has a maximum payload capacity of 18,000 kilograms. "
            "Average fuel consumption is 32 liters per 100 km. Monthly fleet maintenance "
            "cost averages $12,500."
        ),
        "keywords": ["14", "trucks", "18,000", "payload", "hub alpha", "hub beta", "8 delivery vans", "32 liters"],
    },
    {
        "id": "doc_inventory_04",
        "title": "Warehouse Inventory Status",
        "text": (
            "Warehouse A contains 850 units of product Alpha and 1,250 units of product Beta. "
            "Warehouse B holds 400 units of product Alpha and 600 units of product Gamma. "
            "Total inventory value across both warehouses is $2.3 million. Monthly turnover "
            "rate is 4.2 cycles."
        ),
        "keywords": ["850", "1,250", "warehouse a", "product alpha", "product beta", "$2.3 million", "4.2 cycles"],
    },
    {
        "id": "doc_hr_05",
        "title": "Engineering Headcount & Regional Offices",
        "text": (
            "The engineering division employs 96 backend engineers, 48 frontend engineers, "
            "and 24 DevOps specialists across 6 regional offices: San Francisco, Austin, "
            "London, Berlin, Tokyo, and Bengaluru. Total engineering headcount is 168. "
            "The attrition rate is 8.5% annually."
        ),
        "keywords": ["96", "48", "24", "168", "backend", "frontend", "devops", "6 regional offices", "8.5%"],
    },
    {
        "id": "doc_sec_06",
        "title": "Security & Data Retention Policy",
        "text": (
            "Corporate policy mandates 90-day password rotation and multi-factor authentication "
            "(MFA). Security audit logs must be retained in cold storage for 365 days. All "
            "remote access requires VPN with certificate-based authentication. Data encryption "
            "at rest uses AES-256."
        ),
        "keywords": ["90-day", "password", "MFA", "365 days", "VPN", "AES-256", "remote access"],
    },
    {
        "id": "doc_incident_07",
        "title": "Q3 Core API Incident Post-Mortem",
        "text": (
            "On August 14, the core API service experienced 42 minutes of downtime due to "
            "database connection pool exhaustion caused by unindexed queries. The incident "
            "affected 12,000 users. Root cause was traced to a missing index on the orders "
            "table. Resolved in release v2.4.1."
        ),
        "keywords": ["42 minutes", "downtime", "connection pool", "unindexed", "12,000 users", "v2.4.1"],
    },
    {
        "id": "doc_ai_08",
        "title": "AI Cluster Specifications",
        "text": (
            "The AI training cluster houses 32 NVIDIA H100 GPUs interconnected via 800 Gbps "
            "InfiniBand. Model checkpoints are written to distributed NVMe storage with 1.2 PB "
            "capacity. Average GPU utilization is 78%. The cluster supports up to 64 concurrent "
            "training jobs."
        ),
        "keywords": ["32", "H100", "800 Gbps", "InfiniBand", "1.2 PB", "78%", "64 concurrent"],
    },
    {
        "id": "doc_fin_09",
        "title": "Q3 & Q4 Financial Report",
        "text": (
            "Q3 revenue was $710,000 and Q4 revenue reached $820,000, bringing full-year "
            "revenue to $2.6 million. Operating expenses in Q3 were $240,000 and $275,000 in Q4. "
            "The company achieved a full-year net profit margin of 22%. Capital expenditure "
            "totaled $450,000."
        ),
        "keywords": ["$710,000", "$820,000", "$2.6 million", "q3", "q4", "22%", "net profit margin"],
    },
    {
        "id": "doc_compliance_10",
        "title": "Regulatory Compliance & Audit Results",
        "text": (
            "The annual SOC 2 Type II audit was completed with zero critical findings. "
            "GDPR compliance review identified 3 minor gaps in data subject access request "
            "processing times. Average DSAR response time is 18 business days against the "
            "30-day regulatory requirement. ISO 27001 certification was renewed."
        ),
        "keywords": ["SOC 2", "GDPR", "DSAR", "18 business days", "ISO 27001", "zero critical"],
    },
    {
        "id": "doc_budget_11",
        "title": "Annual IT Budget Allocation",
        "text": (
            "Total IT budget for the fiscal year is $8.4 million. Infrastructure receives 35% "
            "($2.94M), engineering salaries 40% ($3.36M), cloud services 15% ($1.26M), and "
            "security 10% ($840K). Budget utilization at mid-year was 48%, tracking on plan."
        ),
        "keywords": ["$8.4 million", "35%", "40%", "15%", "10%", "$2.94M", "$3.36M", "budget"],
    },
    {
        "id": "doc_perf_12",
        "title": "Application Performance Metrics",
        "text": (
            "The production API averages 145ms p50 latency and 320ms p99 latency. Error rate "
            "is 0.03% across all endpoints. Peak traffic reaches 8,500 requests per second "
            "during business hours. Database query p95 latency is 45ms. CDN cache hit ratio "
            "is 94%."
        ),
        "keywords": ["145ms", "320ms", "0.03%", "8,500 rps", "45ms", "94%", "latency"],
    },
]


# ---------------------------------------------------------------------------
# BM25 Search Engine
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split on non-word characters, lowercase, drop short tokens."""
    return [w for w in re.split(r"\W+", text.lower()) if len(w) > 1]


_tokenized_corpus = [_tokenize(doc["text"]) for doc in CORPUS]
_bm25_index = BM25Okapi(_tokenized_corpus)


def bm25_search(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Run BM25 search over the corpus. Returns top-k matching documents."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _bm25_index.get_scores(tokens)
    scored = sorted(zip(CORPUS, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"id": doc["id"], "title": doc["title"], "text": doc["text"], "score": round(float(s), 4)}
        for doc, s in scored
        if s > 0.0
    ]


# ---------------------------------------------------------------------------
# System prompt used across training and inference
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an autonomous problem-solving AI agent with access to an internal "
    "document retrieval tool.\n\n"
    "Available tool:\n"
    '1. `bm25_search(query: str, top_k: int = 2)`: Search the internal text '
    "corpus for facts, documents, logs, or technical specs.\n\n"
    "Protocol:\n"
    "1. Always output internal reasoning inside <thought>...</thought> tags "
    "before taking any action.\n"
    "2. To invoke the search tool, output ONLY a JSON block formatted exactly as:\n"
    "```json\n"
    '{"name": "bm25_search", "arguments": {"query": "keyword search terms", "top_k": 2}}\n'
    "```\n"
    "3. Only use the `bm25_search` tool.\n"
    "4. If no tool is needed or all facts have been retrieved, provide your "
    "final response directly.\n"
    "5. If a query is missing essential parameters, ask for clarification directly."
)
