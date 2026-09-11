# V1 — Accuracy Improvement Methods for Agentic Small LM

## What Changed from Original

This directory contains an **upgraded pipeline** for the Qwen 3B BM25 tool-use agent.
All original files in `small_llm/` are untouched.

---

## Summary of Improvements Implemented

### 1. `corpus.py` — Hybrid BM25 + Semantic Retrieval
- Added sentence-transformer-based semantic search (all-MiniLM-L6-v2)
- Reciprocal Rank Fusion (RRF) to merge BM25 and semantic scores
- Falls back to pure BM25 if sentence-transformers is unavailable

### 2. `prepare_data_3b.py` — Scaled & Diversified Training Data
- Target 2,000+ trajectories (up from ~500)
- Better chain-of-thought reasoning (specific, analytical — not generic)
- Added error-recovery trajectories (bad search → retry)
- Added adversarial/out-of-corpus trajectories
- Added multi-hop (3-step) trajectories
- More aggressive query paraphrasing

### 3. `train_sft.py` — SFT with Curriculum Learning
- Stage 1: Single-step + no-tool examples (2 epochs)
- Stage 2: Multi-step + clarification examples (2 epochs)
- Stage 3: Full dataset including adversarial/error-recovery (1 epoch)

### 4. `train_dpo.py` — DPO Preference Optimization (NEW)
- Generates paired preference data (chosen vs rejected)
- Uses trl DPOTrainer for sample-efficient preference learning
- Replaces reward-weighted SFT with explicit contrastive learning

### 5. `train_grpo.py` — GRPO On-Policy RL (Replaces old on_policy_train)
- Samples G=4 rollouts per task (group relative)
- Computes advantage as (reward - mean) / std
- Improved reward function with faithfulness & query quality scoring
- KL divergence penalty against reference model

### 6. `infer.py` — Inference with Best-of-N Sampling
- Best-of-N mode (sample N candidates, pick highest reward)
- Self-consistency / majority voting mode
- Standard single-pass mode (default)

### 7. `eval.py` — Evaluation Framework (NEW)
- Held-out test set with ground-truth answers
- Metrics: retrieval precision@1, keyword F1, format compliance, thought quality
- JSON results output for tracking experiments

---

## Recommended Training Order

```
1. pip install sentence-transformers trl    # new dependencies
2. python prepare_data_3b.py                # generate 2K+ trajectories
3. python train_sft.py                      # curriculum SFT
4. python train_dpo.py                      # DPO preference tuning
5. python train_grpo.py                     # GRPO self-improvement
6. python eval.py                           # measure accuracy
7. python infer.py                          # interactive testing
```

## New Dependencies

```
pip install sentence-transformers trl
```
