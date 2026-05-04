# Failure Analysis — Lab 18: Production RAG

**Personal (fill in):**
- **Name:** [Họ và tên]
- **Student ID:** [MSSV]
- **Email:** [email@example.com]
- **Date:** [YYYY-MM-DD]
- **Team:** [Tên nhóm]

---

## Summary

Pipeline final scores (production run):

- **Faithfulness:** 1.0000
- **Answer Relevancy:** 0.7675
- **Context Precision:** 0.1026
- **Context Recall:** 1.0000

Observed problem: context precision is very low (0.1026) while recall is perfect (1.0). That means the system returns many context chunks, but only a small fraction are actually useful for the final answer — i.e., high noise in retrieved context.

## Likely Root Causes

1. Chunking too coarse or inconsistent with question granularity — large chunks contain mixed content so retrieved chunks include much irrelevant text.
2. BM25 tokenization (Vietnamese segmentation) or stopword handling produces noisy lexical matches that inflate recall but not precision.
3. Embedding model mismatch / low-quality embeddings for domain Vietnamese content causing dense retrieval to return semantically nearby but not answer-bearing chunks.
4. RRF/hybrid fusion parameters prioritize recall (high top_k) over precision.
5. Reranker thresholds are too permissive (or model not strong enough), so irrelevant candidates pass through.
6. Metadata or document-level duplication: same information repeated across chunks increases recall but not precision.

## Suggested Fixes (actionable)

Short experiments (fast, prioritized):

- EX1 — Reduce chunk size / increase overlap and re-evaluate: split large chunks into smaller, more focused chunks. Acceptance: context_precision increases by at least +0.15.
- EX2 — Tighten retrieval top_k and hybrid fusion: reduce BM25_top_k and DENSE_top_k by half, and lower RRF window. Acceptance: precision ↑ and answer_relevancy not decrease by >0.02.
- EX3 — Improve segmentation for Vietnamese: test `underthesea` vs whitespace tokenization; measure BM25 precision. Acceptance: measurable precision ↑.
- EX4 — Use a stronger embedding model (if resources allow) or fine-tune locally: switch from mini model to multilingual BGE or SBERT large. Acceptance: answer_relevancy ↑ by ≥0.03 and context_precision ↑.
- EX5 — Tighten reranker decision boundary: rerank top 20 and filter by score threshold before answer generation. Acceptance: precision ↑ and faithfulness unchanged.
- EX6 — Deduplicate chunks and add metadata filters (document id, section): dedupe exact/near-duplicate texts before indexing. Acceptance: precision ↑.

Longer experiments (need more time):

- EX7 — Add passage-level dense retriever that jointly encodes (question, passage) and scores directly.
- EX8 — Human-in-the-loop labeling for a small set of retrieval correctness to train a lightweight re-ranker.

## Suggested short-term plan

1. Run EX1 and EX2 (can be done in 1–2 hours).
2. If insufficient, run EX3 and EX5 in parallel.
3. Re-run full pipeline and save `ragas_report.json` for each trial.

## Bottom-5 Failure Templates

Use the template below to record failed examples for later presentation. Fill the fields from `ragas_report.json` (bottom n examples by chosen metric).

### #<rank>
- **Question:**
- **Ground-truth / Expected:**
- **Model Answer:**
- **Context Returned (top 5):**
- **Worst metric:** (e.g., context_precision)
- **Error Tree:** Output correct? → Context correct? → Retrieval correct? → Indexing correct?
- **Root cause hypothesis:**
- **Immediate fix to try:**

## Acceptance Criteria (for this lab)

- Raise `context_precision` from 0.1026 to at least 0.40 for the same test set without reducing `answer_relevancy` by more than 0.05.

## Notes / Who to contact

- **Experiment owner:** [Họ và tên — người chịu trách nhiệm thử nghiệm này]
- **Data file:** `test_set.json` and `data/sample.md`
- **Report output:** `ragas_report.json` (in repo root)

