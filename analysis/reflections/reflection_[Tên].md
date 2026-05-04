# Personal Reflection — Lab 18: Production RAG

**Personal info (fill in):**

- **Name:** [Họ và tên]
- **Student ID:** [MSSV]
- **Email:** [email@example.com]
- **Date:** [YYYY-MM-DD]
- **Role in project / Modules owned:** [e.g., M1 Chunking]

---

## 1) Short summary of my contributions (3–5 lines)

[Tóm tắt công việc bạn làm: module, chính sách, chức năng đã implement, tests bạn viết]

## 2) What I learned (technical and teamwork)

- Technical: [e.g., chunking strategies, hybrid retrieval, reranking, offline evaluation]
- Teamwork: [e.g., coordinating test ownership, sharing models/checkpoints]

## 3) Biggest challenge I faced and how I solved it

[Miêu tả ngắn gọn vấn đề lớn nhất, debug steps, outcome]

## 4) If I had another hour, I would

- [Prioritized next experiment you would run — e.g., reduce chunk size + retune top_k]

## 5) Repro commands I ran locally

```powershell
conda run -n day18 python main.py
pytest tests/test_m1.py
```

## 6) Files I changed / key functions I implemented

- `src/m1_chunking.py` — `chunk_structure_aware()`, `chunk_semantic()`, `compare_strategies()`
- `src/m2_search.py` — BM25, Dense (Qdrant upsert/query_points), `reciprocal_rank_fusion()`
- `src/m3_rerank.py` — `CrossEncoderReranker.rerank()` with deterministic fallback
- `src/m4_eval.py` — `evaluate_ragas()` deterministic implementation
- `src/m5_enrichment.py` — summarization, hypothesis QA generation, contextual prepend

## 7) Time spent (estimate)

- [hours total]

## 8) Anything else to add

- [Links to demos, notes, suggestions for graders]
