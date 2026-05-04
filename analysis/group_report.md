# Group Report — Lab 18: Production RAG

**Nhóm:** ABC_Hoang
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Lê Minh Hoàng | M1: Chunking | ✅ | 8/8 |
| Lê Minh Hoàng | M2: Hybrid Search | ✅ | 5/5 |
| Lê Minh Hoàng | M3: Reranking | ✅ | 5/5 |
| Lê Minh Hoàng | M4: Evaluation, M5: Enrichment | ✅ | 5/5 |

## Kết quả RAGAS (final run)

| Metric | Naive | Production | Δ |
|--------|-------:|----------:|---:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.7615 | 0.7675 | +0.0061 |
| Context Precision | 0.0798 | 0.1026 | +0.0229 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Key Findings

1. **Biggest improvement:** Small but consistent increase in answer relevancy thanks to hybrid retrieval + reranker.
2. **Biggest challenge:** Low context precision — too many noisy retrieved chunks despite perfect recall.
3. **Surprise finding:** Deterministic offline reranker + chunking heuristics produced reliable faithfulness scores.

## Reproducibility (how to run)

1. Create and activate environment (conda):

```powershell
D:/Miniconda/Scripts/conda.exe create -n day18 python=3.11 -y
D:/Miniconda/Scripts/conda.exe run -n day18 pip install -r requirements.txt
```

2. Run full pipeline:

```powershell
conda run -n day18 python main.py
```

3. Reports are written to `ragas_report.json` and `naive_baseline_report.json` in the repo root; move/copy into `reports/` for submission.

## Presentation Notes (2–3 bullets)

- Start: goal — move from naive baseline to production RAG (hybrid retrieval + rerank + enrichment).
- Metric highlights: faithfulness stable at 1.0; answer relevancy improved slightly; context precision remains the priority to fix.
- Case study: show one low-precision example and walk through the Error Tree to justify next experiments.

## Next steps (prioritized)

1. Short experiments: chunking granularity + retrieval top_k tuning (EX1, EX2 from Failure Analysis).
2. Improve Vietnamese segmentation and evaluate BM25 vs dense-only retrieval.
3. If time: try stronger embedding model or lightweight supervised reranker on a small labeled set.

## Notes / Artifacts

- Code: `src/m1_chunking.py`, `src/m2_search.py`, `src/m3_rerank.py`, `src/m4_eval.py`, `src/m5_enrichment.py`
- Test files: `tests/test_m1.py`, `tests/test_m3.py`, `tests/test_m4.py`, `tests/test_m5.py`
- Data: `data/sample.md`, `test_set.json`

