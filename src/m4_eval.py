"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    def _tokens(text: str) -> set[str]:
        return {token.strip(".,!?;:\"'()[]{}<>\n\t").lower() for token in text.split() if token.strip()}

    per_question = []
    faithfulness_scores = []
    answer_relevancy_scores = []
    context_precision_scores = []
    context_recall_scores = []

    for question, answer, question_contexts, ground_truth in zip(questions, answers, contexts, ground_truths):
        q_tokens = _tokens(question)
        a_tokens = _tokens(answer)
        gt_tokens = _tokens(ground_truth)
        ctx_tokens = _tokens(" ".join(question_contexts))

        answer_relevancy = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
        faithfulness = len(a_tokens & ctx_tokens) / max(len(a_tokens), 1)
        context_precision = len(gt_tokens & ctx_tokens) / max(len(ctx_tokens), 1)
        context_recall = len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1)

        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=question_contexts,
            ground_truth=ground_truth,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        ))

        faithfulness_scores.append(faithfulness)
        answer_relevancy_scores.append(answer_relevancy)
        context_precision_scores.append(context_precision)
        context_recall_scores.append(context_recall)

    count = max(len(per_question), 1)
    return {
        "faithfulness": sum(faithfulness_scores) / count,
        "answer_relevancy": sum(answer_relevancy_scores) / count,
        "context_precision": sum(context_precision_scores) / count,
        "context_recall": sum(context_recall_scores) / count,
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for result in eval_results:
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        worst_score = scores[worst_metric]

        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
        else:
            diagnosis = "Low retrieval quality"
            suggested_fix = "Inspect chunking, retrieval, and evaluation inputs"

        scored.append({
            "question": result.question,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored.sort(key=lambda item: item["avg_score"])
    return [{k: v for k, v in item.items() if k != "avg_score"} for item in scored[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
