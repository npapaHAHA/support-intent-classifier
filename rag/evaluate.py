"""Small, authored offline benchmark; this is not an online A/B experiment."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from rag.generation import GENERATION_MODEL, GENERATION_REVISION, LocalGenerator
from rag.retrieval import EMBEDDING_MODEL, EMBEDDING_REVISION, ROOT, Retriever, load_documents
from rag.service import SupportAssistant


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_threshold(cases: list[dict], scores: list[float]) -> float:
    """Maximize balanced answerability accuracy on development questions only."""
    labels = np.array([case["document_id"] is not None for case in cases])
    if not labels.any() or labels.all() or len(scores) != len(labels):
        raise ValueError("Calibration needs scores for both supported and unsupported questions")
    unique = sorted(set(scores))
    candidates = [0.0, 1.0] + [
        (a + b) / 2 for a, b in zip(unique, unique[1:]) if 0 <= (a + b) / 2 <= 1
    ]

    def objective(threshold):
        predicted = np.asarray(scores) >= threshold
        return (np.mean(predicted[labels]) + np.mean(~predicted[~labels])) / 2, threshold

    return float(max(candidates, key=objective))


def retrieval_metrics(cases: list[dict], rankings: list[list], threshold: float) -> dict:
    supported = [(case, hits) for case, hits in zip(cases, rankings) if case["document_id"]]
    ranks = []
    for case, hits in supported:
        ids = [hit.document.id for hit in hits]
        ranks.append(ids.index(case["document_id"]) + 1 if case["document_id"] in ids else 0)
    answerable = np.array([case["document_id"] is not None for case in cases])
    accepted = np.array([hits[0].score >= threshold for hits in rankings])
    return {
        "supported_questions": len(supported),
        "recall_at_1": float(np.mean([rank == 1 for rank in ranks])),
        "recall_at_3": float(np.mean([1 <= rank <= 3 for rank in ranks])),
        "mrr_at_3": float(np.mean([1 / rank if rank else 0 for rank in ranks])),
        "answerability_accuracy": float(np.mean(answerable == accepted)),
        "unsupported_rejection_rate": float(np.mean(~accepted[~answerable])),
        "supported_acceptance_rate": float(np.mean(accepted[answerable])),
    }


def paired_bootstrap(first: list[bool], second: list[bool], seed: int = 42) -> dict:
    """Paired percentile bootstrap over questions, second minus first."""
    if not first or len(first) != len(second):
        raise ValueError("Expected equally sized nonempty paired outcomes")
    differences = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, (5000, len(differences)), replace=True).mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "ci_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "resamples": 5000,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", action="store_true", help="Also run both LLM prompts")
    parser.add_argument("--output", type=Path, default=ROOT / "rag/reports/evaluation.json")
    args = parser.parse_args()
    documents = load_documents()
    dev = load_cases(ROOT / "rag/data/dev.json")
    test = load_cases(ROOT / "rag/data/test.json")
    if {case["question"].casefold() for case in dev} & {case["question"].casefold() for case in test}:
        raise ValueError("Development and test questions overlap")
    known = {doc.id for doc in documents}
    if any(case["document_id"] not in known | {None} for case in dev + test):
        raise ValueError("Unknown relevance label")
    report = {
        "dataset": "hand-authored fictional banking support, not BANKING77 or Sber policies",
        "dev_questions": len(dev), "test_questions": len(test),
        "data_sha256": {
            name: hashlib.sha256((ROOT / "rag/data" / name).read_bytes()).hexdigest()
            for name in ("knowledge.json", "dev.json", "test.json")
        },
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ("torch", "transformers", "sentence-transformers", "scikit-learn")
        }},
        "models": {"embedding": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
                   "generation": GENERATION_MODEL, "generation_revision": GENERATION_REVISION},
        "retrieval": {},
    }
    dense = None
    for backend in ("tfidf", "dense"):
        retriever = Retriever(documents, backend=backend)
        scores = [retriever.search(case["question"])[0].score for case in dev]
        threshold = choose_threshold(dev, scores)
        rankings = [retriever.search(case["question"]) for case in test]
        report["retrieval"][backend] = {
            "threshold_selected_on_dev": threshold,
            **retrieval_metrics(test, rankings, threshold),
            "predictions": [
                {"id": case["id"], "expected": case["document_id"],
                 "hits": [{"id": hit.document.id, "score": hit.score} for hit in hits]}
                for case, hits in zip(test, rankings)
            ],
        }
        print(backend, {k: v for k, v in report["retrieval"][backend].items() if k != "predictions"}, flush=True)
        if backend == "dense":
            dense = retriever
            (ROOT / "rag/config.json").write_text(json.dumps({
                "threshold": threshold,
                "selection": "Balanced answerability accuracy on dev.json only; no test tuning",
            }, indent=2) + "\n", encoding="utf-8")
    if args.generation:
        assistant = SupportAssistant(dense, LocalGenerator(), report["retrieval"]["dense"]["threshold_selected_on_dev"])
        outcomes = {}
        report["generation"] = {}
        for variant in ("plain", "grounded"):
            records, success = [], []
            for case in test:
                result = assistant.answer(case["question"], variant)
                # Keyword coverage is a transparent weak proxy, not factual correctness.
                passed = bool(case["document_id"]) and not result["abstained"] and all(
                    term.casefold() in result["answer"].casefold() for term in case.get("answer_terms", [])
                )
                if case["document_id"]:
                    success.append(passed)
                records.append({"id": case["id"], "question": case["question"],
                                "expected_document": case["document_id"],
                                "keyword_coverage_pass": passed, **result})
                print(variant, case["id"], result["answer"][:100], flush=True)
            outcomes[variant] = success
            report["generation"][variant] = {
                "keyword_coverage_rate_on_supported": float(np.mean(success)),
                "predictions": records,
            }
        report["prompt_comparison"] = {
            "metric": "all annotated action terms present on 18 supported questions; not factuality",
            "design": "paired offline comparison, no production traffic or random user assignment",
            **paired_bootstrap(outcomes["plain"], outcomes["grounded"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
