"""Small authored retrieval regression set, NOT a live LLM quality benchmark."""

import json
import tempfile
from pathlib import Path

from assistant.retrieval import KnowledgeBase


def evaluate():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evals/cases.json").read_text())
    with tempfile.TemporaryDirectory() as temp:
        kb = KnowledgeBase(Path(temp) / "eval.sqlite")
        kb.ingest(root / "knowledge")
        rows = []
        for case in cases:
            hits = kb.search(case["question"])
            expected = case["expected_source"]
            passed = (
                any(h["source"] == expected for h in hits) if expected else not hits
            )
            rows.append(
                {**case, "retrieved": [h["source"] for h in hits], "passed": passed}
            )
    positives = [r for r in rows if r["expected_source"]]
    negatives = [r for r in rows if not r["expected_source"]]
    return {
        "scope": "Authored English lexical retrieval regression; no live LLM calls",
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "recall_at_4": sum(r["passed"] for r in positives) / len(positives),
        "no_match_accuracy": sum(r["passed"] for r in negatives) / len(negatives),
        "results": rows,
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["cases"] else 1)
