import json
import time
from pathlib import Path

import httpx
import pytest

from evals.live import load_cases, plan_live, run_live, run_offline

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "direct",
    "paraphrase",
    "vietnamese",
    "negation",
    "contradiction",
    "no_answer",
}


def corpus_sources():
    return {p.name for p in (ROOT / "knowledge").glob("*.md")}


def test_heldout_dataset_has_60_cases_with_valid_labels_across_six_families():
    cases = load_cases()
    assert len(cases) == 60
    families = [c["family"] for c in cases]
    assert set(families) <= FAMILIES
    for name in FAMILIES:
        assert families.count(name) == 10, name
    for case in cases:
        if case["expect_abstain"]:
            assert case["expected_source"] is None
        else:
            assert case["expected_source"] in corpus_sources(), case


def test_offline_runner_scores_recall_mrr_and_abstain_groups_with_denominators():
    def retriever(question):
        if question == "a hit":
            return [
                {
                    "source": "vpn.md",
                    "id": "vpn:1",
                    "body": "x",
                    "revision": "r1",
                    "title": "t",
                }
            ]
        return []

    rows = run_offline(
        retriever=retriever,
        cases=[
            {
                "family": "direct",
                "question": "a hit",
                "expected_source": "vpn.md",
                "expect_abstain": False,
            },
            {
                "family": "direct",
                "question": "missing",
                "expected_source": "vpn.md",
                "expect_abstain": False,
            },
            {
                "family": "contradiction",
                "question": "a hit",
                "expected_source": None,
                "expect_abstain": True,
                "distractor_source": "vpn.md",
            },
            {
                "family": "no_answer",
                "question": "missing",
                "expected_source": None,
                "expect_abstain": True,
            },
        ],
    )
    assert rows["denominators"] == {
        "positives": 2,
        "abstain_expected": 2,
        "total": 4,
        "errors": 0,
    }
    assert rows["recall_at_4"] == 0.5
    assert rows["mrr"] == 0.5
    assert rows["correct_abstain_rate"] == 0.5
    assert rows["false_lead_rate"] == 0.5
    families = {f: {**v} for f, v in rows["families"].items()}
    assert families["direct"]["n"] == 2 and families["direct"]["hits"] == 1
    assert (
        rows["latency_ms"]["p50"] is not None and rows["latency_ms"]["p95"] is not None
    )


def test_missing_results_from_failures_are_errors_inside_the_denominator():
    def timeout_retriever(question):
        time.sleep(0.01)
        raise TimeoutError("injected by test")

    rows = run_offline(
        retriever=timeout_retriever,
        cases=[
            {
                "family": "direct",
                "question": "a",
                "expected_source": "vpn.md",
                "expect_abstain": False,
            },
            {
                "family": "no_answer",
                "question": "b",
                "expected_source": None,
                "expect_abstain": True,
            },
        ],
    )
    assert rows["denominators"]["errors"] == 2
    assert all(row["outcome"] == "error_timeout" for row in rows["rows"])
    assert rows["recall_at_4"] == 0.0 and rows["correct_abstain_rate"] == 0.0


def test_live_evaluation_is_strictly_opt_in_and_never_dials_without_a_key(monkeypatch):
    called = []

    def factory():
        called.append(True)
        raise AssertionError("no provider may be built when live is disabled")

    monkeypatch.delenv("SUPPORT_LIVE_EVAL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan = plan_live(key=None, factory=factory)
    assert plan["live"] is False and plan["reason"] == "not_opted_in" and called == []

    monkeypatch.setenv("SUPPORT_LIVE_EVAL", "1")
    plan = plan_live(key=None, factory=factory)
    assert plan["live"] is False and plan["reason"] == "missing_key" and called == []

    monkeypatch.setenv("SUPPORT_LIVE_EVAL", "1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "real-key")
    plan = plan_live(key=None, factory=factory)
    assert plan["live"] is True and plan["reason"] == "opted_in"


def test_live_run_stops_at_request_cap_and_records_every_case_in_the_sample():
    class AlwaysTimeout:
        def select(self, *a, **k):
            raise httpx.ReadTimeout("upstream timed out")

    cases = [
        {
            "family": "direct",
            "question": f"q{i}",
            "expected_source": "vpn.md",
            "expect_abstain": False,
        }
        for i in range(5)
    ]
    rows = run_live(
        retriever=lambda q: [
            {
                "source": "vpn.md",
                "id": "vpn:1",
                "revision": "r1",
                "body": "x",
                "title": "t",
            }
        ],
        provider_factory=lambda: AlwaysTimeout(),
        cases=cases,
        max_requests=2,
        max_seconds=60,
    )
    assert rows["requests_used"] <= 2 and rows["status"] == "ran"
    assert len(rows["rows"]) == 5, (
        "cases cut by the cap stay in the sample, counted, not dropped"
    )
    assert [r["outcome"] for r in rows["rows"]].count("error_request_cap") == 3
    assert rows["denominators"]["errors"] == 5
    assert rows["latency_ms"]["p50"] is not None


def test_live_run_validates_citations_through_the_current_provenance_checks():
    class Fabricator:
        def select(self, *a, **k):
            from assistant.provider import Decision

            return Decision.model_validate(
                {
                    "sufficient": True,
                    "evidence": [
                        {
                            "chunk_id": "vpn:9",
                            "quote": "A fabricated quote never present in any indexed body text here.",
                        }
                    ],
                }
            )

    rows = run_live(
        retriever=lambda q: [
            {
                "source": "vpn.md",
                "id": "vpn:1",
                "revision": "r1",
                "body": "Some real body text.",
                "title": "t",
            }
        ],
        provider_factory=Fabricator,
        cases=[
            {
                "family": "direct",
                "question": "q",
                "expected_source": "vpn.md",
                "expect_abstain": False,
            }
        ],
        max_requests=5,
        max_seconds=60,
    )
    assert rows["rows"][0]["outcome"] == "error_provenance", (
        "fabricated quotes fail the live validator"
    )
    assert rows["answered_correct"] == 0


def test_live_run_keeps_no_secret_and_reports_config_without_claims_of_accuracy():
    class Stubbish:
        def select(self, *a, **k):
            from assistant.provider import Decision

            return Decision.model_validate({"sufficient": False, "evidence": []})

    rows = run_live(
        retriever=lambda q: [],
        provider_factory=Stubbish,
        cases=[
            {
                "family": "no_answer",
                "question": "q",
                "expected_source": None,
                "expect_abstain": True,
            }
        ],
        max_requests=5,
        max_seconds=60,
        model="test-model",
    )
    assert rows["config"]["model"] == "test-model"
    assert not any("key" in json.dumps(row) for row in rows["rows"])
    assert rows["status"] == "ran"


def test_heldout_docs_note_never_claims_model_accuracy_offline():
    report = run_offline(cases=load_cases())
    assert "no live LLM calls" in report["scope"]
    assert report["offline"] is True


# Imported late because service validates once; avoid double work at import time.
from assistant.service import validate_evidence  # noqa: E402


def test_validate_evidence_rejects_inconsistent_and_fabricated_selections():
    from assistant.provider import Decision

    chunks = [
        {
            "id": "vpn:1",
            "body": "Real sentence from the corpus.",
            "source": "vpn.md",
            "revision": "r",
            "title": "t",
        }
    ]
    with pytest.raises(Exception):
        validate_evidence(
            Decision.model_validate(
                {
                    "sufficient": False,
                    "evidence": [{"chunk_id": "vpn:1", "quote": "Real sentence,"}],
                }
            ),
            chunks,
        )
    evidence = validate_evidence(
        Decision.model_validate(
            {
                "sufficient": True,
                "evidence": [
                    {"chunk_id": "vpn:1", "quote": "Real sentence from the corpus."}
                ],
            }
        ),
        chunks,
    )
    assert (
        evidence[0]["id"] == "vpn:1"
        and evidence[0]["quote"] == "Real sentence from the corpus."
    )
