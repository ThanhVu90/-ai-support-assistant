"""Held-out retrieval evaluation and strictly opt-in live model probe.

The offline held-out runner needs no credentials and makes no network calls:
it scores the manually labeled dataset in ``evals/heldout.json`` against the
lexical retriever and reports denominators, per-family breakdowns and
latency percentiles. Results that fail (including retriever timeouts) stay in
the sample as counted errors instead of being dropped.

A live model probe runs ONLY when SUPPORT_LIVE_EVAL=1 with an API key and an
explicit model. It stops at the request cap and the wall-clock budget cap,
keeps every unanswered case in the sample, and validates every citation with
the same provenance gate the application uses. The API key is read from the
environment only; it is never printed, never sent anywhere besides the
provider endpoint, and never reaches a public CI run because CI never sets
SUPPORT_LIVE_EVAL.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

SCHEMA = "heldout-v1"
FAMILIES = {
    "direct",
    "paraphrase",
    "vietnamese",
    "negation",
    "contradiction",
    "no_answer",
}


def load_cases() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "evals" / "heldout.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    for case in cases:
        if case["expect_abstain"]:
            case.setdefault("distractor_source")
    return cases


def _percentile(samples: list[float], quantile: float) -> float | None:
    """Linear-interpolated percentile, quantile in [0, 1]."""
    ordered = sorted(samples)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _score_hit(hits: list[dict], expected_source: str | None) -> int:
    """Rank (1-based) of the expected source within the delivered top-k, else 0."""
    if expected_source is None:
        return 0
    for position, hit in enumerate(hits, 1):
        if hit["source"] == expected_source:
            return position
    return 0


def run_offline(
    retriever=None, cases: list[dict] | None = None, corpus: Path | None = None
) -> dict:
    """Score the held-out set against a retriever. No credentials, no network."""
    cases = cases if cases is not None else load_cases()
    if retriever is None:
        from assistant.retrieval import KnowledgeBase

        @contextmanager
        def _real():
            with __import__("tempfile").TemporaryDirectory() as temp:
                kb = KnowledgeBase(Path(temp) / "heldout.sqlite")
                kb.ingest(corpus or Path(__file__).resolve().parents[1] / "knowledge")
                yield kb.search

        with _real() as search:
            return run_offline(retriever=search, cases=cases)
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            hits = retriever(case["question"])
            rank = _score_hit(hits, case["expected_source"])
            row = {
                **{
                    key: case.get(key)
                    for key in (
                        "family",
                        "question",
                        "expected_source",
                        "distractor_source",
                    )
                },
                "outcome": (
                    "hit" if rank > 0 else "silent" if not hits else "wrong_source"
                ),
                "rank": rank,
                "retrieved": [hit["source"] for hit in hits],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except TimeoutError:
            row = {
                **{
                    key: case.get(key)
                    for key in (
                        "family",
                        "question",
                        "expected_source",
                        "distractor_source",
                    )
                },
                "outcome": "error_timeout",
                "rank": 0,
                "retrieved": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except (
            Exception
        ) as failure:  # any other retriever failure is counted, never dropped
            row = {
                **{
                    key: case.get(key)
                    for key in (
                        "family",
                        "question",
                        "expected_source",
                        "distractor_source",
                    )
                },
                "outcome": f"error_{type(failure).__name__.lower()}",
                "rank": 0,
                "retrieved": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        rows.append(row)
    return _summarize(rows, offline=True)


def _summarize(rows: list[dict], offline: bool) -> dict:
    positives = [row for row in rows if row["expected_source"]]
    abstains = [row for row in rows if not row["expected_source"]]
    errors = [
        row
        for row in rows
        if str(row["outcome"]).startswith("error")
        or row["outcome"] == "error_request_cap"
    ]
    hits = [row for row in positives if 0 < row["rank"] <= 4]
    reciprocal = [1 / row["rank"] for row in positives if row["rank"] > 0]
    mrr = round(sum(reciprocal) / len(positives), 4) if positives else 0.0
    silent_perfect = [row for row in abstains if row["outcome"] == "silent"]
    false_leads = [
        row
        for row in abstains
        if row["outcome"] not in ("silent",)
        and not str(row["outcome"]).startswith("error")
    ]
    families: dict[str, dict] = {}
    for row in rows:
        stats = families.setdefault(
            row["family"],
            {"n": 0, "errors": 0, "hits": 0, "silent_correct": 0, "false_leads": 0},
        )
        stats["n"] += 1
        if (
            str(row["outcome"]).startswith("error")
            or row["outcome"] == "error_request_cap"
        ):
            stats["errors"] += 1
        elif row["expected_source"]:
            if 0 < row["rank"] <= 4:
                stats["hits"] += 1
        elif row["outcome"] == "silent":
            stats["silent_correct"] += 1
        else:
            stats["false_leads"] += 1
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    return {
        "schema": SCHEMA,
        "scope": (
            "Manual labeled held-out scoring over the lexical retriever; no live LLM calls. "
            "Citation quality is assessed by a human reviewer using the per-case notes."
            if offline
            else "Live model probe: retrieval + provider selection + the application's provenance gate; results include errors."
        ),
        "offline": offline,
        "cases": len(rows),
        "denominators": {
            "total": len(rows),
            "positives": len(positives),
            "abstain_expected": len(abstains),
            "errors": len(errors),
        },
        "recall_at_4": len(hits) / len(positives) if positives else 0.0,
        "mrr": round(mrr, 4),
        "correct_abstain_rate": len(silent_perfect) / len(abstains)
        if abstains
        else 0.0,
        "false_lead_rate": len(false_leads) / len(abstains) if abstains else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "families": families,
        "rows": rows,
        "config": {
            "labels": "manual-2026-10-29",
            "retriever": "lexical-fts5-unicode61",
            "topk_limit": 4,
            "note": "No accuracy claims are made for the model; this tool measures and reports.",
        },
    }


def plan_live(key: str | None = None, factory=None, model: str | None = None):
    """Decide, without side effects, whether a live probe may run.

    Order matters: opt-in first, then key, then model. No provider is built
    unless all three are satisfied, so a missing key cannot cause a request.
    """
    import os

    model = model or os.environ.get("OPENAI_MODEL")
    opted = os.environ.get("SUPPORT_LIVE_EVAL") == "1"
    key = key if key is not None else os.environ.get("OPENAI_API_KEY")
    if not opted:
        return {"live": False, "reason": "not_opted_in", "model": None, "factory": None}
    if not key:
        return {"live": False, "reason": "missing_key", "model": model, "factory": None}
    if not model:
        return {
            "live": False,
            "reason": "missing_model",
            "model": None,
            "factory": None,
        }
    if factory is None:
        factory = _default_factory
    return {"live": True, "reason": "opted_in", "model": model, "factory": factory}


def _default_factory():
    import os

    from assistant.provider import OpenAIProvider

    return OpenAIProvider(os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"])


def run_live(
    retriever,
    provider_factory,
    cases: list[dict],
    max_requests: int | None = None,
    max_seconds: float | None = None,
    model: str | None = None,
) -> dict:
    """Probe the real model on the held-out sample under hard caps.

    Every case stays in the sample: timed-out or capped-out cases are counted
    as errors or marked not_run, never silently dropped. Citations confirmed by
    the provider pass through the same provenance gate used in production.
    """
    import os

    max_requests = (
        max_requests
        if max_requests is not None
        else int(os.environ.get("SUPPORT_LIVE_MAX_REQUESTS", "12"))
    )
    max_seconds = (
        max_seconds
        if max_seconds is not None
        else float(os.environ.get("SUPPORT_LIVE_MAX_SECONDS", "60"))
    )
    deadline = time.perf_counter() + max_seconds
    provider = provider_factory()
    rows = []
    requests_used = 0
    answered_correct = 0
    errors = 0
    for case in cases:
        if requests_used >= max_requests:
            rows.append(
                {
                    **{
                        key: case.get(key)
                        for key in ("family", "question", "expected_source")
                    },
                    "outcome": "error_request_cap",
                    "retrieved": [],
                    "latency_ms": None,
                }
            )
            continue
        chunks = retriever(case["question"])
        started = time.perf_counter()
        if started >= deadline:
            rows.append(
                {
                    **{
                        key: case.get(key)
                        for key in ("family", "question", "expected_source")
                    },
                    "outcome": "error_time_budget",
                    "retrieved": [],
                    "latency_ms": None,
                }
            )
            continue
        requests_used += 1
        try:
            from assistant.provider import ProviderError
            from assistant.service import validate_evidence

            decision = provider.select(case["question"], chunks)
            evidence = validate_evidence(decision, chunks)
            elapsed = (time.perf_counter() - started) * 1000
            if case["expect_abstain"]:
                outcome = (
                    "abstained_correct"
                    if not evidence
                    else "answered_when_abstain_expected"
                )
                answered_correct += not evidence
            else:
                ranked_hit = [
                    chunk
                    for chunk in chunks
                    if chunk["source"] == case["expected_source"]
                ]
                outcome = (
                    "answered_correct" if evidence and ranked_hit else "wrong_source"
                )
                answered_correct += bool(evidence and ranked_hit)
            rows.append(
                {
                    **{
                        key: case.get(key)
                        for key in ("family", "question", "expected_source")
                    },
                    "outcome": outcome,
                    "retrieved": [chunk["source"] for chunk in chunks],
                    "latency_ms": round(elapsed, 3),
                }
            )
        except ProviderError:
            errors += 1
            rows.append(
                {
                    **{
                        key: case.get(key)
                        for key in ("family", "question", "expected_source")
                    },
                    "outcome": "error_provenance",
                    "retrieved": [chunk["source"] for chunk in chunks],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        except (httpx.HTTPError, TimeoutError):
            errors += 1
            rows.append(
                {
                    **{
                        key: case.get(key)
                        for key in ("family", "question", "expected_source")
                    },
                    "outcome": "error_timeout",
                    "retrieved": [],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    positives = [row for row in rows if row["expected_source"]]
    abstains = [row for row in rows if not row["expected_source"]]
    hits = [row for row in positives if row["outcome"] == "answered_correct"]
    silent_perfect = [row for row in abstains if row["outcome"] == "abstained_correct"]
    return {
        "schema": SCHEMA,
        "status": "ran",
        "offline": False,
        "scope": "Live probe stopped at request/time caps; nothing from the sample was dropped.",
        "cases": len(rows),
        "requests_used": requests_used,
        "max_requests": max_requests,
        "max_seconds": max_seconds,
        "answered_correct": answered_correct,
        "denominators": {
            "total": len(rows),
            "positives": len(positives),
            "abstain_expected": len(abstains),
            "errors": len(
                [row for row in rows if str(row["outcome"]).startswith("error")]
            ),
        },
        "recall_at_4_live": len(hits) / len(positives) if positives else 0.0,
        "correct_abstain_rate_live": len(silent_perfect) / len(abstains)
        if abstains
        else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "config": {
            "model": model,
            "labels": "manual-2026-10-29",
            "note": "No accuracy claims: outcomes include errors and are bounded by the caps.",
        },
        "rows": rows,
    }


def main() -> None:

    report = run_offline()
    plan = plan_live()
    if plan["live"]:
        live = run_live(
            retriever=_offline_knowledge_retriever(),
            provider_factory=plan["factory"],
            cases=load_cases(),
            model=plan["model"],
        )
        report["live"] = live
    else:
        report["live"] = {
            "live": False,
            "reason": plan["reason"],
            "model": plan["model"],
        }
    print(json.dumps(report, indent=2))
    raise SystemExit(0)


def _offline_knowledge_retriever():
    import tempfile

    from assistant.retrieval import KnowledgeBase

    context = tempfile.TemporaryDirectory()
    kb = KnowledgeBase(Path(context.name) / "live.sqlite")
    kb.ingest(Path(__file__).resolve().parents[1] / "knowledge")
    return kb.search


if __name__ == "__main__":
    main()
