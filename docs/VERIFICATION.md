# Verification record

Local verification on Windows, Python 3.12.14, 2026-10-27.

- 37 automated tests passed. Includes real SQLite indexing/reindexing, transaction preservation on invalid documents, API validation, oversized and chunked bodies, cross-origin/Host checks, rate/concurrency boundaries, provider timeouts/errors, malformed or fabricated citations and duplicate citations.
- Authored retrieval regression: 15/15 cases; expected source within top four for 12/12 positive cases; no results for 3/3 unrelated cases. This is a small development regression set, not a held-out benchmark. See the complete report in `evals/report.json`.
- Browser check: VPN search shows source excerpts and revisions; issue + impact creates an unsubmitted draft; no-match state works; desktop 1280px and phone 390px layouts reviewed. No horizontal overflow at the phone breakpoint.
- No live OpenAI request: protocol and failure tests use `httpx.MockTransport`. They do not establish model accuracy or real-account compatibility.
- A test exposed an unclosed SQLite connection on Windows. Explicit connection closure fixed it, and the complete suite was rerun.

CI repeats formatting, linting, tests and retrieval evaluation on Ubuntu and Windows using Python 3.12. Workflow badges report actual remote status; this document does not assume a future run succeeds.
