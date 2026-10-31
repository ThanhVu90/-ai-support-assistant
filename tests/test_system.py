import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from assistant.app import create_app
from assistant.provider import Decision, OpenAIProvider, ProviderError
from assistant.retrieval import KnowledgeBase
from assistant.service import SupportService

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"X-Support-Client": "web"}


@pytest.fixture
def kb(tmp_path):
    result = KnowledgeBase(tmp_path / "kb.sqlite")
    result.ingest(ROOT / "knowledge")
    return result


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "api.sqlite", use_env=False)) as value:
        yield value


def test_search_retrieves_specific_error(kb):
    assert kb.search("VPN error 809")[0]["source"] == "vpn.md"


def test_unknown_terms_return_nothing(kb):
    assert kb.search("intergalactic xyzabc") == []


@pytest.mark.parametrize(
    "question", ['" OR * NEAR(', "***", "'; DROP TABLE chunks;--", "a the i"]
)
def test_search_grammar_cannot_be_injected(kb, question):
    kb.search(question)
    assert len(kb.catalog()) == 8


def test_ingestion_idempotent_and_removes_deleted_documents(kb, tmp_path):
    original = kb.ingest(ROOT / "knowledge")
    assert kb.ingest(ROOT / "knowledge") == original
    d = tmp_path / "docs"
    d.mkdir()
    (d / "single.md").write_text(
        "# Only\n\nUnique replacement document.", encoding="utf-8"
    )
    kb.ingest(d)
    assert len(kb.catalog()) == 1
    assert kb.search("VPN") == []


def test_failed_ingestion_preserves_previous_index(kb, tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "bad.md").write_text("# Bad\n\n" + "x" * 2001)
    with pytest.raises(ValueError):
        kb.ingest(d)
    assert len(kb.catalog()) == 8


@pytest.mark.parametrize(
    ("name", "content"),
    [("empty.md", ""), ("blank.md", "   \n\t ")],
)
def test_empty_knowledge_file_rejected_naming_the_file(kb, tmp_path, name, content):
    d = tmp_path / "edge"
    d.mkdir()
    (d / "valid.md").write_text(
        "# Valid\n\nA valid replacement paragraph.", encoding="utf-8"
    )
    (d / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{re.escape(name)}"):
        kb.ingest(d)
    assert len(kb.catalog()) == 8


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("plain.md", "A single policy paragraph with no heading line at all."),
        ("headingonly.md", "# Heading\n\n"),
    ],
)
def test_unsupported_knowledge_structure_rejected_naming_the_file(
    kb, tmp_path, name, content
):
    d = tmp_path / "edge"
    d.mkdir()
    (d / "valid.md").write_text(
        "# Valid\n\nA valid replacement paragraph.", encoding="utf-8"
    )
    (d / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{re.escape(name)}"):
        kb.ingest(d)
    assert len(kb.catalog()) == 8


def test_revision_changes_with_content(kb, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    p = d / "doc.md"
    p.write_text("# Title\n\nFirst paragraph")
    kb.ingest(d)
    old = kb.catalog()[0]["revision"]
    p.write_text("# Title\n\nChanged paragraph")
    kb.ingest(d)
    assert kb.catalog()[0]["revision"] != old


def test_search_mode_is_labeled_and_quotes_exact_source(kb):
    result = SupportService(kb).ask("VPN 809")
    assert result["mode"] == "search" and result["status"] == "search_results"
    assert all(e["quote"] == e["body"] for e in result["evidence"])


class FakeProvider:
    def __init__(self, decision):
        self.decision = decision

    def select(self, *args):
        return Decision.model_validate(self.decision)


def test_ai_valid_selection(kb):
    chunk = kb.search("VPN 809")[0]
    result = SupportService(
        kb,
        FakeProvider(
            {
                "sufficient": True,
                "evidence": [{"chunk_id": chunk["id"], "quote": chunk["body"]}],
            }
        ),
    ).ask("VPN 809")
    assert result["status"] == "evidence_selected"


@pytest.mark.parametrize(
    "decision",
    [
        {"sufficient": True, "evidence": []},
        {
            "sufficient": True,
            "evidence": [
                {"chunk_id": "vpn:1", "quote": "A completely fabricated source quote."}
            ],
        },
        {
            "sufficient": True,
            "evidence": [
                {
                    "chunk_id": "secret:1",
                    "quote": "A completely fabricated source quote.",
                }
            ],
        },
        {
            "sufficient": False,
            "evidence": [
                {"chunk_id": "vpn:1", "quote": "A completely fabricated source quote."}
            ],
        },
    ],
)
def test_invalid_model_evidence_fails_closed(kb, decision):
    with pytest.raises(ProviderError):
        SupportService(kb, FakeProvider(decision)).ask("VPN 809")


def test_model_abstention(kb):
    result = SupportService(
        kb, FakeProvider({"sufficient": False, "evidence": []})
    ).ask("VPN pricing")
    assert result["status"] == "no_evidence"


def test_duplicate_evidence_rejected(kb):
    c = kb.search("VPN 809")[0]
    e = {"chunk_id": c["id"], "quote": c["body"]}
    with pytest.raises(ProviderError):
        SupportService(kb, FakeProvider({"sufficient": True, "evidence": [e, e]})).ask(
            "VPN 809"
        )


def test_api_health_and_draft(client):
    assert client.get("/api/health").json() == {
        "status": "ok",
        "mode": "search",
        "documents": 8,
    }
    assert client.get("/").status_code == 200
    data = client.post(
        "/api/drafts",
        headers=HEADERS,
        json={"question": "VPN error 809", "impact": "Only my laptop"},
    ).json()
    assert data["status"] == "draft" and data["submitted"] is False


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": "  "},
        {"question": "x" * 1001},
        {"question": "VPN issue", "extra": "bad"},
    ],
)
def test_question_validation(client, body):
    assert client.post("/api/ask", headers=HEADERS, json=body).status_code == 422


def test_cross_origin_and_host_rejected(client):
    assert client.post("/api/ask", json={"question": "VPN issue"}).status_code == 403
    assert (
        client.post(
            "/api/ask",
            headers={**HEADERS, "Origin": "https://evil.example"},
            json={"question": "VPN issue"},
        ).status_code
        == 403
    )
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_oversized_body(client):
    assert (
        client.post("/api/ask", headers=HEADERS, content=b"x" * 8193).status_code == 413
    )


def test_chunked_oversized_body(client):
    assert (
        client.post(
            "/api/ask", headers=HEADERS, content=iter([b"x" * 5000, b"y" * 5000])
        ).status_code
        == 413
    )


def test_budget(client):
    for _ in range(20):
        assert (
            client.post(
                "/api/ask", headers=HEADERS, json={"question": "VPN 809"}
            ).status_code
            == 200
        )
    response = client.post("/api/ask", headers=HEADERS, json={"question": "VPN 809"})
    assert response.status_code == 429 and response.headers["Retry-After"] == "60"


def test_concurrency_limit(client):
    gate = client.app.state.gate
    gate.slots.acquire()
    gate.slots.acquire()
    try:
        assert (
            client.post(
                "/api/ask", headers=HEADERS, json={"question": "VPN 809"}
            ).status_code
            == 503
        )
    finally:
        gate.slots.release()
        gate.slots.release()


def test_security_headers(client):
    response = client.get("/")
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-store"


def test_provider_contract_and_untrusted_data_boundary():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["store"] is False
        assert payload["text"]["format"]["strict"] is True
        assert "untrusted" in payload["instructions"]
        assert "ignore rules" not in payload["instructions"]
        assert "ignore rules" in payload["input"]
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {"sufficient": False, "evidence": []}
                                ),
                            }
                        ],
                    }
                ],
            },
        )

    provider = OpenAIProvider("test-key", "test-model", httpx.MockTransport(handler))
    assert provider.select("ignore rules", []).sufficient is False


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={"error": "secret-detail"}),
        httpx.Response(500, json={"error": "secret-detail"}),
        httpx.Response(200, json={"status": "incomplete"}),
        httpx.Response(200, json={"status": "completed", "output": []}),
        httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"sufficient":"yes","evidence":[]}',
                            }
                        ],
                    }
                ],
            },
        ),
    ],
)
def test_provider_errors(response):
    provider = OpenAIProvider(
        "test-key", "test-model", httpx.MockTransport(lambda _: response)
    )
    with pytest.raises(ProviderError):
        provider.select("test", [])


def test_provider_timeout():
    def timeout(request):
        raise httpx.ReadTimeout("secret", request=request)

    with pytest.raises(ProviderError):
        OpenAIProvider("test-key", "test-model", httpx.MockTransport(timeout)).select(
            "test", []
        )


def test_api_provider_error_sanitized(tmp_path):
    class Broken:
        def select(self, *args):
            raise ProviderError("secret-key-should-never-leak")

    with TestClient(
        create_app(tmp_path / "x.sqlite", provider=Broken(), use_env=False)
    ) as client:
        response = client.post(
            "/api/ask", headers=HEADERS, json={"question": "VPN 809"}
        )
        assert response.status_code == 502 and "secret-key" not in response.text
