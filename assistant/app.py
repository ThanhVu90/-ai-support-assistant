import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .provider import OpenAIProvider, ProviderError
from .retrieval import KnowledgeBase
from .service import SupportService

ROOT = Path(__file__).resolve().parent.parent


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=1000)


class Draft(Question):
    impact: str = Field(min_length=3, max_length=500)


class Gate:
    """Single-process global budget; bounded memory and fail-fast concurrency."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = deque()
        self.slots = threading.BoundedSemaphore(2)

    def allow(self):
        now = time.monotonic()
        with self.lock:
            while self.calls and self.calls[0] <= now - 60:
                self.calls.popleft()
            if len(self.calls) >= 20:
                return False
            self.calls.append(now)
            return True


def create_app(database=None, knowledge_dir=None, provider=None, use_env=True):
    kb = KnowledgeBase(
        Path(database or os.getenv("SUPPORT_DB", ROOT / "var" / "knowledge.sqlite"))
    )
    if use_env and os.getenv("OPENAI_API_KEY"):
        if not os.getenv("OPENAI_MODEL"):
            raise RuntimeError("Set OPENAI_MODEL explicitly when enabling AI")
        provider = OpenAIProvider(
            os.environ["OPENAI_API_KEY"], os.environ["OPENAI_MODEL"]
        )
    service = SupportService(kb, provider)
    gate = Gate()

    @asynccontextmanager
    async def lifespan(app):
        kb.ingest(Path(knowledge_dir or ROOT / "knowledge"))
        yield

    app = FastAPI(title="AI Support Assistant", version="1.0.0", lifespan=lifespan)
    app.state.service = service
    app.state.gate = gate
    from .limits import BodyLimitMiddleware

    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def boundaries(request: Request, call_next):
        if request.method == "POST":
            if request.headers.get("x-support-client") != "web":
                return JSONResponse(
                    {"detail": "Missing client header"}, status_code=403
                )
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
                return JSONResponse(
                    {"detail": "Cross-origin requests rejected"}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "mode": "ai" if provider else "search",
            "documents": len(kb.catalog()),
        }

    @app.get("/api/documents")
    def documents():
        return kb.catalog()

    @app.post("/api/ask")
    def ask(body: Question):
        if not gate.allow():
            raise HTTPException(
                429,
                "Request budget reached. Retry in one minute.",
                headers={"Retry-After": "60"},
            )
        if not gate.slots.acquire(blocking=False):
            raise HTTPException(503, "Assistant busy. Try again shortly.")
        try:
            return service.ask(body.question)
        except ProviderError:
            raise HTTPException(
                502, "AI evidence could not be verified. Try again or contact IT."
            ) from None
        finally:
            gate.slots.release()

    @app.post("/api/drafts")
    def draft(body: Draft):
        return {
            "title": body.question[:100],
            "description": f"Issue: {body.question}\nImpact: {body.impact}\n\nSteps already tried: [complete before submitting]\nExpected result: [complete before submitting]",
            "status": "draft",
            "submitted": False,
        }

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/")
    def home():
        return FileResponse(ROOT / "static" / "index.html")

    return app
