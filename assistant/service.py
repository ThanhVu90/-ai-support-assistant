import time
import uuid

from .provider import Decision, ProviderError


def validate_evidence(decision: Decision, chunks: list[dict]) -> list[dict]:
    """Provenance gate shared by SupportService and the live evaluator.

    Accepts only decisions that abstain consistently (no evidence) or cite
    verbatim quotes from the chunks that were actually sent to the model, each
    cited once. Everything else fails closed.
    """
    by_id = {chunk["id"]: chunk for chunk in chunks}
    evidence = []
    seen = set()
    if not decision.sufficient:
        if decision.evidence:
            raise ProviderError("Inconsistent abstention")
        return evidence
    if not decision.evidence:
        raise ProviderError("Missing supporting evidence")
    for selection in decision.evidence:
        chunk = by_id.get(selection.chunk_id)
        if (
            not chunk
            or selection.quote not in chunk["body"]
            or selection.chunk_id in seen
        ):
            raise ProviderError("Evidence failed provenance validation")
        seen.add(selection.chunk_id)
        evidence.append({**chunk, "quote": selection.quote})
    return evidence


class SupportService:
    def __init__(self, knowledge, provider=None):
        self.knowledge, self.provider = knowledge, provider

    def ask(self, question: str) -> dict:
        started = time.perf_counter()
        chunks = self.knowledge.search(question)
        evidence = []
        status = "no_evidence"
        if chunks and self.provider is None:
            # Deterministic search preview: does not claim semantic answerability.
            evidence = [{**chunk, "quote": chunk["body"]} for chunk in chunks[:3]]
            status = "search_results"
        elif chunks:
            decision = self.provider.select(question, chunks)
            evidence = validate_evidence(decision, chunks)
            if decision.sufficient:
                status = "evidence_selected"
        return {
            "request_id": str(uuid.uuid4()),
            "mode": "ai" if self.provider else "search",
            "status": status,
            "question": question,
            "evidence": evidence,
            "message": "Review the cited excerpts before taking action."
            if evidence
            else "No supported answer found. Escalate to the service desk.",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
