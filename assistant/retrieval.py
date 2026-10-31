"""Atomic, deterministic Markdown ingestion and SQLite FTS5 retrieval."""

import hashlib
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from pathlib import Path

STOP = frozenset(
    "how do i the a an is my to for and of can what why please with it in on does not are".split()
)


def terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return list(
        dict.fromkeys(
            t
            for t in re.findall(r"[^\W_]+", normalized)
            if t not in STOP and len(t) > 1
        )
    )[:32]


class KnowledgeBase:
    def __init__(self, database: Path):
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                'CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(id UNINDEXED, source UNINDEXED, title, body, revision UNINDEXED, tokenize="unicode61 remove_diacritics 2")'
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, directory: Path) -> int:
        rows = []
        for path in sorted(directory.glob("*.md")):
            if path.is_symlink():
                raise ValueError("Symlinks are not accepted")
            text = path.read_text(encoding="utf-8-sig")
            if len(text) > 100_000:
                raise ValueError("Document exceeds 100,000 characters")
            if not text.strip():
                raise ValueError(f"Knowledge file {path.name} is empty")
            title = text.splitlines()[0].lstrip("# ").strip()
            revision = hashlib.sha256(text.encode()).hexdigest()[:16]
            paragraphs = re.split(r"\n\s*\n", text)[1:]
            file_rows = []
            for i, paragraph in enumerate(paragraphs, 1):
                body = paragraph.strip()
                if not body:
                    continue
                if len(body) > 2000:
                    raise ValueError(
                        "Paragraph exceeds 2,000 characters; split it before indexing"
                    )
                file_rows.append((f"{path.stem}:{i}", path.name, title, body, revision))
            if not file_rows:
                raise ValueError(
                    f"Knowledge file {path.name}: needs a heading line and one paragraph"
                )
            rows.extend(file_rows)
        if not rows:
            raise ValueError("Knowledge directory contains no paragraphs")
        with self.connect() as db:
            db.execute("DELETE FROM chunks")
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)
        return len(rows)

    def search(self, question: str, limit: int = 4) -> list[dict]:
        tokens = terms(question)
        if not tokens:
            return []
        query = " OR ".join('"' + token + '"' for token in tokens)
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            results = db.execute(
                "SELECT id, source, title, body, revision, bm25(chunks, 0, 0, 2, 1, 0) AS rank FROM chunks WHERE chunks MATCH ? ORDER BY rank, id LIMIT ?",
                (query, limit),
            ).fetchall()
        return [dict(row) for row in results]

    def catalog(self) -> list[dict]:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute(
                    "SELECT source, title, revision, count(*) AS chunks FROM chunks GROUP BY source, title, revision ORDER BY source"
                )
            ]
