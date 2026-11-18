"""The assistant as a web service: /ask answers, /feedback records an opinion, /metrics is read by Prometheus."""
import json
import os
import time
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.callbacks import get_usage_metadata_callback
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.answer import answer_question
from src.retrieval import EMBEDDINGS, QDRANT_URL

app = FastAPI(title="Assistant congés")
app.mount("/metrics", make_asgi_app())
QDRANT = QdrantClient(url=QDRANT_URL)
USERS = json.loads(Path("users.json").read_text(encoding="utf-8"))  # token -> groups

REQUESTS = Counter("rag_requests_total", "Questions received, by outcome", ["outcome"])
SECONDS = Histogram("rag_request_seconds", "Time to answer a question", buckets=[0.25, 0.5, 1, 2, 4, 8])
TOKENS = Counter("rag_tokens_total", "Tokens sent to and received from the chat model", ["kind"])
FEEDBACK = Counter("rag_feedback_total", "Opinions of the users on the answers", ["useful"])

FEEDBACK_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id         SERIAL PRIMARY KEY,
    question   TEXT,
    answer     TEXT,
    sources    TEXT,
    useful     BOOLEAN,
    comment    TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


class Question(BaseModel):
    """The body of a POST /ask request."""

    question: str


class Feedback(BaseModel):
    """The body of a POST /feedback request: the answer the user saw, and whether it helped."""

    question: str
    answer: str
    sources: list[str]
    useful: bool
    comment: str = ""


def user_groups(authorization: str = Header(default="")) -> list[str]:
    """The groups of the caller, found from the token of the Authorization header; 401 if the token is unknown."""
    token = authorization.removeprefix("Bearer ")
    if token not in USERS:
        raise HTTPException(status_code=401, detail="jeton inconnu")
    return USERS[token]


@app.get("/health")
def health() -> dict:
    """Check, without calling any model, that the index of the alias was built with the embedder of the service."""
    aliases = {alias.alias_name: alias.collection_name for alias in QDRANT.get_aliases().aliases}
    collection = aliases.get("fiches")
    with psycopg.connect(os.environ["POSTGRES_URL"]) as db:
        row = db.execute("SELECT embed_model FROM index_versions WHERE collection = %s", (collection,)).fetchone()
    if row is None or row[0] != EMBEDDINGS.model:
        raise HTTPException(status_code=503, detail="index " + str(collection) + " incompatible avec " + EMBEDDINGS.model)
    return {"status": "ok", "index": collection, "embed_model": row[0]}


@app.post("/ask")
def ask(body: Question, groups: list[str] = Depends(user_groups)) -> dict:
    """Answer the question from the documents the caller may read, and record the metrics of the request."""
    start = time.perf_counter()
    try:
        with get_usage_metadata_callback() as usage:
            answer, docs = answer_question(body.question, "service", groups)
    except Exception:
        REQUESTS.labels(outcome="error").inc()
        raise
    seconds = time.perf_counter() - start
    SECONDS.observe(seconds)
    REQUESTS.labels(outcome="refused" if answer.answer == "NO_ANSWER" else "answered").inc()
    for model_usage in usage.usage_metadata.values():
        TOKENS.labels(kind="input").inc(model_usage["input_tokens"])
        TOKENS.labels(kind="output").inc(model_usage["output_tokens"])
    return {"answer": answer.answer, "sources": answer.sources, "seconds": round(seconds, 2)}


@app.post("/feedback")
def feedback(body: Feedback, groups: list[str] = Depends(user_groups)) -> dict:
    """Store the opinion of the user in PostgreSQL, to review the bad answers later."""
    with psycopg.connect(os.environ["POSTGRES_URL"]) as db:
        db.execute(FEEDBACK_SQL)
        db.execute("INSERT INTO feedback (question, answer, sources, useful, comment) VALUES (%s, %s, %s, %s, %s)",
                   (body.question, body.answer, ",".join(body.sources), body.useful, body.comment))
    FEEDBACK.labels(useful="yes" if body.useful else "no").inc()
    return {"status": "enregistré"}
