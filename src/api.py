"""The assistant as a web service: POST /ask answers a question, GET /health checks the index."""
import os
import time

import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.answer import answer_question
from src.retrieval import EMBEDDINGS, QDRANT_URL

app = FastAPI(title="Assistant congés")
QDRANT = QdrantClient(url=QDRANT_URL)


class Question(BaseModel):
    """The body of a POST /ask request."""

    question: str


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
def ask(body: Question) -> dict:
    """Answer the question with the index of the service, and say how long it took."""
    start = time.perf_counter()
    answer, docs = answer_question(body.question, "service")
    return {"answer": answer.answer, "sources": answer.sources,
            "seconds": round(time.perf_counter() - start, 2)}
