"""The assistant as a web service: POST /ask answers a question, GET /health checks the index."""
import json
import os
import time
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.answer import answer_question
from src.retrieval import EMBEDDINGS, QDRANT_URL

app = FastAPI(title="Assistant congés")
QDRANT = QdrantClient(url=QDRANT_URL)
USERS = json.loads(Path("users.json").read_text(encoding="utf-8"))  # token -> groups

class Question(BaseModel):
    """The body of a POST /ask request."""

    question: str


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
    """Answer the question from the documents the caller may read, and say how long it took."""
    start = time.perf_counter()
    answer, docs = answer_question(body.question, "service", groups)
    return {"answer": answer.answer, "sources": answer.sources,
            "seconds": round(time.perf_counter() - start, 2)}
