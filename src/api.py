"""The assistant as a web service: POST /ask answers a question, GET /health says the service is up."""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.answer import answer_question

app = FastAPI(title="Assistant congés")


class Question(BaseModel):
    """The body of a POST /ask request."""

    question: str


@app.get("/health")
def health() -> dict:
    """Answer quickly, without calling any model: the service is up."""
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question) -> dict:
    """Answer the question with the method kept in module 1, and say how long it took."""
    start = time.perf_counter()
    answer, docs = answer_question(body.question)
    return {"answer": answer.answer, "sources": answer.sources,
            "seconds": round(time.perf_counter() - start, 2)}
