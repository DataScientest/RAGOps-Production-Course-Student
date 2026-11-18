"""Answer a question from the passages found, citing the fiches used, or refuse."""
import sys

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.measure import METHODS
from src.retrieval import LLM

SYSTEM_PROMPT = """Vous répondez aux questions de salariés du secteur privé sur leurs congés.

Règles :
- Répondez uniquement à partir des extraits fournis, en français, de façon concise.
- Citez dans "sources" l'identifiant de chaque fiche utilisée, par exemple F2381.
- Si les extraits ne permettent pas de répondre avec certitude, mettez exactement NO_ANSWER dans "answer" et une liste vide dans "sources"."""


class Answer(BaseModel):
    """The answer of the assistant."""

    answer: str = Field(description="Réponse à la question, ou NO_ANSWER si les extraits ne permettent pas de répondre")
    sources: list[str] = Field(description="Identifiants des fiches utilisées, par exemple F2381 ; liste vide si NO_ANSWER")


def build_context(docs: list[Document]) -> str:
    """Put the passages in one text: the fiche id in brackets, then the passage."""
    return "\n\n".join("[" + doc.metadata["doc_id"] + "]\n" + doc.page_content for doc in docs)


def answer_question(question: str, method: str = "titled+section") -> tuple[Answer, list[Document]]:
    """Search with one method, then ask the model to answer from the passages found."""
    docs = METHODS[method](question)
    messages = [SystemMessage(SYSTEM_PROMPT),
                HumanMessage("Extraits :\n\n" + build_context(docs) + "\n\nQuestion : " + question)]
    return LLM.with_structured_output(Answer).invoke(messages), docs


if __name__ == "__main__":
    answer, docs = answer_question(" ".join(sys.argv[1:]))
    print(answer.answer)
    print("Sources :", ", ".join(answer.sources) or "aucune")
