"""Index the chunks in a local Qdrant base and search them, with the improvements of chapters 3 and 4."""
import os
import re
import sys
from functools import cache

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.chunking import STRATEGIES, section_key

load_dotenv()
QDRANT_PATH = "data/qdrant"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small", base_url=os.environ["LLM_BASE_URL"],
                              api_key=os.environ["LLM_API_KEY"], check_embedding_ctx_length=False)
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url=os.environ["LLM_BASE_URL"],
                 api_key=os.environ["LLM_API_KEY"])


def build_store(strategy: str) -> int:
    """Embed the chunks of one strategy and write them to the Qdrant collection of the same name."""
    chunks = STRATEGIES[strategy]()
    store = QdrantVectorStore.from_documents(chunks, EMBEDDINGS, path=QDRANT_PATH,
                                             collection_name=strategy, force_recreate=True)
    store.client.close()  # only one client at a time can open data/qdrant/
    return len(chunks)


def search(question: str, k: int = 5, strategy: str = "titled") -> list[Document]:
    """Return the k chunks closest to the question."""
    store = QdrantVectorStore.from_existing_collection(strategy, EMBEDDINGS, path=QDRANT_PATH)
    docs = store.similarity_search(question, k=k)
    store.client.close()
    return docs


def expand_to_sections(docs: list[Document], sections: dict[str, str], max_chars: int = 6000) -> list[Document]:
    """Replace each chunk by the text of its whole section, without duplicates.

    A section longer than max_chars characters is kept as the chunk found.
    """
    expanded = []
    seen = set()
    for doc in docs:
        key = section_key(doc.metadata["doc_id"], doc.metadata["section_path"])
        text = sections.get(key)
        if text is None or len(text) > max_chars:
            text = doc.page_content
            key = text
        if key not in seen:
            seen.add(key)
            expanded.append(Document(page_content=text, metadata=doc.metadata))
    return expanded


def tokenize(text: str) -> list[str]:
    """Lowercase words; hyphens are kept, so that "L3142-4" stays one word."""
    return re.findall(r"[\w-]+", text.lower())


@cache
def keyword_index() -> tuple[list[Document], BM25Okapi]:
    """The "titled" chunks and their BM25 index, built once."""
    chunks = STRATEGIES["titled"]()
    return chunks, BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])


def search_hybrid(question: str, k: int = 5) -> list[Document]:
    """Merge the vector ranking and the BM25 ranking with reciprocal rank fusion (RRF)."""
    chunks, bm25 = keyword_index()
    vector_docs = search(question, k=50)
    scores = bm25.get_scores(tokenize(question))
    keyword_docs = [chunks[i] for i in np.argsort(-scores)[:50]]
    fused = {}
    docs_by_text = {}
    for ranking in (vector_docs, keyword_docs):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc.page_content] = fused.get(doc.page_content, 0) + 1 / (60 + rank)
            docs_by_text[doc.page_content] = doc
    best = sorted(fused, key=lambda text: -fused[text])[:k]
    return [docs_by_text[text] for text in best]


class Ranking(BaseModel):
    """The passages kept by the reranker, most useful first."""

    numbers: list[int] = Field(description="Numéros des passages utiles pour répondre, du plus utile au moins utile")


def search_rerank(question: str, k: int = 5, candidates: int = 20) -> list[Document]:
    """Take the 20 closest chunks, then ask the model to pick and order the k most useful ones."""
    docs = search(question, k=candidates)
    passages = "\n\n".join("[" + str(i) + "]\n" + doc.page_content for i, doc in enumerate(docs))
    prompt = ("Question : " + question + "\n\nPassages :\n\n" + passages
              + "\n\nDonnez les numéros des " + str(k) + " passages les plus utiles pour répondre, du plus utile au moins utile.")
    numbers = LLM.with_structured_output(Ranking).invoke(prompt).numbers
    kept = [docs[n] for n in dict.fromkeys(numbers) if 0 <= n < len(docs)]
    return kept[:k]


class SubQuestions(BaseModel):
    """The sub-questions of a question."""

    sub_questions: list[str] = Field(description="1 à 3 questions autonomes, chacune cherchant UNE information")


def split_question(question: str) -> list[str]:
    """Ask the model to cut a question into 1 to 3 standalone sub-questions."""
    prompt = ("Un salarié pose une question sur ses congés. Découpez-la en 1 à 3 sous-questions autonomes, "
              "chacune cherchant UNE seule information. Si la question est simple, renvoyez-la telle quelle.\n\n"
              "Question : " + question)
    return LLM.with_structured_output(SubQuestions).invoke(prompt).sub_questions[:3]


def search_multi(question: str, k: int = 5) -> list[Document]:
    """Search the question and its sub-questions, then take one result of each list in turn."""
    rankings = [search(query, k=k) for query in [question] + split_question(question)]
    merged = []
    seen = set()
    for rank in range(k):
        for docs in rankings:
            if docs[rank].page_content not in seen:
                seen.add(docs[rank].page_content)
                merged.append(docs[rank])
    return merged[:k]


if __name__ == "__main__":
    strategy = sys.argv[1]
    if len(sys.argv) > 2:
        for doc in search(sys.argv[2], strategy=strategy):
            print(doc.metadata["doc_id"], "|", doc.metadata["section_path"] or doc.page_content[:60].replace("\n", " "))
    else:
        print(build_store(strategy), "chunks indexés dans la collection", strategy)
