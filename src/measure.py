"""Measure a search method on the reference set: fiches found, evidence in the context, size and time."""
import json
import statistics
import sys
import time

from src.chunking import load_sections
from src.corpus import count_tokens
from src.retrieval import expand_to_sections, search, search_hybrid, search_multi, search_rerank

REFERENCE_PATH = "data/eval/ref_v2.jsonl"
SECTIONS = load_sections()

METHODS = {
    "fixed": lambda q: search(q, strategy="fixed"),
    "headers": lambda q: search(q, strategy="headers"),
    "titled": lambda q: search(q, strategy="titled"),
    "titled+section": lambda q: expand_to_sections(search(q), SECTIONS),
    "hybrid+section": lambda q: expand_to_sections(search_hybrid(q), SECTIONS),
    "rerank+section": lambda q: expand_to_sections(search_rerank(q), SECTIONS),
    "multi+section": lambda q: expand_to_sections(search_multi(q), SECTIONS),
}


def load_reference(path: str = REFERENCE_PATH) -> list[dict]:
    """Read the reference set: one question per line."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def normalize(text: str) -> str:
    """Lowercase, straighten apostrophes and collapse spaces."""
    return " ".join(text.lower().replace("’", "'").split())


def fiches_found(item: dict, docs: list) -> bool:
    """True if all the expected fiches are among the passages found."""
    return set(item["gold_doc_ids"]) <= {doc.metadata["doc_id"] for doc in docs}


def evidence_in_context(item: dict, context: str) -> bool:
    """True if every phrase of the evidence (parts joined by [...] or |) is in the context."""
    for part in item["evidence"].replace("|", "[...]").split("[...]"):
        phrase = normalize(part)
        if len(phrase) > 15 and phrase not in normalize(context):
            return False
    return True


def measure(method: str) -> None:
    """Run one method on the answerable questions and print the scores and the failures."""
    questions = [item for item in load_reference() if item["type"] != "unanswerable"]
    found, evidence, tokens, seconds, failures = 0, 0, [], [], []
    for item in questions:
        start = time.perf_counter()
        docs = METHODS[method](item["question"])
        seconds.append(time.perf_counter() - start)
        context = "\n\n".join(doc.page_content for doc in docs)
        tokens.append(count_tokens(context))
        found += fiches_found(item, docs)
        evidence += evidence_in_context(item, context)
        if not evidence_in_context(item, context):
            cause = "preuve coupée" if fiches_found(item, docs) else "fiche manquée"
            failures.append(item["id"] + " " + cause + " | attendu " + ",".join(item["gold_doc_ids"])
                            + " | trouvé " + ",".join(sorted({doc.metadata["doc_id"] for doc in docs})))
    print("méthode :", method)
    print("fiches trouvées :", found, "/", len(questions))
    print("preuve dans le contexte :", evidence, "/", len(questions))
    print("contexte moyen :", int(statistics.mean(tokens)), "tokens | durée moyenne :", round(statistics.mean(seconds), 1), "s")
    print("échecs :")
    for line in failures:
        print("  " + line)


if __name__ == "__main__":
    measure(sys.argv[1])
