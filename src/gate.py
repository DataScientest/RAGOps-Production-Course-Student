"""Quality gate: fail when a change lowers the search scores, lets a document leak or stops the refusals."""
import sys

from src.answer import answer_question
from src.measure import evidence_in_context, fiches_found, load_reference
from src.retrieval import search_service

MIN_FICHES = 18
MIN_EVIDENCE = 17
LEAK_PROBES = [
    ("L'entrepôt de Lyon va-t-il fermer ?", "NOTE-RH-02"),
    ("Combien de jours de congé pour mon mariage selon l'accord d'entreprise ?", "ACC-RH-01"),
]


def search_scores(questions: list[dict]) -> tuple[int, int]:
    """Number of answerable questions whose fiches are found, and whose evidence is in the context."""
    found, evidence = 0, 0
    for item in questions:
        docs = search_service(item["question"])
        found += fiches_found(item, docs)
        evidence += evidence_in_context(item, "\n\n".join(doc.page_content for doc in docs))
    return found, evidence


def leaks() -> list[str]:
    """The probes for which a public user gets a passage of a document reserved to a group."""
    leaked = []
    for question, doc_id in LEAK_PROBES:
        if doc_id in {doc.metadata["doc_id"] for doc in search_service(question, groups=["public"])}:
            leaked.append(doc_id)
    return leaked


def refusals(questions: list[dict]) -> int:
    """Number of out-of-scope questions that the assistant refuses to answer."""
    return sum(answer_question(item["question"], "service")[0].answer == "NO_ANSWER" for item in questions)


def main() -> int:
    """Run the three checks, print them, return 1 if one fails."""
    reference = load_reference()
    answerable = [item for item in reference if item["type"] != "unanswerable"]
    out_of_scope = [item for item in reference if item["type"] == "unanswerable"]
    found, evidence = search_scores(answerable)
    leaked = leaks()
    refused = refusals(out_of_scope)
    checks = [
        ("fiches trouvées", str(found) + " / " + str(len(answerable)), found >= MIN_FICHES),
        ("preuves dans le contexte", str(evidence) + " / " + str(len(answerable)), evidence >= MIN_EVIDENCE),
        ("documents réservés vus par un visiteur", str(len(leaked)), not leaked),
        ("questions hors sujet refusées", str(refused) + " / " + str(len(out_of_scope)), refused == len(out_of_scope)),
    ]
    for name, value, ok in checks:
        print(("OK    " if ok else "ÉCHEC ") + name.ljust(40) + value)
    return 0 if all(ok for _, _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
