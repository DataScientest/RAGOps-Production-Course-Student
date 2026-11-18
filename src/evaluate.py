"""Grade the answers of the assistant on the reference set, and compare two runs."""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.answer import answer_question
from src.measure import load_reference
from src.retrieval import LLM

RUNS_DIR = Path("data/eval/runs")

GRADER_PROMPT = """Vous corrigez les réponses d'un assistant sur les congés des salariés. On vous donne la question, la réponse attendue (juste) et la réponse de l'assistant.

Étape 1 : listez les faits que la question demande. Ignorez les précisions de la réponse attendue que la question ne demande pas.
Étape 2 : pour chaque fait, vérifiez s'il est dans la réponse de l'assistant, sans contradiction.
Étape 3 : donnez le verdict."""


class Grade(BaseModel):
    """The verdict of the grader on one answer."""

    asked_facts: list[str] = Field(description="Les faits que la question demande, tirés de la réponse attendue")
    missing_or_wrong: list[str] = Field(description="Les faits demandés qui manquent ou sont contredits ; liste vide si aucun")
    verdict: Literal["correct", "partiel", "faux"] = Field(
        description="correct si aucun fait ne manque ; partiel si un fait manque ; faux si un fait est contredit")


def grade(item: dict, answer: str) -> str:
    """Verdict of one answer: 'refus', or the grader's 'correct', 'partiel', 'faux'."""
    if answer.strip() == "NO_ANSWER":
        return "refus"
    if item["type"] == "unanswerable":
        return "faux"
    text = ("Question : " + item["question"] + "\n\nRéponse attendue : " + item["reference_answer"]
            + "\n\nRéponse de l'assistant : " + answer)
    return LLM.with_structured_output(Grade).invoke([SystemMessage(GRADER_PROMPT), HumanMessage(text)]).verdict


def run(method: str, name: str) -> Counter:
    """Answer and grade every question, save the run in data/eval/runs/<name>.jsonl, count the verdicts."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(RUNS_DIR / (name + ".jsonl"), "w", encoding="utf-8") as file:
        for item in load_reference():
            answer, docs = answer_question(item["question"], method)
            verdict = grade(item, answer.answer)
            counts[item["type"] + " " + verdict] += 1
            result = {"id": item["id"], "type": item["type"], "answer": answer.answer, "verdict": verdict}
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return counts


def load_run(name: str) -> list[dict]:
    """Read a run saved by run()."""
    with open(RUNS_DIR / (name + ".jsonl"), encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def compare(name_a: str, name_b: str) -> None:
    """Print the questions whose verdict differs between two runs."""
    changed = 0
    for a, b in zip(load_run(name_a), load_run(name_b)):
        if a["verdict"] != b["verdict"]:
            changed += 1
            print(a["id"], a["verdict"], "->", b["verdict"])
    print(changed, "verdicts différents sur", len(load_run(name_a)))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        for key, count in sorted(run(sys.argv[1], sys.argv[2]).items()):
            print(key.ljust(22), count)
