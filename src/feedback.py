"""Turn the questions whose answer users found wrong into candidate questions for the reference set.

Only the question and the comment are exported: the answer may quote a document reserved to a group.
"""
import json
import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()
CANDIDATES_PATH = "data/eval/candidates.jsonl"


def export_candidates(path: str = CANDIDATES_PATH) -> int:
    """Write one candidate per question judged not useful; the expected fields stay to fill by hand."""
    with psycopg.connect(os.environ["POSTGRES_URL"]) as db:
        rows = db.execute("SELECT id, question, comment FROM feedback WHERE NOT useful ORDER BY id").fetchall()
    with open(path, "w", encoding="utf-8") as file:
        for feedback_id, question, comment in rows:
            candidate = {"id": "fb" + str(feedback_id), "question": question, "type": "à compléter",
                         "gold_doc_ids": [], "evidence": "", "reference_answer": "", "user_comment": comment}
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    print(export_candidates(), "questions candidates écrites dans", CANDIDATES_PATH)
