"""Send the questions of the reference set to the service, as a user would, to fill the dashboards."""
import json
import sys

import httpx

from src.measure import load_reference


def send_questions(token: str, url: str = "http://localhost:8000") -> None:
    """Ask every question of the reference set and print the outcome of each one."""
    headers = {"Authorization": "Bearer " + token}
    for item in load_reference():
        response = httpx.post(url + "/ask", json={"question": item["question"]}, headers=headers, timeout=60)
        body = response.json()
        print(item["id"], response.status_code, body.get("seconds"), "refus" if body.get("answer") == "NO_ANSWER" else "réponse")


if __name__ == "__main__":
    users = json.load(open("users.json", encoding="utf-8"))
    group = sys.argv[1] if len(sys.argv) > 1 else "sal"
    send_questions(next(token for token in users if token.startswith(group + "-")))
