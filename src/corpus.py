"""Write one Markdown file per fiche in data/corpus/, then describe the corpus."""
import statistics
from pathlib import Path

import tiktoken

from src.parsing import fiche_to_markdown, list_fiches

CORPUS_DIR = Path("data/corpus")
ENCODING = tiktoken.get_encoding("cl100k_base")


def build_corpus(dossier_id: str = "N510") -> int:
    """Convert the fiches of the dossier to Markdown and write them to data/corpus/."""
    fiches = list_fiches(dossier_id=dossier_id)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for fiche_id, root in fiches.items():
        text = fiche_to_markdown(root, set(fiches))
        (CORPUS_DIR / (fiche_id + ".md")).write_text(text, encoding="utf-8")
    return len(fiches)


def count_tokens(text: str) -> int:
    """Number of tokens of a text, with the tokenizer of the OpenAI models."""
    return len(ENCODING.encode(text))


def describe_corpus() -> None:
    """Print the size of the fiches in tokens and their number of sections."""
    tokens = {}
    sections = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        tokens[path.stem] = count_tokens(text)
        sections[path.stem] = text.count("\n## ")
    sizes = list(tokens.values())
    longest = max(tokens, key=tokens.get)
    print(len(tokens), "fiches")
    print("tokens par fiche : min", min(sizes), "| médiane", int(statistics.median(sizes)), "| max", max(sizes))
    print("fiche la plus longue :", longest, "(" + str(tokens[longest]) + " tokens)")
    print("fiches de plus de 2000 tokens :", sum(size > 2000 for size in sizes))
    print("sections (##) par fiche : médiane", int(statistics.median(sections.values())), "| max", max(sections.values()))


if __name__ == "__main__":
    print(build_corpus(), "fiches écrites dans data/corpus/")
    describe_corpus()
