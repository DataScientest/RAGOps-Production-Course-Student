"""Three ways to cut the fiches into chunks: by size, on the titles, on the titles with the title in front."""
import statistics
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.corpus import count_tokens

HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


def size_splitter(chunk_size: int = 500) -> RecursiveCharacterTextSplitter:
    """A splitter that cuts a text into pieces of at most chunk_size tokens."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=0)


def chunk_fixed(corpus_dir: str = "data/corpus", chunk_size: int = 500) -> list[Document]:
    """Cut each fiche into pieces of chunk_size tokens, without looking at its titles."""
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        for text in size_splitter(chunk_size).split_text(path.read_text(encoding="utf-8")):
            chunks.append(Document(page_content=text, metadata={"doc_id": path.stem, "section_path": ""}))
    return chunks


def make_section_path(metadata: dict) -> str:
    """Title path of a chunk: "Fiche > Section > Subsection"."""
    levels = []
    for _, key in HEADERS:
        if key in metadata:
            levels.append(metadata[key])
    return " > ".join(levels)


def chunk_on_headers(corpus_dir: str = "data/corpus", chunk_size: int = 500,
                     with_title: bool = False) -> list[Document]:
    """Cut each fiche on its titles, then cut the long sections into pieces of chunk_size tokens.

    with_title=True writes the title path in front of the chunk: "[Fiche > Section]\\ntext".
    """
    by_title = MarkdownHeaderTextSplitter(HEADERS)
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        pieces = by_title.split_text(path.read_text(encoding="utf-8"))
        for chunk in size_splitter(chunk_size).split_documents(pieces):
            section_path = make_section_path(chunk.metadata)
            if with_title:
                chunk.page_content = "[" + section_path + "]\n" + chunk.page_content
            chunk.metadata = {"doc_id": path.stem, "section_path": section_path}
            chunks.append(chunk)
    return chunks


STRATEGIES = {
    "fixed": lambda: chunk_fixed(),
    "headers": lambda: chunk_on_headers(),
    "titled": lambda: chunk_on_headers(with_title=True),
}


def section_key(doc_id: str, section_path: str) -> str:
    """Identifier of the section of a chunk: the fiche and the first two levels of its title path."""
    return doc_id + " | " + " > ".join(section_path.split(" > ")[:2])


def load_sections(corpus_dir: str = "data/corpus") -> dict[str, str]:
    """Full text of each "##" section of each fiche, indexed by section_key()."""
    by_title = MarkdownHeaderTextSplitter(HEADERS)
    sections = {}
    for path in sorted(Path(corpus_dir).glob("*.md")):
        for piece in by_title.split_text(path.read_text(encoding="utf-8")):
            section_path = make_section_path(piece.metadata)
            key = section_key(path.stem, section_path)
            block = "[" + section_path + "]\n" + piece.page_content.strip()
            if key in sections:
                sections[key] = sections[key] + "\n\n" + block
            else:
                sections[key] = block
    return sections


if __name__ == "__main__":
    for name, make_chunks in STRATEGIES.items():
        sizes = [count_tokens(chunk.page_content) for chunk in make_chunks()]
        print(name.ljust(8), len(sizes), "chunks | médiane", int(statistics.median(sizes)), "tokens | max", max(sizes))
