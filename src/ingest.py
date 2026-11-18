"""Keep the index in step with the sources: the XML files in MinIO, the catalog in PostgreSQL, the chunks in Qdrant."""
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from minio import Minio
from minio.commonconfig import CopySource
from qdrant_client import QdrantClient, models

from src.chunking import fiche_sections, section_key, split_fiche
from src.parsing import fiche_to_markdown, list_fiches
from src.retrieval import EMBEDDINGS, QDRANT_URL

load_dotenv()
BUCKET = "sources"
ALIAS = "fiches"
STORAGE = Minio(os.environ["MINIO_URL"], access_key=os.environ["MINIO_USER"],
                secret_key=os.environ["MINIO_PASSWORD"], secure=False)
QDRANT = QdrantClient(url=QDRANT_URL)

CATALOG_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    collection TEXT,
    doc_id     TEXT,
    title      TEXT,
    sha256     TEXT,
    chunks     INT,
    indexed_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (collection, doc_id)
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS access TEXT;
CREATE TABLE IF NOT EXISTS index_versions (
    collection  TEXT PRIMARY KEY,
    embed_model TEXT,
    chunk_size  INT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
"""


def upload_sources(xml_dir: str = "data/raw/xml", dossier_id: str = "N510") -> int:
    """Copy the XML files of the dossier's fiches into the MinIO bucket."""
    if not STORAGE.bucket_exists(BUCKET):
        STORAGE.make_bucket(BUCKET)
    fiches = list_fiches(xml_dir, dossier_id)
    for doc_id in fiches:
        STORAGE.fput_object(BUCKET, doc_id + ".xml", xml_dir + "/" + doc_id + ".xml")
    return len(fiches)


def upload_document(path: str, access: str) -> None:
    """Copy one Markdown document into the bucket, with the group allowed to read it, waiting for review."""
    STORAGE.fput_object(BUCKET, Path(path).name, path, metadata={"access": access, "status": "pending"})


def approve_document(name: str) -> None:
    """Mark a reviewed document as approved: the next ingestion will index it."""
    metadata = STORAGE.stat_object(BUCKET, name).metadata
    STORAGE.copy_object(BUCKET, name, CopySource(BUCKET, name), metadata_directive="REPLACE",
                        metadata={"access": metadata.get("x-amz-meta-access", "rh"), "status": "approved"})


def read_sources() -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Markdown text and access group of every document to index, and the names of the documents waiting for review.

    The XML fiches come from Service-Public and are public. A Markdown document carries its group and
    its review status in its metadata; it is indexed only once approved.
    """
    roots, sources, pending = {}, {}, []
    for item in STORAGE.list_objects(BUCKET):
        data = STORAGE.get_object(BUCKET, item.object_name).read()
        if item.object_name.endswith(".xml"):
            roots[item.object_name.removesuffix(".xml")] = ET.fromstring(data)
            continue
        metadata = STORAGE.stat_object(BUCKET, item.object_name).metadata
        if metadata.get("x-amz-meta-status") != "approved":
            pending.append(item.object_name)
            continue
        access = metadata.get("x-amz-meta-access", "rh")
        sources[item.object_name.removesuffix(".md")] = (data.decode("utf-8"), access)
    for doc_id, root in roots.items():
        sources[doc_id] = (fiche_to_markdown(root, set(roots)), "public")
    return sources, pending


def fingerprint(text: str) -> str:
    """SHA-256 of the text: two identical texts have the same fingerprint."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunks_with_sections(doc_id: str, text: str, access: str, chunk_size: int) -> list[Document]:
    """The "titled" chunks of one document, each carrying the full text of its section and its access group."""
    sections = fiche_sections(doc_id, text)
    chunks = split_fiche(doc_id, text, chunk_size, with_title=True)
    for chunk in chunks:
        key = section_key(doc_id, chunk.metadata["section_path"])
        chunk.metadata["section_text"] = sections.get(key, chunk.page_content)
        chunk.metadata["access"] = access
    return chunks


def create_collection(collection: str) -> None:
    """Create an empty collection sized for the embedder, with an index on the fiche id."""
    size = len(EMBEDDINGS.embed_query("taille"))
    QDRANT.create_collection(collection, vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE))
    QDRANT.create_payload_index(collection, "metadata.doc_id", models.PayloadSchemaType.KEYWORD)
    QDRANT.create_payload_index(collection, "metadata.access", models.PayloadSchemaType.KEYWORD)


def delete_chunks(collection: str, doc_id: str) -> None:
    """Remove every chunk of one fiche from the collection."""
    match = models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id))
    QDRANT.delete(collection, points_selector=models.FilterSelector(filter=models.Filter(must=[match])))


def ingest(collection: str, chunk_size: int = 500) -> Counter:
    """Index the new and changed fiches, remove the withdrawn ones, skip the unchanged ones."""
    texts, pending = read_sources()
    if not QDRANT.collection_exists(collection):
        create_collection(collection)
    store = QdrantVectorStore(QDRANT, collection, EMBEDDINGS)
    counts = Counter()
    with psycopg.connect(os.environ["POSTGRES_URL"]) as db:
        db.execute(CATALOG_SQL)
        db.execute("INSERT INTO index_versions (collection, embed_model, chunk_size) VALUES (%s, %s, %s) "
                   "ON CONFLICT (collection) DO NOTHING", (collection, EMBEDDINGS.model, chunk_size))
        known = dict(db.execute("SELECT doc_id, sha256 FROM documents WHERE collection = %s", (collection,)).fetchall())
        for doc_id, (text, access) in sorted(texts.items()):
            digest = fingerprint(access + "\n" + text)
            if known.get(doc_id) == digest:
                counts["inchangées"] += 1
                continue
            delete_chunks(collection, doc_id)
            chunks = chunks_with_sections(doc_id, text, access, chunk_size)
            store.add_documents(chunks)
            db.execute("INSERT INTO documents (collection, doc_id, title, access, sha256, chunks) "
                       "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (collection, doc_id) DO UPDATE SET "
                       "title = EXCLUDED.title, access = EXCLUDED.access, sha256 = EXCLUDED.sha256, "
                       "chunks = EXCLUDED.chunks, indexed_at = now()",
                       (collection, doc_id, text.splitlines()[0].removeprefix("# "), access, digest, len(chunks)))
            db.commit()
            counts["modifiées" if doc_id in known else "ajoutées"] += 1
        for doc_id in sorted(set(known) - set(texts)):
            delete_chunks(collection, doc_id)
            db.execute("DELETE FROM documents WHERE collection = %s AND doc_id = %s", (collection, doc_id))
            db.commit()
            counts["retirées"] += 1
    counts["en attente"] = len(pending)
    return counts


def use_version(collection: str) -> None:
    """Point the alias "fiches" to the collection, in one step: the service switches without a pause."""
    actions = [models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=ALIAS))]
    for alias in QDRANT.get_aliases().aliases:
        if alias.alias_name == ALIAS:
            actions.insert(0, models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)))
    QDRANT.update_collection_aliases(change_aliases_operations=actions)


if __name__ == "__main__":
    if sys.argv[1] == "upload":
        xml_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw/xml"
        print(upload_sources(xml_dir), "fiches copiées dans le bucket", BUCKET)
    elif sys.argv[1] == "add":
        upload_document(sys.argv[2], sys.argv[3])
        print(sys.argv[2], "copié dans le bucket", BUCKET, "pour le groupe", sys.argv[3])
    elif sys.argv[1] == "approve":
        approve_document(sys.argv[2])
        print(sys.argv[2], "validé")
    elif sys.argv[1] == "use":
        use_version(sys.argv[2])
        print("l'alias", ALIAS, "pointe vers", sys.argv[2])
    else:
        chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        for key, count in sorted(ingest(sys.argv[1], chunk_size).items()):
            print(key.ljust(11), count)
