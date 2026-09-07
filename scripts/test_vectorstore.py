"""Tests for the vector store layer, against a throwaway embedded Qdrant.

    python scripts/test_vectorstore.py

Uses a temporary directory, so it never touches the project's real collection.
Focuses on the properties that fail silently rather than loudly:

  * deterministic point ids, so re-indexing overwrites instead of duplicating
  * a dimension change forcing a recreate, since mismatched widths cannot coexist
  * document deletion removing surplus points, so a shorter re-chunk does not
    leave orphans that keep getting retrieved
  * offsets surviving the round trip, since citations resolve through them
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.config as config

TMP = Path(tempfile.mkdtemp(prefix="qdrant_test_"))
config.QDRANT_PATH = TMP
config.QDRANT_URL = ""
config.QDRANT_COLLECTION = "test_chunks"

import app.vectorstore as vs  # noqa: E402

vs.QDRANT_PATH = TMP
vs.QDRANT_URL = ""
vs.QDRANT_COLLECTION = "test_chunks"

failures = []


def check(label, cond, detail=""):
    print(f"{'  ok  ' if cond else ' FAIL '} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


def vector(seed, dim):
    """A deterministic unit vector, distinct per seed."""
    raw = [((seed * 7 + i * 13) % 100) / 100 - 0.5 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


def main():
    ok, why = vs.available()
    if not ok:
        raise SystemExit(f"qdrant-client unavailable: {why}")

    # deterministic ids, no client needed
    a, b = vs.point_id("doc:1:0"), vs.point_id("doc:1:0")
    check("point ids are deterministic", a == b, a)
    check("different chunks get different point ids",
          vs.point_id("doc:1:0") != vs.point_id("doc:1:1"))

    dim = 8
    client = vs.connect()
    try:
        created = vs.ensure_collection(client, dim, name="test_chunks")
        check("collection is created on first use", created)
        check("second call is a no-op",
              vs.ensure_collection(client, dim, name="test_chunks") is False)

        exists, got_dim = vs.collection_info(client, "test_chunks")
        check("collection reports its dimension", exists and got_dim == dim,
              str(got_dim))

        records = [
            (f"docA:1:{i}", vector(i, dim), {
                "chunk_id": f"docA:1:{i}", "document_id": "docA",
                "page_no": 1, "ordinal": i,
                "char_start": i * 100, "char_end": i * 100 + 90,
                "text": f"অনুচ্ছেদ {i}", "model": "test:8",
            }) for i in range(5)
        ]
        vs.upsert_chunks(client, records, name="test_chunks")
        check("points are upserted", vs.count(client, "test_chunks") == 5,
              str(vs.count(client, "test_chunks")))

        # re-upserting the same ids must overwrite, not duplicate
        vs.upsert_chunks(client, records, name="test_chunks")
        check("re-upserting the same chunks does not duplicate",
              vs.count(client, "test_chunks") == 5,
              str(vs.count(client, "test_chunks")))

        hits = vs.search(client, vector(2, dim), limit=3, name="test_chunks")
        check("search returns hits", len(hits) == 3, str(len(hits)))
        check("nearest hit is the exact match",
              hits[0][1]["chunk_id"] == "docA:1:2", hits[0][1]["chunk_id"])
        check("scores descend", all(a >= b for a, b in
                                    zip([h[0] for h in hits],
                                        [h[0] for h in hits][1:])))
        payload = hits[0][1]
        check("offsets survive the round trip",
              payload["char_start"] == 200 and payload["char_end"] == 290,
              f"{payload['char_start']}..{payload['char_end']}")
        check("text travels in the payload, no DB lookup needed",
              payload["text"] == "অনুচ্ছেদ 2", payload["text"])

        # a second document, then filtered search and targeted deletion
        vs.upsert_chunks(client, [
            (f"docB:1:{i}", vector(50 + i, dim), {
                "chunk_id": f"docB:1:{i}", "document_id": "docB",
                "page_no": 1, "ordinal": i, "char_start": 0, "char_end": 10,
                "text": "খ", "model": "test:8"}) for i in range(3)
        ], name="test_chunks")
        check("both documents are present",
              vs.count(client, "test_chunks") == 8,
              str(vs.count(client, "test_chunks")))

        filtered = vs.search(client, vector(2, dim), limit=10,
                             document_id="docA", name="test_chunks")
        check("filtered search returns only that document",
              all(h[1]["document_id"] == "docA" for h in filtered),
              str(len(filtered)))

        # the orphan case: re-index with fewer chunks
        vs.delete_document(client, "docA", name="test_chunks")
        check("deleting a document removes only its points",
              vs.count(client, "test_chunks") == 3,
              str(vs.count(client, "test_chunks")))
        vs.upsert_chunks(client, records[:2], name="test_chunks")
        check("a shorter re-index leaves no orphaned points",
              vs.count(client, "test_chunks") == 5,
              str(vs.count(client, "test_chunks")))
    finally:
        if hasattr(client, "close"):
            client.close()

    # a dimension change must force a recreate
    client = vs.connect()
    try:
        recreated = vs.ensure_collection(client, 16, name="test_chunks")
        check("a dimension change recreates the collection", recreated)
        _, got_dim = vs.collection_info(client, "test_chunks")
        check("recreated collection has the new dimension", got_dim == 16,
              str(got_dim))
        check("recreate drops the old points - a rebuild is required",
              vs.count(client, "test_chunks") == 0,
              str(vs.count(client, "test_chunks")))
    finally:
        if hasattr(client, "close"):
            client.close()


try:
    main()
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nall passed")
sys.exit(1 if failures else 0)
