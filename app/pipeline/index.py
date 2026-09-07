"""index - upsert a document's vectors into Qdrant.

Reads the embed stage's artifact rather than re-embedding, because the artifact
is the durable record and the collection is a rebuildable projection of it.

Two correctness concerns drive most of this code:

  Mixed models. Vectors from two different embedding models in one collection
  do not error - they return confident nonsense that nothing downstream can
  detect, including the abstention gate, because the scores look normal. So the
  manifest's model is checked against vector_collections. On a mismatch the
  collection is recreated and every *other* document's index stage is marked
  stale, which puts the required re-index in the admin stage table instead of
  leaving a quietly poisoned corpus.

  Orphaned points. Chunk ids are deterministic, so re-indexing overwrites
  points that still exist. But if the chunker produced fewer chunks this time,
  the surplus points from the previous run would linger and keep being
  retrieved. So a document's points are deleted before its upsert.
"""
import json

from .. import db
from ..config import QDRANT_COLLECTION, stage_dir
from ..db import now_iso
from ..embedding import unpack
from .. import vectorstore as vs
from .stages import STAGE_ORDER, Stage, StageResult, register  # noqa: F401


def _load_vectors(vectors_path, dim, n_chunks):
    """Read the raw float32 artifact into a list of vectors, checking the size.

    The size check matters: a truncated artifact would otherwise be read as
    fewer (or misaligned) vectors, and misaligned vectors mean every citation
    points at the wrong chunk while looking perfectly valid.
    """
    raw = open(vectors_path, "rb").read()
    expected = n_chunks * dim * 4
    if len(raw) != expected:
        raise RuntimeError(
            f"vector artifact is {len(raw)} bytes, expected {expected} "
            f"({n_chunks} x {dim} float32). Re-run the embed stage.")
    stride = dim * 4
    return [unpack(raw[i * stride:(i + 1) * stride]) for i in range(n_chunks)]


@register
class Index(Stage):
    name = "index"
    description = ("Upsert the document's vectors into the Qdrant collection, "
                   "guarding against a mixed-model corpus.")

    def run(self, doc):
        ok, why = vs.available()
        if not ok:
            raise RuntimeError(f"vector store unavailable: {why}")

        meta = db.query(
            "SELECT * FROM document_embeddings WHERE document_id = %s",
            (doc["id"],), one=True)
        if not meta:
            raise RuntimeError("no embeddings; run the embed stage first")

        manifest = json.loads(
            open(meta["manifest_path"], encoding="utf-8").read())
        model, dim = manifest["model"], manifest["dim"]
        chunk_ids = manifest["chunk_ids"]

        if model != meta["model"]:
            raise RuntimeError(
                f"manifest model {model!r} disagrees with the recorded "
                f"{meta['model']!r}; re-run embed")

        vectors = _load_vectors(meta["vectors_path"], dim, len(chunk_ids))

        chunks = {c["id"]: c for c in db.query(
            "SELECT id, page_no, ordinal, char_start, char_end, text "
            "FROM chunks WHERE document_id = %s", (doc["id"],))}
        missing = [cid for cid in chunk_ids if cid not in chunks]
        if missing:
            raise RuntimeError(
                f"{len(missing)} chunk(s) in the manifest no longer exist "
                f"(e.g. {missing[0]}); the chunk stage re-ran after embed. "
                f"Re-run embed.")

        client = vs.connect()
        try:
            tracked = db.query(
                "SELECT * FROM vector_collections WHERE name = %s",
                (QDRANT_COLLECTION,), one=True)

            model_changed = bool(tracked) and tracked["model"] != model
            created = vs.ensure_collection(client, dim, recreate=model_changed)

            stale_marked = 0
            if model_changed:
                # every other document's points went away with the collection
                with db.connect() as conn:
                    cur = conn.execute(
                        """
                        UPDATE stage_runs SET status = 'stale'
                         WHERE stage = 'index' AND document_id != %s
                           AND status IN ('succeeded', 'held')
                        """,
                        (doc["id"],),
                    )
                    stale_marked = cur.rowcount
                    conn.commit()

            if not created:
                vs.delete_document(client, doc["id"])

            records = []
            for chunk_id, vector in zip(chunk_ids, vectors):
                c = chunks[chunk_id]
                records.append((chunk_id, vector, {
                    "chunk_id": chunk_id,
                    "document_id": doc["id"],
                    "filename": doc["filename"],
                    "language": doc["language"],
                    "page_no": c["page_no"],
                    "ordinal": c["ordinal"],
                    # offsets travel with the point so a retrieved hit can be
                    # resolved back to exact page characters for citation
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                    # carried in the payload so retrieval needs no database
                    # round-trip - the 5s budget does not allow a lookup per hit
                    "text": c["text"],
                    "model": model,
                }))

            upserted = vs.upsert_chunks(client, records)
            total = vs.count(client)
        finally:
            # embedded mode holds an exclusive directory lock; releasing it
            # here is what lets the next process open the store at all
            if hasattr(client, "close"):
                client.close()

        now = now_iso()
        with db.connect() as conn:
            if tracked:
                conn.execute(
                    "UPDATE vector_collections SET model = %s, dim = %s, "
                    "updated_at = %s WHERE name = %s",
                    (model, dim, now, QDRANT_COLLECTION))
            else:
                conn.execute(
                    "INSERT INTO vector_collections "
                    "(name, model, dim, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (QDRANT_COLLECTION, model, dim, now, now))
            conn.commit()

        artifact = stage_dir(doc["id"], "index") / "index.json"
        artifact.write_text(json.dumps({
            "collection": QDRANT_COLLECTION,
            "model": model,
            "dim": dim,
            "points_upserted": upserted,
            "collection_recreated": created,
            "model_changed": model_changed,
            "other_documents_marked_stale": stale_marked,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        note = f"indexed {upserted} points into {QDRANT_COLLECTION!r}"
        if model_changed:
            note += (f"; embedding model changed to {model!r}, so the collection "
                     f"was rebuilt and {stale_marked} other document(s) need "
                     f"re-indexing")

        return StageResult(
            status="succeeded",
            output_ref=str(artifact),
            metrics={
                "collection": QDRANT_COLLECTION,
                "model": model,
                "dim": dim,
                "points_upserted": upserted,
                "points_in_collection": total,
                "collection_recreated": created,
                "model_changed": model_changed,
                "other_documents_marked_stale": stale_marked,
            },
            note=note,
        )
