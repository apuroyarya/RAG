"""embed - turn chunks into vectors.

Embeds `embed_text` (the chunk plus its lead-in overlap), not `text`. The
overlap exists so a sentence split across a chunk boundary is still findable
from both sides; citations still resolve against `text`, so retrieval gains the
recall without citations borrowing their neighbour's words.

Vectors go to a per-document artifact file and the row here records only
metadata. The artifact is the durable record; the vector store is a rebuildable
projection of it. Re-indexing is then free, and only an actual model change
costs a re-embed.
"""
import json

from .. import db
from ..config import stage_dir
from ..db import now_iso
from ..embedding import get_embedder, pack
from .stages import Stage, StageResult, register


@register
class Embed(Stage):
    name = "embed"
    description = ("Embed chunk text into vectors with the configured model, "
                   "writing them to a per-document artifact.")

    def run(self, doc):
        chunks = db.query(
            """
            SELECT id, embed_text, n_chars FROM chunks
             WHERE document_id = %s ORDER BY page_no, ordinal
            """,
            (doc["id"],),
        )
        if not chunks:
            raise RuntimeError("no chunks; run the chunk stage first")

        embedder = get_embedder()
        ok, why = embedder.available()
        if not ok:
            raise RuntimeError(f"embedding backend is not usable: {why}")

        vectors = embedder.embed([c["embed_text"] for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"backend returned {len(vectors)} vectors for {len(chunks)} "
                f"chunks; refusing to write a misaligned artifact")

        dim = len(vectors[0])
        if any(len(v) != dim for v in vectors):
            raise RuntimeError("backend returned vectors of inconsistent width")

        out = stage_dir(doc["id"], "embed")
        vectors_path = out / "vectors.f32"
        manifest_path = out / "manifest.json"

        with open(vectors_path, "wb") as fh:
            for vec in vectors:
                fh.write(pack(vec))

        manifest = {
            "model": embedder.model_id,
            "dim": dim,
            "n_chunks": len(chunks),
            # row order is the contract between this artifact and the index
            # stage; without it a vector cannot be traced back to its chunk
            "chunk_ids": [c["id"] for c in chunks],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

        with db.connect() as conn:
            conn.execute("DELETE FROM document_embeddings WHERE document_id = %s",
                         (doc["id"],))
            conn.execute(
                """
                INSERT INTO document_embeddings
                    (document_id, model, dim, n_chunks, vectors_path,
                     manifest_path, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (doc["id"], embedder.model_id, dim, len(chunks),
                 str(vectors_path), str(manifest_path), now_iso()),
            )
            conn.commit()

        warning = None
        if embedder.name == "fake":
            warning = ("embedded with the FAKE backend: hashed vectors with no "
                       "semantic structure. Plumbing only - retrieval results "
                       "and any threshold tuned against them are meaningless.")

        return StageResult(
            status="succeeded",
            output_ref=str(manifest_path),
            metrics={
                "model": embedder.model_id,
                "dim": dim,
                "chunks_embedded": len(chunks),
                "bytes_written": vectors_path.stat().st_size,
                "backend": embedder.name,
                **({"warning": warning} if warning else {}),
            },
            note=warning or f"embedded {len(chunks)} chunks with {embedder.model_id}",
        )
