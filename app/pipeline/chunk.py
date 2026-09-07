"""chunk - split normalized page text into retrievable units.

Chunks never cross a page boundary, so every chunk cites exactly one page, and
every chunk records the character offsets it came from. That is what makes a
citation checkable later: the abstention gate rejects an answer whose cited span
does not support its claim, and it can only do that if the span is exact.

Pages that failed the Bengali validity check are skipped rather than chunked.
normalize already holds such a document, but if a human clears the hold, the
individual bad pages still must not reach the index - corrupt text that gets
indexed cannot be detected afterwards, because by then it *is* the corpus.
"""
import json

from .. import db
from ..chunking import chunk_page
from ..config import (CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS,
                      CHUNK_TARGET_CHARS, stage_dir)
from ..db import now_iso
from .stages import Stage, StageResult, register


@register
class Chunk(Stage):
    name = "chunk"
    description = ("Split normalized page text into retrievable chunks, keeping "
                   "exact offsets so citations stay verifiable.")

    def run(self, doc):
        pages = db.query(
            """
            SELECT page_no, text, trustworthy FROM document_pages
             WHERE document_id = %s AND text IS NOT NULL
             ORDER BY page_no
            """,
            (doc["id"],),
        )
        if not pages:
            raise RuntimeError("no normalized page text; run normalize first")

        usable = [p for p in pages if p["trustworthy"]]
        skipped = [p["page_no"] for p in pages if not p["trustworthy"]]
        if not usable:
            raise RuntimeError(
                f"every page failed the Bengali validity check "
                f"({len(skipped)} page(s)); nothing safe to index")

        rows, report = [], []
        for page in usable:
            chunks = chunk_page(
                page["text"],
                target_chars=CHUNK_TARGET_CHARS,
                max_chars=CHUNK_MAX_CHARS,
                min_chars=CHUNK_MIN_CHARS,
                overlap_chars=CHUNK_OVERLAP_CHARS,
            )
            for c in chunks:
                rows.append((
                    f"{doc['id']}:{page['page_no']}:{c['ordinal']}",
                    doc["id"], page["page_no"], c["ordinal"],
                    c["char_start"], c["char_end"],
                    c["text"], c["embed_text"], c["n_chars"], now_iso(),
                ))
            report.append({
                "page_no": page["page_no"],
                "page_chars": len(page["text"]),
                "chunks": len(chunks),
                "sizes": [c["n_chars"] for c in chunks],
            })

        with db.connect() as conn:
            # replace wholesale: re-running must not leave chunks from a
            # previous, differently-sized run alongside the new ones
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc["id"],))
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO chunks
                        (id, document_id, page_no, ordinal, char_start, char_end,
                         text, embed_text, n_chars, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    row,
                )
            conn.commit()

        artifact = stage_dir(doc["id"], "chunk") / "chunks.json"
        artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        sizes = [r[8] for r in rows]
        note = (f"{len(rows)} chunks from {len(usable)} page(s)"
                + (f"; skipped {len(skipped)} page(s) that failed validity"
                   if skipped else ""))
        return StageResult(
            status="succeeded",
            output_ref=str(artifact),
            metrics={
                "chunks": len(rows),
                "pages_chunked": len(usable),
                "pages_skipped": len(skipped),
                "skipped_page_numbers": skipped[:50],
                "mean_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0,
                "min_chars": min(sizes) if sizes else 0,
                "max_chars": max(sizes) if sizes else 0,
                "target_chars": CHUNK_TARGET_CHARS,
            },
            note=note,
        )
