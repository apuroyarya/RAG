"""normalize - NFC-normalize every page and gate on Bengali validity.

Two jobs:

  1. NFC normalization. Both PDF extraction and OCR emit Indic conjuncts in
     visual rather than logical order, so without this `ক্ষ` in the corpus will
     not match `ক্ষ` as a user types it, and retrieval silently misses.

  2. The validity gate. Any page still showing Bengali words that begin with a
     dependent vowel sign is corrupt, whatever produced it. Those pages mark the
     document for review and the stage returns 'held', which blocks chunk/embed/
     index. This is the review queue: a corrupt page that reaches the index
     cannot be detected later, because by then the bad text IS the corpus.
"""
import json

from .. import db
from ..db import now_iso
from ..bengali import latin_intrusion
from ..bengali import normalize as nfc_normalize
from ..bengali import orthographic_report
from ..config import VALIDITY_THRESHOLD, stage_dir
from .stages import Stage, StageResult, register

#: Above this share of Latin word tokens, a Bengali page is flagged as likely
#: OCR mis-recognition. A warning, not a gate - the page still indexes.
LATIN_WARN_RATE = 0.10


@register
class Normalize(Stage):
    name = "normalize"
    description = ("NFC-normalize page text and hold the document if any page "
                   "fails the Bengali validity check.")

    def run(self, doc):
        pages = db.query(
            """
            SELECT page_no, source, raw_text FROM document_pages
             WHERE document_id = %s ORDER BY page_no
            """,
            (doc["id"],),
        )
        if not pages:
            raise RuntimeError("no pages recorded; run extract first")

        missing = [p["page_no"] for p in pages
                   if p["source"] in ("needs_ocr", "empty")]
        if missing:
            raise RuntimeError(
                f"{len(missing)} page(s) still have no text (pages {missing[:5]}); "
                f"run the ocr stage first")

        report, bad_pages, suspect_pages = [], [], []
        with db.connect() as conn:
            for p in pages:
                text = nfc_normalize(p["raw_text"] or "", collapse_whitespace=False)
                orth = orthographic_report(text)
                trustworthy = (
                    orth["initial_dependent_rate"] <= VALIDITY_THRESHOLD
                    and orth["triple_letter_runs"] == 0
                )
                conn.execute(
                    """
                    UPDATE document_pages
                       SET text = %s, trustworthy = %s, updated_at = %s
                     WHERE document_id = %s AND page_no = %s
                    """,
                    (text, int(trustworthy), now_iso(), doc["id"], p["page_no"]),
                )
                latin = latin_intrusion(text)
                entry = {
                    "page_no": p["page_no"],
                    "source": p["source"],
                    "chars": len(text),
                    "trustworthy": trustworthy,
                    "initial_dependent_rate": round(orth["initial_dependent_rate"], 4),
                    "samples": orth["initial_dependent_samples"],
                    # reported, not gated on: scattered Latin words in Bengali
                    # prose usually mean OCR mis-recognition, but a Bengali
                    # document may legitimately quote English
                    "latin_rate": round(latin["latin_rate"], 4),
                    "latin_samples": latin["samples"],
                }
                report.append(entry)
                if not trustworthy:
                    bad_pages.append(p["page_no"])
                elif latin["latin_rate"] > LATIN_WARN_RATE:
                    suspect_pages.append(p["page_no"])
            conn.commit()

        artifact = stage_dir(doc["id"], "normalize") / "validity.json"
        artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        note = f"{len(pages) - len(bad_pages)}/{len(pages)} pages passed validity"
        if suspect_pages and not bad_pages:
            note += (f"; {len(suspect_pages)} page(s) have heavy Latin intrusion "
                     f"(pages {suspect_pages[:5]}) - likely OCR mis-recognition, "
                     f"indexed anyway")
        if bad_pages:
            note = (f"{len(bad_pages)} page(s) failed the Bengali validity check "
                    f"(pages {bad_pages[:8]}) - held for review")
            with db.connect() as conn:
                conn.execute(
                    "UPDATE documents SET review_required = 1, review_note = %s "
                    "WHERE id = %s",
                    (note, doc["id"]),
                )
                conn.commit()
        else:
            with db.connect() as conn:
                conn.execute(
                    "UPDATE documents SET review_required = 0, review_note = NULL "
                    "WHERE id = %s", (doc["id"],))
                conn.commit()

        return StageResult(
            # 'held' deliberately blocks downstream stages until a human looks
            status="held" if bad_pages else "succeeded",
            output_ref=str(artifact),
            metrics={
                "pages": len(pages),
                "pages_trustworthy": len(pages) - len(bad_pages),
                "pages_failed": len(bad_pages),
                "failed_page_numbers": bad_pages[:50],
                "pages_latin_suspect": len(suspect_pages),
                "latin_suspect_page_numbers": suspect_pages[:50],
                "total_chars": sum(e["chars"] for e in report),
            },
            note=note,
        )
