"""extract - triage the text layer, and take it only where it is trustworthy.

This is the highest-risk stage in the system. On the two sample documents the
embedded text layer is 92-96% Bengali codepoints and systematically wrong,
because the fonts carry no /ToUnicode cmap (see docs/extraction-findings.md).
Corrupt text embeds cleanly, retrieves cleanly, and produces confident wrong
answers that the abstention gate cannot detect - because the bad text IS the
corpus. So this stage assumes nothing and checks per page.

Pages that fail triage are recorded as source='needs_ocr' and left with no text.
The `ocr` stage fills them in. Nothing here silently guesses.
"""
import json

import pymupdf
from .. import db
from ..db import now_iso, to_json
from ..config import VALIDITY_THRESHOLD, stage_dir
from ..bengali import (LEGACY_FONT_HINT, bengali_ratio, is_bengali,
                       orthographic_report)
from .stages import Stage, StageResult, register


def font_key(name):
    """Normalize a font name so the two PyMuPDF spellings compare equal.

    `page.get_fonts()` reports the embedded name with its subset prefix
    ("BCDJEE+FNNuri52Unicode"), while `get_text("dict")` reports span fonts
    without the prefix and truncated to 24 characters
    ("ShorifKarukaVintageUnico"). Comparing the raw strings silently never
    matches, which makes every page look clean.
    """
    name = (name or "").split("+", 1)[-1]
    return name[:24]


def suspect_fonts(doc, page_index):
    """Normalized names of fonts on this page that cannot round-trip to Unicode.

    A font with no /ToUnicode cmap, or one from the legacy Bijoy/SutonnyMJ
    family, cannot be trusted. Both sample documents mix good and bad fonts
    within a single file, so this is judged per page, not per document.
    """
    bad = set()
    for xref, _, _, name, *_ in doc[page_index].get_fonts():
        obj = doc.xref_object(xref)
        if "/ToUnicode" not in obj or LEGACY_FONT_HINT.search(name or ""):
            bad.add(font_key(name))
    return bad


def page_font_verdict(doc, page_index):
    """(ok, offending_fonts) for the Bengali text on this page.

    Only fonts that actually *draw Bengali* matter. Latin fonts legitimately use
    WinAnsiEncoding without a /ToUnicode cmap, so condemning every such font
    would send any page with an English footer to OCR - correct but wasteful,
    and at thousands of pages OCR is the dominant cost. So attribute fonts to
    spans and judge only the spans containing Bengali codepoints.
    """
    bad = suspect_fonts(doc, page_index)
    if not bad:
        return True, []

    offending = set()
    page = doc[page_index]
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if font_key(span.get("font")) in bad and any(
                        is_bengali(c) for c in text):
                    offending.add(span.get("font"))

    return (not offending), sorted(offending)


@register
class Extract(Stage):
    name = "extract"
    description = ("Triage each page's text layer; take the text where it is "
                   "trustworthy, mark the rest for OCR.")

    def run(self, doc):
        pdf_path = doc["storage_path"]
        pdf = pymupdf.open(pdf_path)
        pages = []

        try:
            for i in range(pdf.page_count):
                raw = pdf[i].get_text()
                fonts_ok, bad_fonts = page_font_verdict(pdf, i)
                orth = orthographic_report(raw)
                ratio = bengali_ratio(raw)
                visible = len(raw.strip())

                illegal_rate = orth["initial_dependent_rate"]
                orthography_ok = (illegal_rate <= VALIDITY_THRESHOLD
                                  and orth["triple_letter_runs"] == 0)

                if visible < 20:
                    source = "empty"       # image-only page; OCR is the only option
                elif fonts_ok and orthography_ok:
                    source = "textlayer"
                else:
                    source = "needs_ocr"

                quality = {
                    "bengali_ratio": round(ratio, 4),
                    "initial_dependent_rate": round(illegal_rate, 4),
                    "initial_dependent_samples": orth["initial_dependent_samples"],
                    "triple_letter_runs": orth["triple_letter_runs"],
                    "tokens": orth["tokens"],
                    "chars": visible,
                    "fonts_ok": fonts_ok,
                    "bad_fonts": bad_fonts,
                }
                pages.append({
                    "page_no": i + 1,
                    "source": source,
                    "raw_text": raw if source == "textlayer" else None,
                    "quality": quality,
                })
        finally:
            page_count = pdf.page_count
            pdf.close()

        with db.connect() as conn:
            conn.execute("UPDATE documents SET page_count = %s WHERE id = %s",
                         (page_count, doc["id"]))
            for p in pages:
                conn.execute(
                    """
                    INSERT INTO document_pages
                        (document_id, page_no, source, raw_text, quality, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, page_no) DO UPDATE
                       SET source = EXCLUDED.source,
                           raw_text = EXCLUDED.raw_text,
                           quality = EXCLUDED.quality,
                           text = NULL,          -- normalize must run again
                           trustworthy = NULL,
                           ocr_engine = NULL,
                           updated_at = excluded.updated_at
                    """,
                    (doc["id"], p["page_no"], p["source"], p["raw_text"],
                     to_json(p["quality"]), now_iso()),
                )
            conn.commit()

        counts = {}
        for p in pages:
            counts[p["source"]] = counts.get(p["source"], 0) + 1

        # artifact: the full triage, so a decision can be audited without re-running
        artifact = stage_dir(doc["id"], "extract") / "triage.json"
        artifact.write_text(json.dumps(pages, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        return StageResult(
            status="succeeded",
            output_ref=str(artifact),
            metrics={
                "pages": page_count,
                "by_source": counts,
                "text_layer_usable_pages": counts.get("textlayer", 0),
                "pages_needing_ocr": counts.get("needs_ocr", 0) + counts.get("empty", 0),
            },
            note=(f"{counts.get('textlayer', 0)}/{page_count} pages have a usable "
                  f"text layer; {counts.get('needs_ocr', 0) + counts.get('empty', 0)} "
                  f"need OCR"),
        )
