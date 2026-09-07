"""ocr - fill in the pages whose text layer extract refused to trust.

Reuses the engine adapters from tools/ocr_bench so the benchmark and production
run identical code. Whichever engine wins the benchmark becomes OCR_ENGINE in
the environment; nothing else changes.

Skips itself when every page passed triage, and fails loudly rather than
guessing when pages need OCR and no engine is configured. A silent skip there
would leave holes in the corpus that only surface as unanswerable questions.
"""
import json
import sys
from os.path import abspath, dirname

import pymupdf

from .. import db
from ..config import OCR_DPI, OCR_ENGINE, stage_dir
from .stages import Stage, StageResult, register

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))


def available_engines():
    from tools.ocr_bench.engines import OCR_ENGINES
    return OCR_ENGINES


@register
class Ocr(Stage):
    name = "ocr"
    description = ("Render and OCR the pages extract marked as needs_ocr or "
                   "empty. No-op when the text layer covered everything.")

    def run(self, doc):
        pending = db.query(
            """
            SELECT page_no FROM document_pages
             WHERE document_id = %s AND source IN ('needs_ocr', 'empty')
             ORDER BY page_no
            """,
            (doc["id"],),
        )
        page_nos = [r["page_no"] for r in pending]

        if not page_nos:
            return StageResult(
                status="skipped",
                metrics={"pages_ocred": 0},
                note="every page had a usable text layer; nothing to OCR",
            )

        if not OCR_ENGINE:
            raise RuntimeError(
                f"{len(page_nos)} page(s) need OCR but OCR_ENGINE is not set. "
                f"Run the benchmark (tools/ocr_bench) to pick an engine, then "
                f"set OCR_ENGINE to one of: {sorted(available_engines())}"
            )

        engines = available_engines()
        if OCR_ENGINE not in engines:
            raise RuntimeError(
                f"OCR_ENGINE={OCR_ENGINE!r} is unknown; "
                f"available: {sorted(engines)}")

        engine = engines[OCR_ENGINE]()
        ok, why = engine.available()
        if not ok:
            raise RuntimeError(f"OCR engine {OCR_ENGINE!r} is not usable: {why}")

        render_dir = stage_dir(doc["id"], "ocr") / "render"
        render_dir.mkdir(parents=True, exist_ok=True)

        pdf = pymupdf.open(doc["storage_path"])
        done, failures = [], []
        try:
            for page_no in page_nos:
                png = render_dir / f"page_{page_no:04d}.png"
                if not png.exists():
                    pdf[page_no - 1].get_pixmap(dpi=OCR_DPI).save(png)
                try:
                    text, secs = engine.run(str(png))
                except Exception as exc:
                    failures.append({"page_no": page_no, "error": str(exc)})
                    continue
                with db.connect() as conn:
                    conn.execute(
                        """
                        UPDATE document_pages
                           SET source = 'ocr', ocr_engine = %s, raw_text = %s,
                               text = NULL, trustworthy = NULL, updated_at = now()
                         WHERE document_id = %s AND page_no = %s
                        """,
                        (OCR_ENGINE, text, doc["id"], page_no),
                    )
                    conn.commit()
                done.append({"page_no": page_no, "chars": len(text),
                             "seconds": round(secs, 2)})
        finally:
            pdf.close()

        artifact = stage_dir(doc["id"], "ocr") / "ocr_runs.json"
        artifact.write_text(
            json.dumps({"engine": OCR_ENGINE, "dpi": OCR_DPI,
                        "pages": done, "failures": failures},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")

        if failures:
            raise RuntimeError(
                f"OCR failed on {len(failures)} of {len(page_nos)} page(s): "
                f"{failures[:3]}")

        return StageResult(
            status="succeeded",
            output_ref=str(artifact),
            metrics={
                "engine": OCR_ENGINE,
                "dpi": OCR_DPI,
                "pages_ocred": len(done),
                "total_seconds": round(sum(d["seconds"] for d in done), 1),
            },
            note=f"OCR'd {len(done)} page(s) with {OCR_ENGINE}",
        )
