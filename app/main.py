"""Admin API for the ingestion pipeline.

Every stage is visible and individually triggerable, which is the phase-1
requirement. The endpoints are a thin shell over app.pipeline.runner - the same
functions an auto-mode worker would call - so switching to automatic progression
later adds a worker, it does not rewrite this.
"""
import hashlib
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from . import db
from .db import new_id, now_iso
from .config import DEFAULT_LANGUAGE, OCR_ENGINE, STORAGE_DIR, document_dir
from .pipeline import runner, stages

app = FastAPI(
    title="RAG ingestion admin",
    description="Bengali-first RAG knowledge base - phase 1 (manual stage control)",
)


@app.on_event("startup")
def startup():
    stages.load_all()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    db_ok, db_msg = db.healthcheck()
    return {
        "database": {"ok": db_ok, "detail": db_msg},
        "stages_implemented": stages.implemented(),
        "stages_pending": [s for s in stages.STAGE_ORDER
                           if s not in stages.implemented()],
        "ocr_engine": OCR_ENGINE or None,
        "storage": str(STORAGE_DIR),
    }


@app.post("/documents")
async def upload(file: UploadFile = File(...),
                 language: str = Query(DEFAULT_LANGUAGE)):
    """Upload a PDF. Content-hashed, so re-uploading the same file is a no-op."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")

    tmp = STORAGE_DIR / f".incoming-{file.filename}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with open(tmp, "wb") as out:
        while chunk := await file.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    sha = digest.hexdigest()

    existing = db.query("SELECT * FROM documents WHERE sha256 = %s", (sha,), one=True)
    if existing:
        tmp.unlink(missing_ok=True)
        return {"document": _doc_summary(existing), "created": False,
                "note": "identical file already uploaded"}

    doc_id = new_id()
    db.execute(
        """
        INSERT INTO documents (id, filename, sha256, storage_path, byte_size,
                               language, uploaded_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (doc_id, file.filename, sha, "", size, language, now_iso()),
    )
    row = db.query("SELECT * FROM documents WHERE id = %s", (doc_id,), one=True)
    dest = document_dir(doc_id) / "source.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), dest)
    db.execute("UPDATE documents SET storage_path = %s WHERE id = %s",
               (str(dest), doc_id))
    row["storage_path"] = str(dest)

    runner.ensure_stage_runs(doc_id)
    return {"document": _doc_summary(row), "created": True,
            "next_stage": runner.next_runnable(doc_id)}


@app.get("/documents")
def list_documents():
    rows = db.query("SELECT * FROM documents ORDER BY uploaded_at DESC")
    return {"documents": [_doc_summary(r) for r in rows]}


@app.get("/documents/{document_id}")
def document_detail(document_id: str):
    doc = db.query("SELECT * FROM documents WHERE id = %s", (document_id,), one=True)
    if not doc:
        raise HTTPException(404, "no such document")

    runner.ensure_stage_runs(document_id)
    stage_list = []
    for row in runner.stage_rows(document_id):
        name = row["stage"]
        stage_list.append({
            "stage": name,
            "status": row["status"],
            "attempt": row["attempt"],
            "triggered_by": row["triggered_by"],
            "metrics": row["metrics"],
            "output_ref": row["output_ref"],
            "error": row["error"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "implemented": name in stages.implemented(),
            "blockers": runner.blockers(document_id, name),
        })

    pages = db.query(
        """
        SELECT page_no, source, ocr_engine, trustworthy,
               length(text) AS chars, quality
          FROM document_pages WHERE document_id = %s ORDER BY page_no
        """,
        (document_id,),
    )
    return {
        "document": _doc_summary(doc),
        "stages": stage_list,
        "next_runnable": runner.next_runnable(document_id),
        "pages": pages,
    }


@app.post("/documents/{document_id}/stages/{stage}/run")
def run_stage(document_id: str, stage: str, force: bool = Query(False)):
    """Manually trigger one stage.

    force=true re-runs a stage that already succeeded, and marks every later
    stage stale - their results describe input that no longer exists.
    """
    try:
        return runner.run_stage(document_id, stage,
                                triggered_by="manual", force=force)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except KeyError as exc:
        raise HTTPException(501, str(exc).strip('"'))
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.post("/documents/{document_id}/advance")
def advance(document_id: str, max_stages: int = Query(None)):
    """Run stages until one blocks. This is auto mode, exposed manually.

    Present to prove the manual path did not paint us into a corner: a worker
    would call runner.advance on a loop instead of an admin hitting this.
    """
    try:
        return {"ran": runner.advance(document_id, triggered_by="auto",
                                      max_stages=max_stages)}
    except (RuntimeError, NotImplementedError, KeyError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/documents/{document_id}/pages/{page_no}", response_class=PlainTextResponse)
def page_text(document_id: str, page_no: int, raw: bool = Query(False)):
    """The actual text for one page - the point of 'every stage visible'.

    raw=true returns the text as obtained, before NFC normalization, which is
    what you want when deciding whether a page is genuinely corrupt.
    """
    row = db.query(
        "SELECT raw_text, text FROM document_pages "
        "WHERE document_id = %s AND page_no = %s",
        (document_id, page_no), one=True)
    if not row:
        raise HTTPException(404, "no such page")
    return (row["raw_text"] if raw else row["text"]) or ""


@app.get("/documents/{document_id}/artifacts/{stage}", response_class=PlainTextResponse)
def artifact(document_id: str, stage: str):
    """Raw stage artifact, so a decision can be audited without re-running it."""
    row = db.query(
        "SELECT output_ref FROM stage_runs WHERE document_id = %s AND stage = %s",
        (document_id, stage), one=True)
    if not row or not row["output_ref"]:
        raise HTTPException(404, "no artifact for that stage")
    path = Path(row["output_ref"])
    # never serve outside the storage root, whatever the database says
    if not path.resolve().is_relative_to(STORAGE_DIR) or not path.exists():
        raise HTTPException(404, "artifact missing")
    return path.read_text(encoding="utf-8")


@app.get("/review")
def review_queue():
    """Documents held because pages failed the Bengali validity check."""
    rows = db.query(
        "SELECT * FROM documents WHERE review_required = 1 "
        "ORDER BY uploaded_at DESC")
    return {"documents": [_doc_summary(r) for r in rows]}


def _doc_summary(row):
    return {
        "id": str(row["id"]),
        "filename": row["filename"],
        "sha256": row["sha256"][:12],
        "page_count": row["page_count"],
        "language": row["language"],
        "byte_size": row["byte_size"],
        "review_required": row["review_required"],
        "review_note": row["review_note"],
        "uploaded_at": row["uploaded_at"],
    }
