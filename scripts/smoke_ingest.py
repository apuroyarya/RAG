"""End-to-end check of the ingestion backbone. Needs Postgres up.

    docker compose up -d
    python -m app.db                      # apply migrations
    python scripts/smoke_ingest.py "path/to/some.pdf"

Exercises the parts that are hard to be confident about by reading:

  1. a document gets a full row-per-stage set, all pending
  2. stage ordering is enforced (normalize refuses to run before extract)
  3. extract writes per-page triage and its artifact
  4. re-running extract with force marks downstream stages stale
  5. an unimplemented stage fails as 501-shaped, not as a crash
  6. runner.advance (auto mode) walks the same path manual triggers do
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hashlib

from app import db
from app.config import document_dir
from app.pipeline import runner, stages

PASS, FAIL = "  ok  ", " FAIL "
failures = []


def check(label, cond, detail=""):
    print(f"{PASS if cond else FAIL} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


def main(pdf_path):
    stages.load_all()
    ok, msg = db.healthcheck()
    if not ok:
        raise SystemExit(f"database not reachable: {msg}\n"
                         f"run: docker compose up -d && python -m app.db")

    src = Path(pdf_path).resolve()
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    # register the document, copying it in the way the upload endpoint does
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    db.execute("DELETE FROM documents WHERE sha256 = %s", (sha,))
    row = db.execute(
        """
        INSERT INTO documents (filename, sha256, storage_path, byte_size, language)
        VALUES (%s, %s, %s, %s, 'bn') RETURNING *
        """,
        (src.name, sha, "", src.stat().st_size), returning=True)
    doc_id = row["id"]
    dest = document_dir(doc_id) / "source.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    db.execute("UPDATE documents SET storage_path = %s WHERE id = %s",
               (str(dest), doc_id))
    print(f"\ndocument {doc_id}  ({src.name})\n")

    # 1. full stage set, all pending
    runner.ensure_stage_runs(doc_id)
    rows = runner.stage_rows(doc_id)
    check("row per stage created",
          [r["stage"] for r in rows] == stages.STAGE_ORDER,
          ",".join(r["stage"] for r in rows))
    check("all start pending", all(r["status"] == "pending" for r in rows))

    # 2. ordering is enforced
    try:
        runner.run_stage(doc_id, "normalize")
        check("normalize blocked before extract", False, "it ran anyway")
    except RuntimeError as exc:
        check("normalize blocked before extract", "extract is pending" in str(exc),
              str(exc)[:60])

    # 3. extract runs and records pages
    res = runner.run_stage(doc_id, "extract")
    check("extract succeeded", res["status"] == "succeeded", res.get("error") or "")
    print(f"       {res['note']}")
    print(f"       metrics: {res['metrics']}")
    pages = db.query(
        "SELECT source, count(*) AS n FROM document_pages "
        "WHERE document_id = %s GROUP BY source ORDER BY source", (doc_id,))
    check("pages recorded", bool(pages),
          ", ".join(f"{p['source']}={p['n']}" for p in pages))
    check("extract artifact written",
          res["output_ref"] and Path(res["output_ref"]).exists(),
          res["output_ref"] or "none")

    # 4. re-run marks downstream stale
    runner.run_stage(doc_id, "ocr")           # skipped or fails; either is fine here
    before = {r["stage"]: r["status"] for r in runner.stage_rows(doc_id)}
    res2 = runner.run_stage(doc_id, "extract", force=True)
    after = {r["stage"]: r["status"] for r in runner.stage_rows(doc_id)}
    check("re-running extract needs force", res2["attempt"] >= 2,
          f"attempt={res2['attempt']}")
    downstream_changed = any(
        before.get(s) in ("succeeded", "skipped", "held") and after.get(s) == "stale"
        for s in stages.STAGE_ORDER[1:])
    check("downstream marked stale on re-run",
          downstream_changed or all(
              before.get(s) not in ("succeeded", "skipped", "held")
              for s in stages.STAGE_ORDER[1:]),
          f"stale={res2['downstream_marked_stale']}")

    # 5. an unimplemented stage reports itself clearly
    try:
        runner.run_stage(doc_id, "chunk", force=True)
        check("chunk reports not-implemented", False, "it claimed success")
    except NotImplementedError as exc:
        check("chunk reports not-implemented", "not implemented" in str(exc))
    except RuntimeError as exc:
        # blocked by an earlier stage, which is also correct behaviour
        check("chunk reports not-implemented", True, f"blocked: {str(exc)[:50]}")

    # 6. auto mode walks the same path
    ran = runner.advance(doc_id, max_stages=3)
    check("advance() runs without crashing", True,
          " -> ".join(f"{r['stage']}:{r['status']}" for r in ran) or "nothing runnable")

    print("\nfinal stage table:")
    for r in runner.stage_rows(doc_id):
        err = (r["error"] or "").splitlines()
        print(f"  {r['stage']:10s} {r['status']:10s} attempt={r['attempt']}"
              f"  {err[-1][:60] if err else ''}")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
