"""Stage execution and state transitions.

This is the whole difference between manual and automatic mode. `run_stage` is
called either by an admin clicking a button or by a worker claiming a pending
row - the guard rules, artifact handling and status transitions are identical.

The rule worth understanding is downstream invalidation. Re-running extract
means every later stage's result describes input that no longer exists, so those
rows go to 'stale' rather than staying 'succeeded'. Without this, a re-extracted
document keeps an index built from the old text and nothing tells you.
"""
import traceback

from .. import db
from . import stages
from ..db import now_iso, to_json


def ensure_stage_runs(document_id):
    """Give a document its full row-per-stage set, all pending."""
    with db.connect() as conn:
        for stage in stages.STAGE_ORDER:
            conn.execute(
                """
                INSERT INTO stage_runs (document_id, stage, status, created_at)
                VALUES (%s, %s, 'pending', %s)
                ON CONFLICT (document_id, stage) DO NOTHING
                """,
                (document_id, stage, now_iso()),
            )
        conn.commit()


def stage_rows(document_id):
    """All stage rows for a document, in execution order."""
    rows = db.query(
        "SELECT * FROM stage_runs WHERE document_id = %s", (document_id,))
    by_name = {r["stage"]: r for r in rows}
    return [by_name[s] for s in stages.STAGE_ORDER if s in by_name]


def blockers(document_id, stage):
    """Why `stage` cannot run right now. Empty list means it can.

    A stage is runnable when every earlier stage has succeeded or been skipped.
    'held' explicitly blocks: that is the review gate doing its job.
    """
    if stage not in stages.STAGE_ORDER:
        return [f"unknown stage {stage!r}"]
    rows = {r["stage"]: r for r in stage_rows(document_id)}
    reasons = []
    for earlier in stages.STAGE_ORDER[:stages.STAGE_ORDER.index(stage)]:
        st = rows.get(earlier, {}).get("status")
        if st in ("succeeded", "skipped"):
            continue
        if st == "held":
            reasons.append(f"{earlier} is held for review")
        else:
            reasons.append(f"{earlier} is {st}")
    return reasons


def mark_downstream_stale(conn, document_id, stage):
    """Anything after `stage` that had a result no longer describes the input."""
    later = stages.STAGE_ORDER[stages.STAGE_ORDER.index(stage) + 1:]
    if not later:
        return 0
    placeholders = ", ".join(["%s"] * len(later))
    cur = conn.execute(
        f"""
        UPDATE stage_runs SET status = 'stale'
        WHERE document_id = %s AND stage IN ({placeholders})
          AND status IN ('succeeded', 'failed', 'held', 'skipped')
        """,
        (document_id, *later),
    )
    return cur.rowcount


def run_stage(document_id, stage, triggered_by="manual", force=False):
    """Execute one stage. Returns the updated stage row as a dict.

    force=True re-runs a stage that already succeeded, which is the point of
    having per-stage triggers at all - you fix the extract logic and re-run
    extract on one document without re-uploading it.
    """
    doc = db.query("SELECT * FROM documents WHERE id = %s", (document_id,), one=True)
    if not doc:
        raise LookupError(f"no document {document_id}")

    ensure_stage_runs(document_id)
    row = db.query(
        "SELECT * FROM stage_runs WHERE document_id = %s AND stage = %s",
        (document_id, stage), one=True)
    if not row:
        raise LookupError(f"no stage {stage!r} for document {document_id}")

    if row["status"] == "running":
        raise RuntimeError(f"{stage} is already running for this document")
    if row["status"] == "succeeded" and not force:
        raise RuntimeError(
            f"{stage} already succeeded; pass force=true to re-run it")

    reasons = blockers(document_id, stage)
    if reasons:
        raise RuntimeError(f"cannot run {stage}: " + "; ".join(reasons))

    impl = stages.get(stage)  # raises with a clear message if not implemented

    with db.connect() as conn:
        conn.execute(
            """
            UPDATE stage_runs
               SET status = 'running', attempt = attempt + 1,
                   triggered_by = %s, started_at = %s,
                   finished_at = NULL, error = NULL
             WHERE document_id = %s AND stage = %s
            """,
            (triggered_by, now_iso(), document_id, stage),
        )
        conn.commit()

    try:
        result = impl.run(doc)
    except Exception:
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE stage_runs
                   SET status = 'failed', finished_at = %s, error = %s
                 WHERE document_id = %s AND stage = %s
                """,
                (now_iso(), traceback.format_exc(limit=8), document_id, stage),
            )
            conn.commit()
        raise

    with db.connect() as conn:
        conn.execute(
            """
            UPDATE stage_runs
               SET status = %s, finished_at = %s, output_ref = %s,
                   metrics = %s, error = NULL
             WHERE document_id = %s AND stage = %s
            """,
            (result.status, now_iso(), result.output_ref,
             to_json(result.metrics), document_id, stage),
        )
        stale = mark_downstream_stale(conn, document_id, stage)
        conn.commit()

    out = db.query(
        "SELECT * FROM stage_runs WHERE document_id = %s AND stage = %s",
        (document_id, stage), one=True)
    out["downstream_marked_stale"] = stale
    out["note"] = result.note
    return out


#: How many times auto mode will retry a failing stage before giving up on it.
#: Without a retry path a stage that failed once is stuck forever - a worker
#: would skip that document silently even after the cause is fixed. Without a
#: limit, a permanently-broken stage would be retried on every sweep.
MAX_STAGE_ATTEMPTS = 3


def is_runnable(row):
    """Whether a stage row is eligible to be picked up, ignoring blockers."""
    if row["status"] in ("pending", "stale"):
        return True
    # a failed stage stays retryable until it has burned its attempts; past that
    # it needs a human, who can still force it explicitly
    return row["status"] == "failed" and row["attempt"] < MAX_STAGE_ATTEMPTS


def next_runnable(document_id):
    """The first stage that could run now, or None. Used by auto mode."""
    for row in stage_rows(document_id):
        if is_runnable(row) and not blockers(document_id, row["stage"]):
            return row["stage"]
    return None


def advance(document_id, triggered_by="auto", max_stages=None):
    """Run stages until something blocks. This IS auto mode.

    Phase 1 drives stages by hand, so nothing calls this yet - but it exists to
    prove the manual path did not paint us into a corner. A background worker
    would loop over documents calling this instead of an admin clicking.
    """
    ran = []
    while True:
        stage = next_runnable(document_id)
        if stage is None or stage not in stages.implemented():
            break
        if max_stages is not None and len(ran) >= max_stages:
            break
        try:
            row = run_stage(document_id, stage, triggered_by=triggered_by,
                            force=True)
        except Exception as exc:
            # A stage failing is a normal outcome here, not an error in advancing.
            # run_stage has already recorded 'failed' with the traceback, so a
            # worker should stop this document and move on to the next one -
            # not crash the loop and take every other document down with it.
            ran.append({"stage": stage, "status": "failed", "error": str(exc)})
            break
        ran.append({"stage": stage, "status": row["status"]})
        if row["status"] in ("held", "failed"):
            break
    return ran
