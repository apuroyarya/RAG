"""Database access.

Raw SQL over psycopg3 rather than an ORM. The schema is small, and the stage
state transitions are the interesting logic here - they read better as explicit
SQL than as ORM session juggling, and the job table is easier to reason about
when you can see exactly which row moves to which status.
"""
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


@contextmanager
def connect():
    """One short-lived connection per unit of work.

    A pool belongs here once there is real traffic; for manual stage triggers
    and a single admin, per-request connections are simpler to follow.
    """
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def query(sql, params=None, one=False):
    with connect() as conn:
        rows = conn.execute(sql, params or ()).fetchall()
    return (rows[0] if rows else None) if one else rows


def execute(sql, params=None, returning=False):
    with connect() as conn:
        cur = conn.execute(sql, params or ())
        result = cur.fetchone() if returning else None
        conn.commit()
    return result


def migrate():
    """Apply any .sql file in migrations/ that has not run yet.

    Numbered SQL files with a tracking table, deliberately instead of Alembic:
    the migrations stay readable as plain SQL, which matters when the schema is
    the thing being discussed.
    """
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        conn.commit()
        applied = {r["filename"] for r in
                   conn.execute("SELECT filename FROM schema_migrations").fetchall()}

        pending = sorted(f for f in MIGRATIONS_DIR.glob("*.sql")
                         if f.name not in applied)
        for path in pending:
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)",
                         (path.name,))
            conn.commit()
            print(f"applied {path.name}")

        if not pending:
            print("schema up to date")
        return [p.name for p in pending]


def healthcheck():
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    migrate()
