"""Database access, working on SQLite or Postgres from one set of SQL.

SQLite is the default so the project runs with nothing installed - no Docker, no
server. Set DATABASE_URL to a postgresql:// URL and the same code and the same
migrations run there instead.

The dialect layer is deliberately thin. It does three things:
  * picks the driver from the URL scheme
  * rewrites %s placeholders to ? for SQLite (queries are written in %s style)
  * returns rows as dicts, with JSON columns already parsed

Everything else is portable SQL living in migrations/. Raw SQL rather than an
ORM because the stage state transitions are the interesting logic here, and they
read better as explicit statements than as session juggling.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DATABASE_URL

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: Columns stored as JSON text. Parsed on read so callers see dicts either way.
JSON_COLUMNS = {"metrics", "quality"}

#: Columns stored as 0/1. Returned as real bools; NULL stays None.
BOOL_COLUMNS = {"review_required", "trustworthy"}

IS_SQLITE = DATABASE_URL.startswith("sqlite")


def new_id():
    """Ids come from Python, not the database - one less dialect difference."""
    return str(uuid.uuid4())


def now_iso():
    """ISO-8601 UTC. Sorts correctly as text, so ORDER BY works on both."""
    return datetime.now(timezone.utc).isoformat()


def to_json(value):
    return json.dumps(value, ensure_ascii=False)


def _sqlite_path():
    # sqlite:///relative.db  or  sqlite:////absolute/path.db
    return DATABASE_URL.split("sqlite://", 1)[1].lstrip("/") or "rag.db"


def _adapt(sql):
    """SQL is written in %s style; SQLite wants ?."""
    return sql.replace("%s", "?") if IS_SQLITE else sql


def _row_to_dict(row, columns):
    out = dict(zip(columns, row))
    for key in JSON_COLUMNS & out.keys():
        if isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    for key in BOOL_COLUMNS & out.keys():
        if out[key] is not None:
            out[key] = bool(out[key])
    return out


class Cursorish:
    """Uniform cursor: .execute returns self, .fetchall gives dicts."""

    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=()):
        self._cur.execute(_adapt(sql), tuple(params))
        return self

    def fetchall(self):
        if self._cur.description is None:
            return []
        cols = [d[0] for d in self._cur.description]
        return [_row_to_dict(r, cols) for r in self._cur.fetchall()]

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None

    @property
    def rowcount(self):
        return self._cur.rowcount


class Connish:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        return Cursorish(self._conn.cursor()).execute(sql, params)

    def executescript(self, sql):
        if IS_SQLITE:
            self._conn.executescript(sql)
        else:
            self._conn.cursor().execute(sql)

    def commit(self):
        self._conn.commit()


@contextmanager
def connect():
    """One short-lived connection per unit of work.

    A pool belongs here once there is real traffic; for manual stage triggers
    and a single admin, per-request connections are simpler to follow.
    """
    if IS_SQLITE:
        path = Path(_sqlite_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield Connish(conn)
        finally:
            conn.close()
    else:
        import psycopg
        with psycopg.connect(DATABASE_URL) as conn:
            yield Connish(conn)


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
                applied_at TEXT NOT NULL
            )
        """)
        conn.commit()
        applied = {r["filename"] for r in
                   conn.execute("SELECT filename FROM schema_migrations").fetchall()}

        pending = sorted(f for f in MIGRATIONS_DIR.glob("*.sql")
                         if f.name not in applied)
        for path in pending:
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (filename, applied_at) "
                         "VALUES (%s, %s)", (path.name, now_iso()))
            conn.commit()
            print(f"applied {path.name}")

        if not pending:
            print("schema up to date")
        return [p.name for p in pending]


def healthcheck():
    try:
        with connect() as conn:
            conn.execute("SELECT 1").fetchall()
        return True, f"ok ({'sqlite' if IS_SQLITE else 'postgres'})"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    migrate()
