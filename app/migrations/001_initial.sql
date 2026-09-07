-- Ingestion backbone.
--
-- Two ideas carry this schema:
--
-- 1. stage_runs is the job table. Every document has one row per pipeline stage,
--    carrying its status, its output artifact and its metrics. Manual mode
--    inserts a run and executes it now; auto mode has a worker claim pending
--    runs. Same table, same artifacts, same inspection surface - only the
--    trigger differs. Building manual mode as direct function calls would make
--    auto mode a rewrite, which is why this exists before any stage logic.
--
-- 2. document_pages holds extracted text separately from any embedding. The
--    embedding model will change at least once, and at thousands of Bengali
--    pages that has to be a re-embed, not a re-extract.
--
-- Deliberately portable SQL: runs on SQLite (local dev, zero dependencies) and
-- Postgres (deployment) with no dialect branching. That costs three things,
-- each a conscious trade:
--   * TEXT + CHECK instead of Postgres ENUM types
--   * TEXT holding JSON instead of JSONB - the app parses it
--   * TEXT holding ISO-8601 UTC instead of TIMESTAMPTZ; such strings sort
--     correctly as text, so ORDER BY still works
-- Ids are generated in Python rather than by the database, for the same reason.

CREATE TABLE documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT    NOT NULL,
    sha256          TEXT    NOT NULL UNIQUE,   -- re-uploading the same PDF is a no-op
    storage_path    TEXT    NOT NULL,
    byte_size       INTEGER NOT NULL,
    page_count      INTEGER,
    language        TEXT    NOT NULL DEFAULT 'bn',
    -- set by normalize when pages fail the Bengali validity gate; these must be
    -- looked at by a human before the document can be indexed
    review_required INTEGER NOT NULL DEFAULT 0 CHECK (review_required IN (0, 1)),
    review_note     TEXT,
    uploaded_at     TEXT    NOT NULL
);

CREATE TABLE stage_runs (
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL CHECK (stage IN
                     ('extract', 'ocr', 'normalize', 'chunk', 'embed', 'index')),
    -- held:  finished, but a human must look before it advances (review gate)
    -- stale: an upstream stage re-ran, so this result describes input that is gone
    status       TEXT NOT NULL DEFAULT 'pending' CHECK (status IN
                     ('pending', 'running', 'succeeded', 'failed',
                      'skipped', 'held', 'stale')),
    attempt      INTEGER NOT NULL DEFAULT 0,
    triggered_by TEXT NOT NULL DEFAULT 'manual',  -- 'manual' | 'auto'
    output_ref   TEXT,          -- artifact path on disk, when the stage writes one
    metrics      TEXT NOT NULL DEFAULT '{}',      -- JSON
    error        TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    created_at   TEXT NOT NULL,
    -- one row per stage per document; re-running updates in place, bumps attempt
    PRIMARY KEY (document_id, stage)
);

CREATE INDEX stage_runs_claimable ON stage_runs (status, stage)
    WHERE status = 'pending';

-- Page-level text with provenance. `source` records HOW the text was obtained,
-- because the two paths have very different trust levels:
--   textlayer - the PDF's own text, used only where triage passed
--   ocr       - rendered and OCR'd, used where the text layer was untrustworthy
--   needs_ocr - triage rejected the text layer; no text yet
--   empty     - image-only page; OCR is the only option
CREATE TABLE document_pages (
    document_id TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INTEGER NOT NULL,              -- 1-based
    source      TEXT    NOT NULL CHECK (source IN
                    ('textlayer', 'ocr', 'needs_ocr', 'empty')),
    ocr_engine  TEXT,                          -- which engine, when source='ocr'
    raw_text    TEXT,                          -- as obtained, before normalization
    text        TEXT,                          -- NFC-normalized; what downstream uses
    -- triage + validity signals, so a bad page is visible without re-deriving it
    quality     TEXT    NOT NULL DEFAULT '{}',  -- JSON
    trustworthy INTEGER CHECK (trustworthy IN (0, 1)),  -- NULL until normalize judges
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (document_id, page_no)
);

CREATE INDEX document_pages_untrustworthy ON document_pages (document_id)
    WHERE trustworthy = 0;
