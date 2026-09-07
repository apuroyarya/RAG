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

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename      TEXT        NOT NULL,
    sha256        TEXT        NOT NULL UNIQUE,  -- re-uploading the same PDF is a no-op
    storage_path  TEXT        NOT NULL,
    byte_size     BIGINT      NOT NULL,
    page_count    INTEGER,
    language      TEXT        NOT NULL DEFAULT 'bn',
    -- set by normalize when pages fail the Bengali validity gate; these must be
    -- looked at by a human before the document can be indexed
    review_required BOOLEAN   NOT NULL DEFAULT FALSE,
    review_note   TEXT,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TYPE stage_name AS ENUM
    ('extract', 'ocr', 'normalize', 'chunk', 'embed', 'index');

-- held: deliberately parked so a human can inspect before it advances.
-- stale: an upstream stage re-ran, so this result no longer describes the input.
CREATE TYPE stage_status AS ENUM
    ('pending', 'running', 'succeeded', 'failed', 'skipped', 'held', 'stale');

CREATE TABLE stage_runs (
    id           BIGSERIAL PRIMARY KEY,
    document_id  UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stage        stage_name  NOT NULL,
    status       stage_status NOT NULL DEFAULT 'pending',
    attempt      INTEGER     NOT NULL DEFAULT 0,
    triggered_by TEXT        NOT NULL DEFAULT 'manual',   -- 'manual' | 'auto'
    output_ref   TEXT,        -- artifact path on disk, when the stage writes one
    metrics      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    error        TEXT,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- one row per stage per document; re-running updates in place and bumps attempt
    UNIQUE (document_id, stage)
);

CREATE INDEX stage_runs_claimable ON stage_runs (status, stage)
    WHERE status = 'pending';

-- Page-level text with provenance. `source` records HOW the text was obtained,
-- which matters because the two paths have very different trust levels:
--   textlayer - the PDF's own text, only used where triage passed
--   ocr       - rendered and OCR'd, used where the text layer was untrustworthy
CREATE TYPE page_source AS ENUM ('textlayer', 'ocr', 'needs_ocr', 'empty');

CREATE TABLE document_pages (
    document_id  UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      INTEGER     NOT NULL,          -- 1-based
    source       page_source NOT NULL,
    ocr_engine   TEXT,                          -- which engine, when source='ocr'
    raw_text     TEXT,                          -- as obtained, before normalization
    text         TEXT,                          -- NFC-normalized, what downstream uses
    -- triage + validity signals, so a bad page is visible without re-deriving it
    quality      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    trustworthy  BOOLEAN,                       -- NULL until normalize has judged
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, page_no)
);

CREATE INDEX document_pages_untrustworthy ON document_pages (document_id)
    WHERE trustworthy IS FALSE;
