-- Retrievable chunks, with the offsets that make citations verifiable.
--
-- `id` is deterministic - "<document_id>:<page_no>:<ordinal>" - rather than a
-- random uuid. Two reasons:
--   * re-running the chunk stage upserts in place instead of duplicating
--   * the vector store keys on the same id, so re-indexing overwrites the right
--     point rather than leaving an orphan behind
--
-- char_start/char_end index into document_pages.text for the same page. The
-- abstention design requires the model to cite spans, and a citation is only
-- verifiable if it resolves back to exact characters - these are that anchor.
--
-- `text` is the chunk's own extent; `embed_text` additionally carries the
-- lead-in overlap from the previous chunk. They differ on purpose: overlap
-- helps retrieval find a sentence split across a boundary, but a citation must
-- point at the chunk's real span, not at text borrowed from its neighbour.

CREATE TABLE chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INTEGER NOT NULL,
    ordinal     INTEGER NOT NULL,        -- position within the page, 0-based
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    embed_text  TEXT    NOT NULL,
    n_chars     INTEGER NOT NULL,
    -- filled in once an embedding model is chosen and its tokenizer is known;
    -- NULL is honest until then rather than a guessed English-derived number
    n_tokens    INTEGER,
    created_at  TEXT    NOT NULL,
    UNIQUE (document_id, page_no, ordinal)
);

CREATE INDEX chunks_by_document ON chunks (document_id, page_no, ordinal);
