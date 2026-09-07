-- Which embedding model the vector collection currently holds.
--
-- Qdrant has no place to record this, and it is the one thing that must not be
-- got wrong: vectors from two different models in one collection do not error.
-- They return confident nonsense, and nothing downstream can tell. The
-- abstention gate cannot save us here either, because the scores look normal.
--
-- So the index stage compares the embedding artifact's model against this row.
-- On a mismatch it recreates the collection and marks every *other* document's
-- index stage 'stale', which makes the required re-index visible in the admin
-- stage table instead of leaving a silently mixed corpus. That is affordable
-- precisely because the embedding artifacts are the durable record - a rebuild
-- re-indexes, it does not re-embed.

CREATE TABLE vector_collections (
    name       TEXT PRIMARY KEY,
    model      TEXT    NOT NULL,     -- "<model name>:<dim>", from the manifest
    dim        INTEGER NOT NULL,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);
