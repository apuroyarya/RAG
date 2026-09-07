-- Embedding metadata. The vectors themselves live in a per-document artifact
-- file, not in here.
--
-- Why not store vectors in the database: at thousands of documents this is
-- roughly 4KB per chunk (1024 float32s), so a 400k-chunk corpus is ~1.6GB of
-- blobs. They are large, write-once, derived data - exactly what the existing
-- stage-artifact mechanism is for. Keeping them out also avoids the one real
-- portability trap in this schema, since SQLite spells the type BLOB and
-- Postgres spells it BYTEA.
--
-- The artifact is the durable record and the vector store is a *rebuildable
-- projection* of it. That matters because the design expects the embedding
-- model to change at least once: re-indexing then costs nothing, and only a
-- genuine model change costs a re-embed.
--
-- `model` is recorded so a mixed-model corpus is detectable. Vectors from two
-- different models in one collection do not error - they silently return
-- nonsense, which is the failure mode this project can least afford.

CREATE TABLE document_embeddings (
    document_id  TEXT    PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    model        TEXT    NOT NULL,        -- "<model name>:<dim>"
    dim          INTEGER NOT NULL,
    n_chunks     INTEGER NOT NULL,
    vectors_path TEXT    NOT NULL,        -- raw float32, n_chunks x dim, row-major
    manifest_path TEXT   NOT NULL,        -- chunk ids, in the row order above
    created_at   TEXT    NOT NULL
);
