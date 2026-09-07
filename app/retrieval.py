"""Retrieval: embed the question, search, rerank.

Returns candidates carrying both scores plus everything needed to cite them,
and reports whether the scores are thresholdable. The answering layer needs
that flag: gating on an uncalibrated score would produce an abstention decision
that looks principled and is not.

Retrieval deliberately does not decide anything. It returns ranked candidates
and their scores; the gate lives in answering.py, so the threshold logic is in
one place and can be reasoned about on its own.
"""
from dataclasses import dataclass, field

from .config import RETRIEVE_DENSE_K, RETRIEVE_RERANK_K
from .embedding import get_embedder
from .reranking import get_reranker
from . import vectorstore as vs


@dataclass
class Candidate:
    chunk_id: str
    document_id: str
    filename: str
    page_no: int
    ordinal: int
    char_start: int
    char_end: int
    text: str
    dense_score: float
    rerank_score: float | None = None
    #: the score the gate should read - rerank when available, else dense
    gate_score: float = 0.0

    def citation(self):
        """Where this text came from, precisely enough to check by hand."""
        return {
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "page_no": self.page_no,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class RetrievalResult:
    question: str
    candidates: list = field(default_factory=list)
    embed_model: str = ""
    rerank_backend: str = ""
    #: False when the gate would be reading embedding cosine rather than a
    #: cross-encoder score. Callers must surface this, not silently threshold.
    scores_thresholdable: bool = False
    diagnostics: dict = field(default_factory=dict)

    @property
    def best_score(self):
        return self.candidates[0].gate_score if self.candidates else None


def _check_collection_matches(embedder):
    """Fail clearly when the collection was built by a different model.

    Without this the query vector reaches Qdrant at the wrong width and the
    failure surfaces as `ValueError: shapes (24,1024) and (256,) not aligned`
    from inside numpy - which says nothing about the actual mistake, namely
    that EMBED_BACKEND changed without re-running embed and index.
    """
    from . import db
    from .config import QDRANT_COLLECTION

    tracked = db.query("SELECT * FROM vector_collections WHERE name = %s",
                       (QDRANT_COLLECTION,), one=True)
    if not tracked:
        raise RuntimeError(
            f"collection {QDRANT_COLLECTION!r} has never been indexed. "
            f"Run the embed and index stages on at least one document.")

    if tracked["model"] != embedder.model_id:
        raise RuntimeError(
            f"collection {QDRANT_COLLECTION!r} holds vectors from "
            f"{tracked['model']!r} but the configured embedder is "
            f"{embedder.model_id!r}. Vectors from two models are not "
            f"comparable - re-run the embed and index stages for every "
            f"document, or switch EMBED_BACKEND/EMBED_MODEL back.")


def retrieve(question, dense_k=None, rerank_k=None, document_id=None):
    """Embed, search, rerank. Returns a RetrievalResult, decides nothing."""
    dense_k = dense_k or RETRIEVE_DENSE_K
    rerank_k = rerank_k or RETRIEVE_RERANK_K

    embedder = get_embedder()
    ok, why = embedder.available()
    if not ok:
        raise RuntimeError(f"embedding backend unavailable: {why}")

    reranker = get_reranker()
    ok, why = reranker.available()
    if not ok:
        raise RuntimeError(f"reranker unavailable: {why}")

    _check_collection_matches(embedder)

    query_vector = embedder.embed([question])[0]

    client = vs.connect()
    try:
        hits = vs.search(client, query_vector, limit=dense_k,
                         document_id=document_id)
    finally:
        if hasattr(client, "close"):
            client.close()

    candidates = []
    for score, payload in hits:
        # a point written by a different embedding model is not comparable;
        # mixing them is the silent-nonsense failure the index stage guards
        if payload.get("model") and payload["model"] != embedder.model_id:
            continue
        candidates.append(Candidate(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            filename=payload.get("filename", ""),
            page_no=payload["page_no"],
            ordinal=payload["ordinal"],
            char_start=payload["char_start"],
            char_end=payload["char_end"],
            text=payload["text"],
            dense_score=score,
            gate_score=score,
        ))

    skipped = len(hits) - len(candidates)

    scores = reranker.rerank(question, [c.text for c in candidates])
    for cand, score in zip(candidates, scores):
        cand.rerank_score = score
        if score is not None:
            cand.gate_score = score

    candidates.sort(key=lambda c: c.gate_score, reverse=True)
    candidates = candidates[:rerank_k]

    return RetrievalResult(
        question=question,
        candidates=candidates,
        embed_model=embedder.model_id,
        rerank_backend=reranker.name,
        scores_thresholdable=reranker.thresholdable,
        diagnostics={
            "dense_k": dense_k,
            "rerank_k": rerank_k,
            "dense_hits": len(hits),
            "skipped_wrong_model": skipped,
            "returned": len(candidates),
            "score_range": (
                [round(candidates[-1].gate_score, 4),
                 round(candidates[0].gate_score, 4)] if candidates else None),
        },
    )
