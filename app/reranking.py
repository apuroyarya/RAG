"""Rerankers.

The reranker is not an optional quality boost here - it is what the abstention
gate reads. DESIGN.md gates on the rerank score specifically because raw
embedding cosine similarity is not reliably calibrated for "is this good enough
to answer from": cosine scores are compressed into a narrow band, and where that
band sits shifts with query length and phrasing. A cross-encoder scores the
query and passage together and separates relevant from irrelevant far more
sharply, which is what makes a threshold meaningful at all.

Two backends:

  cross_encoder - BAAI/bge-reranker-v2-m3, the real one. Multilingual, strong on
      Bengali, and the natural pairing with BGE-M3 embeddings.

  passthrough - keeps the dense order and reports the dense score as the rerank
      score. For plumbing only. Gating on it means gating on embedding cosine,
      which is the thing the design explicitly rejects - so any threshold tuned
      against it is meaningless, and the answering layer says so in its output.
"""
from .config import RERANK_BACKEND, RERANK_MODEL


class Reranker:
    name = "base"
    #: True when scores are calibrated enough to threshold against
    thresholdable = False

    def available(self):
        raise NotImplementedError

    def rerank(self, query, passages):
        """passages: list[str] -> list[float] scores, same order as input."""
        raise NotImplementedError


class PassthroughReranker(Reranker):
    """Keeps dense order. Plumbing only - see module docstring."""
    name = "passthrough"
    thresholdable = False

    def available(self):
        return True, "ok (plumbing only - scores are embedding cosine, not calibrated)"

    def rerank(self, query, passages):
        # the caller substitutes dense scores; returning None marks "no opinion"
        return [None] * len(passages)


class CrossEncoderReranker(Reranker):
    """BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder.

    Scores are logits, not probabilities: they are unbounded and centred near
    zero, so a threshold is a raw score cutoff rather than anything like a
    percentage. That cutoff still has to be measured against the eval set - the
    default in config is a placeholder, not a calibrated value.
    """
    name = "cross_encoder"
    thresholdable = True
    _model = None

    def __init__(self, model_name=None):
        self.model_name = model_name or RERANK_MODEL

    def available(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False, "pip install sentence-transformers"
        return True, "ok"

    def _load(self):
        if CrossEncoderReranker._model is None:
            from sentence_transformers import CrossEncoder
            CrossEncoderReranker._model = CrossEncoder(self.model_name, device="cpu")
        return CrossEncoderReranker._model

    def rerank(self, query, passages):
        if not passages:
            return []
        model = self._load()
        scores = model.predict([(query, p) for p in passages])
        return [float(s) for s in scores]


BACKENDS = {
    "passthrough": PassthroughReranker,
    "cross_encoder": CrossEncoderReranker,
}


def get_reranker(backend=None):
    backend = backend or RERANK_BACKEND
    if backend not in BACKENDS:
        raise KeyError(f"unknown reranker backend {backend!r}; "
                       f"known: {sorted(BACKENDS)}")
    return BACKENDS[backend]()
