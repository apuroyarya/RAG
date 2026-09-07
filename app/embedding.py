"""Embedding backends.

Two backends, for two different jobs:

  sentence_transformers - the real one. Defaults to BAAI/bge-m3, chosen in
      DESIGN.md because it is strong on Bengali *and* covers English and Hindi,
      so phases 2 and 3 need no reindex of the whole corpus.

  fake - deterministic hashed vectors, no model download. This exists to test
      plumbing: that the embed stage writes the right number of vectors of the
      right dimension, that index upserts them, that retrieval returns
      something. It says NOTHING about retrieval quality - hashed vectors have
      no semantic structure, so a search over them is meaningless. Any
      threshold tuned against it would be noise. Never draw a quality
      conclusion from a `fake` run.

Vectors are serialized as raw float32 via the stdlib `array` module, so nothing
in the storage or retrieval path depends on numpy.
"""
import hashlib
from array import array

from .config import EMBED_BACKEND, EMBED_BATCH_SIZE, EMBED_MODEL


def pack(vector):
    """float sequence -> raw float32 bytes."""
    return array("f", vector).tobytes()


def unpack(blob):
    """raw float32 bytes -> list of floats."""
    out = array("f")
    out.frombytes(blob)
    return list(out)


def normalize(vector):
    """L2-normalize so a dot product is cosine similarity.

    Doing this once at embed time means retrieval and reranking can compare with
    a plain dot product, and Qdrant's Cosine distance stays consistent with any
    local scoring we do.
    """
    total = sum(x * x for x in vector) ** 0.5
    if total == 0:
        return list(vector)
    return [x / total for x in vector]


class Embedder:
    name = "base"
    #: vector width; retrieval and the vector-store collection both key on this
    dim = 0

    def embed(self, texts):
        """list[str] -> list[list[float]], L2-normalized, order preserved."""
        raise NotImplementedError

    def available(self):
        raise NotImplementedError

    @property
    def model_id(self):
        """Recorded with every vector, so a model change is detectable.

        Without this a re-embed with a different model would leave a corpus of
        mixed vectors that silently returns nonsense.
        """
        return f"{self.name}:{self.dim}"


class FakeEmbedder(Embedder):
    """Deterministic hashed vectors. For plumbing tests only - see module docs."""
    name = "fake"

    def __init__(self, dim=256):
        self.dim = dim

    def available(self):
        return True, "ok (plumbing only - no semantic meaning)"

    def embed(self, texts):
        vectors = []
        for text in texts:
            # expand a digest to `dim` floats deterministically
            raw = b""
            counter = 0
            while len(raw) < self.dim * 2:
                raw += hashlib.sha256(
                    text.encode("utf-8") + counter.to_bytes(2, "big")).digest()
                counter += 1
            vec = [((raw[i * 2] << 8 | raw[i * 2 + 1]) / 32767.5) - 1.0
                   for i in range(self.dim)]
            vectors.append(normalize(vec))
        return vectors

    @property
    def model_id(self):
        return f"fake:{self.dim}"


class SentenceTransformerEmbedder(Embedder):
    """Real embeddings via sentence-transformers. Model name is configuration.

    BAAI/bge-m3 is ~2.2GB on first use and is downloaded to the HuggingFace
    cache. CPU inference is slow but fine at this corpus size; the batch size is
    configurable because memory, not speed, is what breaks first on CPU.
    """
    name = "sentence_transformers"
    _model = None

    def __init__(self, model_name=None, batch_size=None):
        self.model_name = model_name or EMBED_MODEL
        self.batch_size = batch_size or EMBED_BATCH_SIZE
        self._dim = None

    def available(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False, ("pip install sentence-transformers "
                           "(and torch; use the CPU index to avoid the CUDA "
                           "bundle: --index-url https://download.pytorch.org/whl/cpu)")
        return True, "ok"

    def _load(self):
        if SentenceTransformerEmbedder._model is None:
            from sentence_transformers import SentenceTransformer
            SentenceTransformerEmbedder._model = SentenceTransformer(
                self.model_name, device="cpu")
        return SentenceTransformerEmbedder._model

    @property
    def dim(self):
        if self._dim is None:
            model = self._load()
            # renamed in sentence-transformers 6.x; the old name still works but
            # warns, so prefer the new one when present
            getter = (getattr(model, "get_embedding_dimension", None)
                      or model.get_sentence_embedding_dimension)
            self._dim = getter()
        return self._dim

    def embed(self, texts):
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine == dot product downstream
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    @property
    def model_id(self):
        return f"{self.model_name}:{self.dim}"


BACKENDS = {
    "fake": FakeEmbedder,
    "sentence_transformers": SentenceTransformerEmbedder,
}


def get_embedder(backend=None):
    backend = backend or EMBED_BACKEND
    if backend not in BACKENDS:
        raise KeyError(f"unknown embedding backend {backend!r}; "
                       f"known: {sorted(BACKENDS)}")
    return BACKENDS[backend]()
