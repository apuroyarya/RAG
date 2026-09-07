"""Runtime configuration, all from the environment. See .env.example."""
import os
from pathlib import Path

#: SQLite by default so the project runs with nothing installed. Point this at
#: a postgresql:// URL for deployment; the same portable SQL runs on both.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rag.db")

#: Empty means embedded mode: a local directory, no server, no Docker. Note
#: embedded mode takes an EXCLUSIVE lock on that directory, so only one process
#: can hold it - set QDRANT_URL to a server for anything concurrent.
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_PATH = Path(os.environ.get("QDRANT_PATH", "qdrant_data")).resolve()
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "chunks")

#: Where stage artifacts and uploaded PDFs live. Artifacts are files on disk
#: referenced by path from stage_runs.output_ref, not blobs in the database -
#: they are large, write-once, and useful to open directly while debugging.
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "storage")).resolve()

#: Which OCR adapter the `ocr` stage uses. Empty until the benchmark in
#: tools/ocr_bench picks a winner; the stage reports "no engine configured"
#: rather than guessing.
OCR_ENGINE = os.environ.get("OCR_ENGINE", "").strip()

#: Render resolution for the OCR path. 300 is the usual sweet spot.
OCR_DPI = int(os.environ.get("OCR_DPI", "300"))

#: Phase 1 is monolingual Bengali. Phases 2 and 3 add English then Hindi.
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "bn")

#: Embedding backend: "sentence_transformers" (real) or "fake" (deterministic
#: hashed vectors for plumbing tests - no semantic meaning, never tune against).
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "sentence_transformers")

#: BGE-M3 covers Bengali strongly plus English and Hindi, so phases 2-3 need no
#: reindex. ~2.2GB on first use, cached by HuggingFace.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

#: On CPU, memory is what breaks first rather than speed.
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "8"))

#: Chunk sizing, in CHARACTERS not tokens. Bengali tokenizes far heavier than
#: English, so an English token default would be wrong - but the true ratio
#: depends on the embedding model's tokenizer, which is not chosen yet.
#: Defaults measured against the sample corpus: pages average ~1000 chars and
#: hold several independent mantras, so ~400 gives 3-4 chunks per page. The
#: retrieval-optimal value needs the eval set to determine.
CHUNK_TARGET_CHARS = int(os.environ.get("CHUNK_TARGET_CHARS", "400"))
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "700"))
CHUNK_MIN_CHARS = int(os.environ.get("CHUNK_MIN_CHARS", "80"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "60"))

#: A page whose token-initial-dependent-sign rate exceeds this is corrupt.
#: Bengali words cannot begin with a dependent vowel sign, so the honest
#: threshold is 0 - this allows a hair of slack for stray artefacts.
VALIDITY_THRESHOLD = float(os.environ.get("VALIDITY_THRESHOLD", "0.005"))


def document_dir(document_id):
    return STORAGE_DIR / str(document_id)


def stage_dir(document_id, stage):
    d = document_dir(document_id) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Retrieval and abstention
# ---------------------------------------------------------------------------

#: Dense candidates pulled before reranking. Wider is better for recall since
#: the reranker is what decides the final order.
RETRIEVE_DENSE_K = int(os.environ.get("RETRIEVE_DENSE_K", "30"))

#: Candidates kept after reranking.
RETRIEVE_RERANK_K = int(os.environ.get("RETRIEVE_RERANK_K", "8"))

#: Spans actually shown to the model. Fewer, better spans beat more, weaker
#: ones - extra weak context is what invites a stretched answer.
MAX_CONTEXT_SPANS = int(os.environ.get("MAX_CONTEXT_SPANS", "5"))

#: Reranker: "cross_encoder" (real) or "passthrough" (plumbing only - gates on
#: embedding cosine, which the design explicitly rejects as uncalibrated).
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "cross_encoder")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

#: THIS IS A PLACEHOLDER, NOT A CALIBRATED VALUE.
#: The abstention gate compares the best cross-encoder score against this.
#: bge-reranker-v2-m3 emits unbounded logits centred near zero, so 0.0 is
#: roughly "the reranker is ambivalent". The real value must be measured on the
#: eval set - 60 answerable questions with known gold spans and 40 deliberately
#: unanswerable ones - by picking the cutoff that maximises correct abstention
#: without losing answerable questions. Until then, treat every abstention
#: decision as unvalidated.
ABSTAIN_THRESHOLD = float(os.environ.get("ABSTAIN_THRESHOLD", "0.0"))

#: Minimum share of a quote's tokens that must appear in the span it cites.
#: Not a semantic check - it catches a fabricated or paraphrased quote, not a
#: claim that misreads a real quote. Also a placeholder pending the eval set.
GROUNDING_MIN_OVERLAP = float(os.environ.get("GROUNDING_MIN_OVERLAP", "0.6"))

#: Returned verbatim whenever the system abstains. Phase 1 is Bengali.
#: "I do not have enough information to answer this question."
ABSTAIN_MESSAGE = os.environ.get(
    "ABSTAIN_MESSAGE",
    "এই প্রশ্নের উত্তর দেওয়ার জন্য আমার কাছে পর্যাপ্ত তথ্য নেই।")

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-opus-5")
