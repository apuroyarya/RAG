"""Runtime configuration, all from the environment. See .env.example."""
import os
from pathlib import Path

#: SQLite by default so the project runs with nothing installed. Point this at
#: a postgresql:// URL for deployment; the same portable SQL runs on both.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///rag.db")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

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
