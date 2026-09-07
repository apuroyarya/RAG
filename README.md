# RAG knowledge base — Bengali first

A retrieval-augmented question answering system over uploaded PDFs, where every
ingestion stage is visible and individually triggerable, and answers come only
from the indexed corpus — abstaining rather than guessing when the corpus does
not cover the question.

Phase 1 is monolingual Bengali (Bengali documents, Bengali questions, Bengali
answers). English and Hindi follow in phases 2 and 3.

- [DESIGN.md](DESIGN.md) — architecture and the reasoning behind it
- [docs/extraction-findings.md](docs/extraction-findings.md) — why OCR is on the
  critical path even though the PDFs contain digital text

## Current state

| Stage | Status |
|---|---|
| `extract` | implemented — per-page text-layer triage |
| `ocr` | implemented — running Tesseract as a stopgap |
| `normalize` | implemented — NFC + Bengali validity gate |
| `chunk` | implemented — paragraph-first, offsets preserved for citations |
| `embed` | implemented — BGE-M3 self-hosted, or a fake backend for tests |
| `index` | implemented — Qdrant embedded, dense vectors |

All six ingestion stages are implemented. Retrieval, abstention and the query
API are not built yet.

Every page of both sample documents needs OCR — their text layers are corrupt —
so OCR quality sets the ceiling for the whole system. Tesseract is wired up as a
stopgap to get real text flowing; **it is not the final answer**. It reads most
Bengali correctly but substitutes Latin words where it fails (`ও৩ম্` becomes
"Boy", `জাতঃ` becomes "ates"), and flagged 6 of 8 pages on the first sample.
Benchmark the paid engines before committing: see
[tools/ocr_bench/README.md](tools/ocr_bench/README.md).

### OCR setup (Tesseract stopgap)

```bash
winget install UB-Mannheim.TesseractOCR      # or apt install tesseract-ocr
mkdir .tessdata
curl -sSL -o .tessdata/ben.traineddata   https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata
cp "/c/Program Files/Tesseract-OCR/tessdata/"{eng,osd}.traineddata .tessdata/
```

`.tessdata/` beside the project avoids needing admin rights to write into
Program Files. Note `TESSDATA_PREFIX` *replaces* the default directory rather
than adding to it, which is why `eng` and `osd` get copied in alongside `ben`.
The adapter finds both the binary and `.tessdata/` automatically; set
`TESSERACT_CMD` or `TESSDATA_PREFIX` only to override.

Then set `OCR_ENGINE=tesseract` and the `ocr` stage will run.

## Running it

Local development needs no database server - it runs on SQLite out of the box.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Linux/macOS: .venv/bin/pip
python -m app.db                                       # apply migrations
uvicorn app.main:app --reload
```

For deployment, point `DATABASE_URL` at Postgres and run the same migrations -
the SQL is portable, with no dialect branching. `docker compose up -d` brings up
Postgres and Qdrant if you want them locally.

Then `http://localhost:8000/docs` for the API, or check wiring with:

```bash
curl http://localhost:8000/health
```

End-to-end check of the ingestion backbone:

```bash
python scripts/smoke_ingest.py "path/to/some.pdf"
python scripts/test_chunking.py        # chunker logic, no DB or OCR needed
python scripts/test_embedding.py       # embedding layer, fake backend
python scripts/test_vectorstore.py     # vector store, throwaway collection
EMBED_BACKEND=sentence_transformers python scripts/test_embedding.py   # real model
```

### Vector store

Qdrant in **embedded mode** by default — a local directory, no server, no
Docker. Set `QDRANT_URL` to use a server instead.

One caveat that will bite otherwise: embedded mode takes an **exclusive lock**
on its directory, so only one process can hold it. The API server and a CLI
script cannot both open it at once. That is fine for phase 1's manual triggers,
and it is the main reason to move to a server for anything concurrent.

The collection uses a *named* dense vector from the start. Qdrant cannot add a
new named vector to an existing collection, and the design wants hybrid search
later — so naming it now makes adding sparse a collection rebuild rather than a
schema problem. Rebuilds are cheap: the embedding artifacts are the durable
record, so a rebuild re-indexes without re-embedding.

### Embeddings

BGE-M3 self-hosted, chosen because it is strong on Bengali *and* covers English
and Hindi — so phases 2 and 3 need no reindex of the corpus. ~2.2GB on first
use, cached by HuggingFace. CPU inference is slow but fine at this corpus size.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU-only, much smaller
pip install sentence-transformers
```

`EMBED_BACKEND=fake` gives deterministic hashed vectors with no model download.
It exists to test plumbing — that vectors are the right count and width, that
indexing and retrieval wire up. **It says nothing about retrieval quality**:
hashed vectors have no semantic structure, so any threshold tuned against them
is noise.

It asserts the things that are hard to be confident about by reading: stage
ordering is enforced, `ocr` refuses rather than leaving silent holes, re-running
a stage marks downstream stages stale, and unimplemented stages fail clearly
instead of claiming success.

## Ingestion, by hand

Phase 1 drives every stage manually. Upload, then trigger stages one at a time:

```bash
curl -F "file=@document.pdf" http://localhost:8000/documents
curl -X POST http://localhost:8000/documents/<id>/stages/extract/run
curl http://localhost:8000/documents/<id>            # the stage table
curl http://localhost:8000/documents/<id>/pages/2    # what a page actually says
```

A stage that already succeeded needs `?force=true` to re-run, and re-running one
marks every later stage `stale` — their results describe input that no longer
exists.

`POST /documents/<id>/advance` runs stages until one blocks. That is automatic
mode, exposed manually: a background worker would call the same function, which
is why the job table exists before any stage logic.

## Tools

Standalone, no database needed:

```bash
# is a PDF's text layer trustworthy?
python -m tools.encoding_probe "document.pdf"

# which OCR engine should ingestion use?
python -m tools.ocr_bench.run "document.pdf" --pages 2,3,5
```
