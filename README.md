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
| `ocr` | implemented — engine pluggable, **engine not yet chosen** |
| `normalize` | implemented — NFC + Bengali validity gate |
| `chunk` | not implemented (blocked on real extracted text) |
| `embed` | not implemented (blocked on BGE-M3 hosting decision) |
| `index` | not implemented (blocked on hybrid-search decision) |

Retrieval, abstention and the query API are not built yet.

The open decision blocking the rest is **which OCR engine** — see
[tools/ocr_bench/README.md](tools/ocr_bench/README.md).

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
```

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
