# Setup on Ubuntu (with Docker)

Everything so far was built and verified on Windows without Docker. On Ubuntu
with Docker available, three constraints that shaped earlier decisions go away.
Nothing needs rewriting — they were deliberately kept behind configuration.

## What gets better, and why

**Postgres instead of SQLite — one environment variable.** The schema is
portable SQL with no dialect branching, precisely so this would be a config
change rather than a migration:

```bash
export DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
python -m app.db          # same migrations, now against Postgres
```

**Qdrant as a server instead of embedded.** Embedded mode takes an exclusive
lock on its directory, so the API server and a CLI script cannot both hold it.
With Docker, that limitation disappears:

```bash
export QDRANT_URL=http://localhost:6333
```

**HuggingFace symlinks work.** On Windows without developer mode, HF stores a
full copy of every file instead of deduplicating — which is how a 2.2GB model
occupied 4.3GB and eventually ran the disk out of space mid-download. On Linux
this is simply not a problem.

## Full setup

```bash
git clone https://github.com/apuroyarya/RAG.git
cd RAG

python3 -m venv .venv
source .venv/bin/activate           # note: bin/, not Scripts/
pip install -r requirements.txt

docker compose up -d                # Postgres + Qdrant

cp .env.example .env                # then edit per below
python -m app.db                    # apply migrations
```

Set these in `.env`:

```bash
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
QDRANT_URL=http://localhost:6333
OCR_ENGINE=tesseract
EMBED_BACKEND=sentence_transformers
RERANK_BACKEND=cross_encoder
```

### OCR

Much simpler than on Windows — the Bengali language pack is a package, so no
manual `.tessdata` directory and no `TESSDATA_PREFIX`:

```bash
sudo apt install tesseract-ocr tesseract-ocr-ben
tesseract --list-langs              # expect: ben, eng, osd
```

The adapter probes `/usr/bin/tesseract` among its candidates, so it needs no
configuration. Set `TESSERACT_CMD` only to override.

For better quality than the distro model, drop `ben.traineddata` from
`tesseract-ocr/tessdata_best` into a directory and point `TESSDATA_PREFIX` at
it — remembering that `TESSDATA_PREFIX` *replaces* the default directory, so
`eng` and `osd` must be copied in alongside `ben`.

### Models

Two downloads, ~4.5GB total, cached under `~/.cache/huggingface`:

- `BAAI/bge-m3` (~2.2GB) — embeddings, already verified working
- `BAAI/bge-reranker-v2-m3` (~2.3GB) — the reranker, **never successfully
  downloaded on Windows**; this is the one thing still unexercised

If disk or bandwidth is tight, set `HF_HOME` to a roomier location first.

## Verify the port

Run these in order. All but the last need no model downloads and no API key:

```bash
python scripts/test_chunking.py        # 23 checks
python scripts/test_embedding.py       # fake backend
python scripts/test_vectorstore.py     # throwaway collection
python scripts/test_answering.py       # 26 abstention checks
python scripts/smoke_ingest.py path/to/some.pdf
```

Then ingest for real and ask something:

```bash
uvicorn app.main:app --reload
curl http://localhost:8000/health      # confirms which backends are live
```

## The one thing that has never run

**The cross-encoder reranker.** It failed to download on Windows purely for want
of disk space, so `bge-reranker-v2-m3` has never scored a single passage in this
project. That matters more than it sounds:

- the abstention gate reads the reranker's score, so its behaviour is the whole
  basis of the "I don't have enough information" decision
- `ABSTAIN_THRESHOLD` is a placeholder and cannot be set until real scores exist

First thing worth running on Ubuntu:

```bash
RERANK_BACKEND=cross_encoder python scripts/sweep_threshold.py eval/questions.example.json
```

That prints, per candidate cutoff, how many answerable questions would still be
answered against how many unanswerable ones would leak through. Read the
`leaked` column first. Then replace the example file with your real 100-question
set — see [../eval/README.md](../eval/README.md), particularly the part about
making most of the unanswerable questions *topically adjacent* to the corpus
rather than obviously off-topic.
