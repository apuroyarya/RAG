# Bengali OCR benchmark

Picks the OCR engine the ingestion pipeline will be built on. This is currently
the load-bearing decision for the project: both sample PDFs have unusable text
layers (see `docs/extraction-findings.md`), so OCR quality sets the accuracy
ceiling for the whole RAG system. No amount of good retrieval recovers from a
corrupt corpus.

## The ground-truth problem

You cannot score OCR without correct text to compare against, and the PDFs' own
text layers are wrong — so there is nothing to score against for free. The
harness handles this in two modes:

**Consensus mode (default, free).** Runs every engine, then reports:

- `illegal-initial` — the share of Bengali words beginning with a dependent
  vowel sign or virama. This is orthographically impossible in Bengali, so any
  nonzero value is *proof* of corruption, with no reference text needed. Read
  this column first.
- `agreement` — mean pairwise similarity with the other engines. An engine that
  disagrees with everyone is either much better or much worse; look at the text
  to decide which.
- `chars/pg` — an engine silently dropping half a page passes validity checks
  while being useless. Compare across engines.

**Gold mode (real numbers).** Transcribe 3–5 representative pages by hand and
get actual CER/WER. Consensus tells you *where* engines disagree so you only
transcribe the pages that matter.

Consensus is a proxy, not truth: two engines can agree and both be wrong, most
plausibly on the same rare conjuncts. Confirm with gold before committing.

## Quick start

From the repo root, with the venv active:

```bash
# 1. what is set up? spends nothing
python -m tools.ocr_bench.run "path/to/doc.pdf" --pages 2,3,5 --dry-run

# 2. benchmark whatever is available
python -m tools.ocr_bench.run "path/to/doc.pdf" --pages 2,3,5

# 3. transcribe a few pages, then get real CER/WER
python -m tools.ocr_bench.run "path/to/doc.pdf" --pages 2,3,5 --make-gold-template
python -m tools.ocr_bench.run "path/to/doc.pdf" --pages 2,3,5 --gold
```

Renders and OCR output are cached under `tools/ocr_bench/_out/<pdf-name>/`;
re-running costs nothing. `--force` re-does the work.

Output per run:

| file | what it is |
|---|---|
| `report.md` | the comparison table and ranking |
| `page_NNNN.compare.txt` | every engine's first lines side by side, for eyeballing |
| `<engine>/page_NNNN.txt` | full text per engine per page |
| `render/page_NNNN.png` | the rasterised page all engines saw |
| `raw.json` | everything, for further analysis |

**Keep `--pages` small.** Three to five pages is enough to pick an engine, and
a 354-page book does not need full OCR to make that decision. Pick pages that
represent the corpus — dense body text, a page with a table or verse numbering,
and one that looked bad in the probe.

## Engine setup

All four are optional. Missing ones report why and are skipped, so you can start
with one and add others later.

### google_vision — quickest to get running

```bash
pip install google-cloud-vision
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # PowerShell: $env:GOOGLE_APPLICATION_CREDENTIALS="..."
```

Uses `DOCUMENT_TEXT_DETECTION` with a `bn` language hint. Simpler than Document
AI (no processor to create) and the right first datapoint. **If it wins, retest
with Document AI's layout-aware processor before committing** — that usually
handles multi-column text and tables better, which matters for chunking.

### azure_di

```bash
pip install azure-ai-documentintelligence
export AZURE_DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
export AZURE_DI_KEY=<key>
```

Uses the `prebuilt-read` model. Bengali is supported for printed text.

### surya — self-hosted, no per-page cost

```bash
pip install surya-ocr
```

Strong on Indic scripts. GPU strongly recommended; on CPU expect tens of seconds
per page. Note that Surya's Python API has changed shape across releases — if
the adapter errors, check the API for your installed version before concluding
the engine is bad. The failure message will name the problem.

### tesseract — the pessimistic baseline

```bash
pip install pytesseract
# plus the tesseract binary and the Bengali model:
#   Windows: install from UB-Mannheim's build, select Bengali during setup
#   Linux:   apt install tesseract-ocr tesseract-ocr-ben
```

Included to quantify what the paid engines actually buy you. Expect it to
struggle on conjuncts — but if it does not, you save real money.

## Cost

The harness reports pages processed and seconds per page; it deliberately does
**not** print prices, because published per-1000-page rates change and a stale
number in a repo is worse than none. Take `pages × your current rate` from the
provider's own pricing page.

For scale planning: it is pages, not documents, that you pay for. One 354-page
book is two orders of magnitude more than one 8-page handout.

## Reading the results

1. **Any engine with nonzero `illegal-initial` is producing corrupt Bengali.**
   Disqualifying, regardless of how good the rest of the row looks.
2. Compare `chars/pg` across engines. A low outlier is dropping content.
3. Check the lowest-agreement page in `page_NNNN.compare.txt` by eye.
4. Transcribe that page, re-run with `--gold`, and rank on CER.
5. `textlayer` is in the table as the baseline you are replacing. On the known
   samples it shows ~10% illegal-initial. Any engine you pick must be dramatically
   better than it — for calibration, the corrupt text layer scores roughly 67% WER
   against correct text.

A note on gold files: transcribe the **whole** page or leave it out. A partial
reference makes CER explode past 1.0 through counted insertions, which is
meaningless rather than merely pessimistic. The report warns when it sees this.
