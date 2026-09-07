# RAG Knowledge Base — Design (post-grilling)

Repo: https://github.com/apuroyarya/RAG

## Scope by phase

| Phase | Corpus | Query | Answer |
|-------|--------|-------|--------|
| 1 | Bengali | Bengali | Bengali |
| 2 | + English | English | English |
| 3 | + Hindi | Hindi | Hindi |

Phase 1 is **monolingual**. Cross-lingual retrieval (English question over Bengali
docs) is explicitly out of scope until phase 2+, and is what makes abstention
thresholds hard. Deferring it is the single biggest simplification in this design.

Non-goals: fine-tuning (RAG does not train a model), multi-tenancy, cross-lingual
answers in phase 1.

## Established constraints

- Source PDFs are digital text (not scans), **but their text layers are wrong**.
  Verified on two samples: see `docs/extraction-findings.md`. Embedded Bengali
  fonts lack `/ToUnicode` cmaps, so extraction produces Bengali-looking but
  systematically incorrect text. Pages render correctly; only the text mapping is
  broken. → **render + OCR is the primary extraction path**, text layer is the
  optimisation, not the default.
- Cloud APIs are permitted.
- Target scale: thousands of documents (100k+ pages).
- Latency budget: **under 5s** per question → no two serial LLM calls.
- Every stage inspectable; manual triggers in phase 1, auto-advance later.

## Pipeline

    upload → extract → normalize → chunk → embed → index

`extract` triages the text layer and then takes one of two paths — text-layer
extraction or render+OCR. See below; this is the highest-risk stage in the
system, and on the evidence so far most documents will need OCR.

### extract is the stage that can silently ruin everything

Measured on both sample documents, the text layer is unusable and a naive
codepoint-ratio check passes it anyway (92% and 96% "Bengali", both wrong). So
extraction is a **triage** followed by one of two paths:

**Triage (per document, all three visible as stage output):**

1. **Structural** (free) — any font declaring no `/ToUnicode`, or whose name
   matches the legacy family pattern (`*MJ`, Bijoy, Boishakhi), condemns the
   text layer. Caught both samples. Note both samples mix good and bad fonts
   *within one document*, so this is per-font-run, not per-document.
2. **Orthographic** (free) — a Bengali word cannot begin with a dependent vowel
   sign (U+09BE–U+09CC) or a virama. Any occurrence is corruption. Caught both
   samples (9.27% and 0.66% of tokens).
3. **OCR cross-check** (costs money; ground truth) — render a sample of pages,
   OCR them, compare against the text layer at character level. Agreement above
   threshold promotes the document to the cheap text-layer path for all pages;
   disagreement sends the whole document to OCR.

**Path A — text layer** (only if all three pass): PyMuPDF extraction + NFC
normalization. Cheap, exact.

**Path B — render + OCR** (expected to be the majority): render at ~300 dpi,
Bengali OCR, then the same normalization. Slower and metered, but it bypasses
the broken cmap entirely because the glyphs on the page are correct.

Per-font correction tables were considered and rejected: the mapping differs per
font subset and sample file 2 alone has 11 bad subsets.

Apply NFC normalization on both paths — extractors emit Indic conjuncts in visual
rather than logical order, so `ক্ষ` otherwise won't match how a user types it.

Corpus corruption is the worst failure mode available to this project: it embeds
cleanly, retrieves cleanly, and produces confident wrong answers that the
abstention gate structurally cannot detect, because the bad text *is* the corpus.
This is why the triage output is a visible, per-document pass/fail with a human
review queue rather than a silent "extraction: success".

## Retrieval

- **Embeddings: BGE-M3.** Strong on Bengali, and covers English + Hindi, so
  phases 2–3 need no reindex. A Bengali-only embedder would force re-embedding
  thousands of documents later.
- **Hybrid: dense + sparse.** Same-language lexical matching genuinely helps in
  phase 1. Note Postgres has no Bengali FTS config (`to_tsvector('bengali', …)`
  does not exist) — use Qdrant sparse vectors or a custom analyzer.
- **Rerank: BGE-reranker-v2-m3** over the top ~50 candidates.
- Chunking is structure-aware (headings, font sizes from the PDF layout), not
  fixed token windows. Bengali tokenizes 3–4x heavier than English, so size
  chunks by characters or with a tokenizer-aware splitter — do not copy an
  English 512-token default.

Deferred behind an interface for phase 2+: query translation, RRF fusion of
translated and untranslated retrieval paths.

## Abstention and grounding

The requirement "answer only from the knowledge, otherwise say I don't have
enough information" is a subsystem, not a prompt.

1. **Gate before generation.** If the top reranker score is below a calibrated
   threshold, abstain without calling the LLM. Gate on the *reranker* score, not
   embedding cosine distance — raw cosine similarity is not reliably calibrated
   for this.
2. **Structural citations.** The generation call must emit span IDs with every
   claim. Answers whose citations don't resolve are rejected.
3. **Cheap verification in the hot path.** Confirm each cited span exists and that
   the claim overlaps it (lexical + embedding overlap). No second LLM call — the
   5s budget doesn't allow one. A full LLM verifier is available for offline
   evaluation only.

Thresholds are set from the eval set, not by intuition.

## Stage control and job model

Build the **job table first**. Every document has one row per stage: status, input
reference, output artifact, timestamps, error. Manual mode inserts a job and runs
it now; auto mode has a worker poll for pending jobs. Same core, same artifacts,
same inspection UI — only the trigger differs.

Building manual mode as buttons that call pipeline functions directly makes auto
mode a rewrite. This is worth roughly one extra day up front.

Persist extracted and normalized text **separately from embeddings**. The
embedding model will change at least once; that must be a re-embed, not a
re-extract.

## Evaluation — definition of done for phase 1

100 Bengali questions, authored **before** any threshold tuning, split dev/test
50/50. Do not look at the test half while tuning.

- 60 answerable, each with a known gold span
- 40 deliberately unanswerable

Phase 1 ships when:

- ≥85% of answerable questions get a correct, correctly-cited answer
- ≥90% of unanswerable questions produce the abstention message
- **zero** fabricated citations — a citation that doesn't support its claim is
  worse than a wrong answer, because it looks trustworthy

Without this set, "it doesn't hallucinate" is unverifiable and phase 1 has no end.

## Stack

- Python / FastAPI
- Postgres — document metadata, job/stage table, artifacts
- Qdrant — dense + sparse vectors
- Next.js — admin pipeline UI and chat UI
- Generation: Claude Sonnet 5 (strong Bengali, fits the latency budget)

## Open items

- [x] Encoding probe on real sample PDFs — done, see `docs/extraction-findings.md`.
      Both samples' text layers are unusable; root cause is missing `/ToUnicode`
      cmaps, not legacy ASCII encoding.
- [ ] Choose and validate a Bengali OCR engine against the two known-bad samples
      **before** building the OCR path
- [ ] Run the probe over ~20 more real documents to get the trustworthy/untrustworthy
      split — this sets the OCR budget and the ingestion time estimate
- [ ] Verify whether file 2's doubled letters are two interleaved text runs; if so,
      deduplication may recover some pages without OCR
- [ ] Schedule the eval-set authoring session (realistically a full day)
- [ ] Auth model and who the end users are
- [ ] Hosting target
