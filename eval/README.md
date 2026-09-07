# Evaluation set

`ABSTAIN_THRESHOLD` cannot be set without this. Until it exists, every
abstention decision the system makes is unvalidated.

## What is needed

100 Bengali questions, authored **before** any threshold tuning:

- **60 answerable**, each with the page and span that answers it
- **40 unanswerable** — and this is the half that does the work

Split dev/test 50/50 and do not look at the test half while tuning, or the
threshold is fitted to noise.

## The unanswerable half is where the difficulty is

`questions.example.json` shows the shape, and deliberately includes both kinds:

**Easy rejections** — `ক্রিকেট খেলার নিয়ম কী?`, `ভারতের রাজধানী কোথায়?`
Far outside the corpus. A cross-encoder should reject these trivially. They
prove the plumbing works and little else.

**Hard rejections** — `দেবযজ্ঞের জন্য কত টাকা খরচ হয়?`,
`এই গ্রন্থের লেখক কোন সালে জন্মগ্রহণ করেন?`
These use the corpus's own vocabulary and are *about* its subject, but the
answer is not in it. Retrieval will return confident-looking, topically related
spans — and this is exactly where a RAG system hallucinates, because the context
looks relevant. **Most of your 40 should be this kind.**

A threshold tuned only against easy rejections will look excellent and fail in
production.

## Running the sweep

```bash
RERANK_BACKEND=cross_encoder python scripts/sweep_threshold.py eval/questions.json
```

Retrieval only — no model calls, so a sweep costs nothing and can be re-run
freely after any change to OCR, chunking, or the reranker.

Read the `leaked` column first: those are unanswerable questions the system
would answer anyway. If no cutoff drives it to zero while keeping acceptable
recall, the scores do not separate and no threshold will fix it — the problem is
upstream.
