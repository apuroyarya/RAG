"""Sweep the abstention threshold against labelled questions.

    python scripts/sweep_threshold.py questions.json

`questions.json` is a list of {"question": ..., "answerable": true|false}.
Answerable means the corpus genuinely contains the answer; unanswerable means it
does not and the system must abstain.

For every candidate cutoff this reports:

  answered   - answerable questions that would still get an answer (recall)
  rejected   - unanswerable questions correctly abstained on
  leaked     - unanswerable questions that would get an ANSWER anyway

`leaked` is the column that matters. A leaked question is the system answering
from context that does not support an answer - the exact failure the whole
design exists to prevent - so the right cutoff is the lowest one that drives
leaked to zero while keeping `answered` acceptable. If no cutoff does both, the
scores do not separate and the fix is upstream (better OCR, better chunking,
better reranker), not a different number.

Retrieval only - no model calls, so a sweep costs nothing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.retrieval import retrieve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", help="JSON list of {question, answerable}")
    ap.add_argument("--out", default=None, help="write scores to this JSON file")
    args = ap.parse_args()

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if not items:
        raise SystemExit("no questions")

    scored = []
    for item in items:
        result = retrieve(item["question"])
        if not result.scores_thresholdable:
            raise SystemExit(
                f"reranker {result.rerank_backend!r} produces uncalibrated "
                f"scores - a sweep against them is meaningless. "
                f"Set RERANK_BACKEND=cross_encoder.")
        scored.append({
            "question": item["question"],
            "answerable": bool(item["answerable"]),
            "best_score": result.best_score,
            "top_page": (result.candidates[0].page_no
                         if result.candidates else None),
        })
        mark = "A" if item["answerable"] else "U"
        best = result.best_score
        print(f"  [{mark}] {'-' if best is None else f'{best:8.4f}'}  "
              f"{item['question'][:56]}")

    answerable = [s for s in scored if s["answerable"]]
    unanswerable = [s for s in scored if not s["answerable"]]
    print(f"\n{len(answerable)} answerable, {len(unanswerable)} unanswerable")

    if not answerable or not unanswerable:
        print("\nA sweep needs both kinds. Unanswerable questions are what make "
              "the threshold measurable - without them you can only see recall.")
        return 0

    cuts = sorted({round(s["best_score"], 3) for s in scored
                   if s["best_score"] is not None})
    # include a point below the lowest score so "answer everything" is visible
    cuts = [cuts[0] - 0.5] + cuts

    print(f"\n{'cutoff':>9}  {'answered':>9}  {'rejected':>9}  {'leaked':>7}")
    print("-" * 42)
    best_row = None
    for cut in cuts:
        answered = sum(1 for s in answerable
                       if s["best_score"] is not None and s["best_score"] >= cut)
        leaked = sum(1 for s in unanswerable
                     if s["best_score"] is not None and s["best_score"] >= cut)
        rejected = len(unanswerable) - leaked
        print(f"{cut:9.3f}  {answered:4d}/{len(answerable):<4d}  "
              f"{rejected:4d}/{len(unanswerable):<4d}  {leaked:7d}")
        if leaked == 0 and best_row is None:
            best_row = (cut, answered)

    print()
    if best_row is None:
        print("NO cutoff rejects every unanswerable question. The scores do not "
              "separate, so no threshold will fix this - look upstream at OCR "
              "quality, chunking, and the reranker.")
    else:
        cut, answered = best_row
        print(f"Lowest cutoff with zero leakage: {cut:.3f} "
              f"(still answers {answered}/{len(answerable)} answerable)")
        print(f"Set ABSTAIN_THRESHOLD={cut:.3f} - and note this is fitted to "
              f"these questions. Hold back a test split, or it is tuned to noise.")

    if args.out:
        Path(args.out).write_text(json.dumps(scored, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nscores written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
