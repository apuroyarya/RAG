"""Ask the corpus a question, showing every score behind the decision.

    python scripts/ask.py "প্রশ্ন এখানে"
    python scripts/ask.py "..." --retrieval-only     # no model call, no cost
    python scripts/ask.py "..." --threshold 0.5

`--retrieval-only` is the mode to use while collecting threshold data: it prints
the rerank score for every candidate without spending a token, so you can run
your eval questions through and see where answerable and unanswerable ones
actually separate. That separation is what sets ABSTAIN_THRESHOLD; the value in
config is a placeholder.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import config
from app.answering import answer
from app.retrieval import retrieve


def show_candidates(result):
    print(f"\nembedding : {result.embed_model}")
    print(f"reranker  : {result.rerank_backend}"
          f"{'' if result.scores_thresholdable else '  <-- NOT thresholdable'}")
    print(f"retrieval : {result.diagnostics}")
    if not result.candidates:
        print("\nno candidates retrieved")
        return
    print(f"\n{'gate':>9}  {'dense':>7}  {'rerank':>8}  page  text")
    print("-" * 78)
    for c in result.candidates:
        rerank = "-" if c.rerank_score is None else f"{c.rerank_score:8.4f}"
        preview = c.text[:44].replace("\n", " ")
        print(f"{c.gate_score:9.4f}  {c.dense_score:7.4f}  {rerank}  "
              f"{c.page_no:4d}  {preview}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="show scores without calling the model (free)")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"override the abstention gate "
                         f"(config default: {config.ABSTAIN_THRESHOLD})")
    ap.add_argument("--document-id", default=None,
                    help="restrict retrieval to one document")
    args = ap.parse_args()

    print(f"question: {args.question}")

    if args.retrieval_only:
        result = retrieve(args.question, document_id=args.document_id)
        show_candidates(result)
        threshold = (config.ABSTAIN_THRESHOLD if args.threshold is None
                     else args.threshold)
        best = result.best_score
        if best is None:
            verdict = "ABSTAIN (nothing retrieved)"
        else:
            verdict = ("ANSWER" if best >= threshold else "ABSTAIN")
            verdict += f" (best {best:.4f} vs threshold {threshold})"
        print(f"\ngate would: {verdict}")
        if not result.scores_thresholdable:
            print("  but that verdict is meaningless: this reranker's scores "
                  "are embedding cosine, not calibrated cross-encoder scores")
        return 0

    result = answer(args.question, document_id=args.document_id,
                    threshold=args.threshold)

    print(f"\n{'ANSWERED' if result.answered else 'ABSTAINED'}")
    if not result.answered:
        print(f"reason: {result.abstain_reason}")
    print(f"\n{result.text}\n")

    for i, cite in enumerate(result.citations, 1):
        print(f"  [{i}] {cite['filename']} page {cite['page_no']} "
              f"chars {cite['char_start']}-{cite['char_end']} "
              f"(quote overlap {cite['quote_overlap']})")
        print(f"      claim: {cite['claim'][:80]}")
        print(f"      quote: {cite['quote'][:80]}")

    for warning in result.warnings:
        print(f"\n  ! {warning}")

    print("\ndiagnostics:")
    for key, value in result.diagnostics.items():
        if key == "candidates":
            print(f"  candidates:")
            for c in value:
                print(f"    gate={c['gate']:8.4f} dense={c['dense']:7.4f} "
                      f"rerank={c['rerank']} p{c['page_no']} {c['preview'][:40]}")
        else:
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
