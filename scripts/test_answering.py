"""Tests for the abstention subsystem, with a stubbed model. No API key, no cost.

    python scripts/test_answering.py

This is the most important test file in the project. The requirement is that the
system answers only from the corpus and otherwise says it does not know, and
these cases are the ways that requirement gets broken:

  * a weak-context question slipping past the gate
  * an answer citing a span that was never supplied
  * an answer whose quote does not appear in the span it cites
  * a model claiming sufficiency but returning no citations

Every one of those must end in abstention, not an answer.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

import app.answering as answering
from app.answering import Answer, answer, overlap
from app.config import ABSTAIN_MESSAGE
from app.retrieval import Candidate, RetrievalResult

failures = []


def check(label, cond, detail=""):
    print(f"{'  ok  ' if cond else ' FAIL '} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


SPAN_TEXT = ("অথ দেবযজ্ঞ বিধিঃ। দেবযজ্ঞ প্রতিদিন প্রাতঃকালে ও সায়ংকালে "
             "সম্পন্ন করিতে হয়।")


def fake_candidate(score, text=SPAN_TEXT, page=2, ordinal=0):
    return Candidate(
        chunk_id=f"doc:{page}:{ordinal}", document_id="doc",
        filename="test.pdf", page_no=page, ordinal=ordinal,
        char_start=100, char_end=100 + len(text), text=text,
        dense_score=score, rerank_score=score, gate_score=score)


def stub_retrieve(candidates, thresholdable=True):
    def _retrieve(question, dense_k=None, rerank_k=None, document_id=None):
        return RetrievalResult(
            question=question, candidates=candidates,
            embed_model="stub:8", rerank_backend="stub",
            scores_thresholdable=thresholdable, diagnostics={})
    return _retrieve


def stub_generate(payload):
    def _generate(question, candidates, client=None, model=None):
        return payload, SimpleNamespace(input_tokens=100, output_tokens=50)
    return _generate


def run(candidates, payload, threshold=0.0, thresholdable=True, **kwargs):
    answering.retrieve = stub_retrieve(candidates, thresholdable)
    answering.generate = stub_generate(payload)
    return answer("দেবযজ্ঞ কখন করা হয়?", threshold=threshold, **kwargs)


GOOD_PAYLOAD = {
    "sufficient": True,
    "answer": "দেবযজ্ঞ প্রতিদিন প্রাতঃকালে ও সায়ংকালে সম্পন্ন করিতে হয়।",
    "claims": [{
        "claim": "দেবযজ্ঞ প্রাতঃকালে ও সায়ংকালে করা হয়",
        "span_id": "S1",
        "quote": "দেবযজ্ঞ প্রতিদিন প্রাতঃকালে ও সায়ংকালে সম্পন্ন করিতে হয়",
    }],
}


def test_overlap():
    check("an exact quote scores 1.0", overlap("দেবযজ্ঞ বিধিঃ", SPAN_TEXT) == 1.0)
    check("a fabricated quote scores 0.0",
          overlap("ক্রিকেট খেলার নিয়ম সম্পূর্ণ", SPAN_TEXT) == 0.0)
    check("an empty quote scores 0.0", overlap("", SPAN_TEXT) == 0.0)
    check("overlap is asymmetric - a short claim from a long span scores high",
          overlap("দেবযজ্ঞ", SPAN_TEXT) == 1.0)


def test_gate():
    res = run([], GOOD_PAYLOAD)
    check("abstains when nothing is retrieved", not res.answered)
    check("  and says why", "nothing retrieved" in (res.abstain_reason or ""))

    res = run([fake_candidate(-2.5)], GOOD_PAYLOAD, threshold=0.0)
    check("abstains when the best score is below threshold", not res.answered)
    check("  and reports the score and threshold",
          "below the threshold" in (res.abstain_reason or ""),
          res.abstain_reason or "")

    # the gate must fire BEFORE generation - a model that never sees weak
    # context cannot be tempted by it, and it costs nothing
    check("  without calling the model", "usage" not in res.diagnostics)

    res = run([fake_candidate(3.0)], GOOD_PAYLOAD, threshold=0.0)
    check("answers when the score clears the threshold", res.answered,
          res.abstain_reason or "")


def test_abstain_message():
    res = run([], GOOD_PAYLOAD)
    check("abstention returns the configured Bengali message",
          res.text == ABSTAIN_MESSAGE, res.text)


def test_model_says_insufficient():
    res = run([fake_candidate(3.0)],
              {"sufficient": False, "answer": "", "claims": []})
    check("respects the model judging the spans insufficient", not res.answered)
    check("  and does not leak a partial answer", res.text == ABSTAIN_MESSAGE)


def test_no_claims():
    res = run([fake_candidate(3.0)],
              {"sufficient": True, "answer": "কিছু উত্তর", "claims": []})
    check("abstains when an answer carries no cited claims", not res.answered,
          res.abstain_reason or "")


def test_unresolved_citation():
    payload = json.loads(json.dumps(GOOD_PAYLOAD))
    payload["claims"][0]["span_id"] = "S99"     # never supplied
    res = run([fake_candidate(3.0)], payload)
    check("abstains when a citation names a span that was not supplied",
          not res.answered, res.abstain_reason or "")
    check("  and records the bad id",
          res.diagnostics.get("unresolved_span_ids") == ["S99"],
          str(res.diagnostics.get("unresolved_span_ids")))


def test_fabricated_quote():
    payload = json.loads(json.dumps(GOOD_PAYLOAD))
    payload["claims"][0]["quote"] = "এই বাক্যটি কোথাও নেই একেবারেই ভিন্ন"
    res = run([fake_candidate(3.0)], payload)
    check("abstains when no quote is found in its cited span",
          not res.answered, res.abstain_reason or "")
    check("  and records the ungrounded claim",
          len(res.diagnostics.get("ungrounded_claims", [])) == 1)


def test_partially_grounded():
    payload = json.loads(json.dumps(GOOD_PAYLOAD))
    payload["claims"].append({
        "claim": "অন্য একটি দাবি",
        "span_id": "S1",
        "quote": "সম্পূর্ণ বানানো উদ্ধৃতি যা নেই কোথাও",
    })
    res = run([fake_candidate(3.0)], payload)
    check("answers when at least one claim is grounded", res.answered,
          res.abstain_reason or "")
    check("  keeping only the grounded citation", len(res.citations) == 1,
          str(len(res.citations)))
    check("  and warning about the dropped one",
          any("do not appear" in w for w in res.warnings), str(res.warnings))


def test_citations_carry_offsets():
    res = run([fake_candidate(3.0)], GOOD_PAYLOAD)
    cite = res.citations[0]
    check("citation carries the page number", cite["page_no"] == 2)
    check("citation carries character offsets",
          cite["char_start"] == 100 and cite["char_end"] == 100 + len(SPAN_TEXT),
          f"{cite['char_start']}..{cite['char_end']}")
    check("citation carries the chunk id", cite["chunk_id"] == "doc:2:0")
    check("citation records its measured overlap",
          cite["quote_overlap"] >= 0.6, str(cite["quote_overlap"]))


def test_uncalibrated_warning():
    res = run([fake_candidate(3.0)], GOOD_PAYLOAD, thresholdable=False)
    check("warns loudly when the reranker's scores are not thresholdable",
          any("uncalibrated" in w for w in res.warnings), str(res.warnings))
    check("  and records it in diagnostics",
          res.diagnostics["scores_thresholdable"] is False)


for fn in (test_overlap, test_gate, test_abstain_message,
           test_model_says_insufficient, test_no_claims,
           test_unresolved_citation, test_fabricated_quote,
           test_partially_grounded, test_citations_carry_offsets,
           test_uncalibrated_warning):
    print(f"\n{fn.__name__}")
    fn()

print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nall passed")
sys.exit(1 if failures else 0)
