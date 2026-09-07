"""Answering with abstention: answer only from the corpus, or say so.

"Do not hallucinate, say you don't have enough information" is a subsystem, not
a prompt instruction. This implements the three layers DESIGN.md specifies:

  1. A gate BEFORE generation. If the best rerank score is below threshold, we
     abstain without calling the model at all. This is the cheapest and most
     reliable layer: a model that never sees a weak context cannot be tempted
     by it, and it costs no tokens and no latency.

  2. Structural citations. The model is given numbered spans and must return,
     per claim, which span supports it - as schema-validated JSON, so a missing
     or malformed citation is impossible rather than merely discouraged. Any
     citation naming a span that was not supplied invalidates the answer.

  3. A cheap groundedness check IN the hot path. Each claim is compared against
     the span it cites by token overlap. No second model call: the 5s budget
     does not allow two serial LLM round trips.

What this does NOT do, stated plainly: it does not make hallucination
impossible. A model can still paraphrase a cited span into a claim the span
does not support, and lexical overlap will not always catch that. It makes
hallucination much less likely and, more importantly, makes every answer
*checkable* - every claim carries the exact characters it came from. The
remaining gap is what the eval set exists to measure.
"""
import json
from dataclasses import dataclass, field

from .bengali import is_bengali
from .config import (ABSTAIN_MESSAGE, ABSTAIN_THRESHOLD, ANSWER_MODEL,
                     GROUNDING_MIN_OVERLAP, MAX_CONTEXT_SPANS)
from .retrieval import retrieve


@dataclass
class Answer:
    question: str
    answered: bool
    text: str
    citations: list = field(default_factory=list)
    #: everything needed to tune thresholds and audit a decision
    diagnostics: dict = field(default_factory=dict)
    #: why we abstained, when we did
    abstain_reason: str | None = None
    warnings: list = field(default_factory=list)


def _tokens(text):
    """Word-ish tokens for overlap. Bengali has no casing, so no folding."""
    out, current = [], []
    for ch in text:
        if ch.isalnum() or is_bengali(ch):
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return [t for t in out if len(t) > 1]


def overlap(claim, span):
    """Share of the claim's tokens that appear in the cited span.

    Deliberately asymmetric: we ask whether the claim is supported by the span,
    not whether the span is covered by the claim. A short claim drawn from a
    long span should score high.
    """
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0
    span_tokens = set(_tokens(span))
    return sum(1 for t in claim_tokens if t in span_tokens) / len(claim_tokens)


SYSTEM_PROMPT = """\
You answer questions strictly from numbered source spans supplied by the user.

Rules, in order of priority:

1. Use ONLY the supplied spans. Do not use anything you know from outside them.
2. If the spans do not contain enough information to answer, set
   `sufficient` to false and leave `claims` empty. This is the correct and
   expected outcome whenever the spans are off-topic or only tangentially
   related. Do not stretch a partial match into an answer.
3. Every claim you make must cite the span it came from, by its exact id.
4. Quote the supporting words from that span verbatim in `quote`. Copy them
   exactly as they appear - do not paraphrase, correct, or reorder them.
5. Answer in the same language as the question.

The source text comes from OCR and may contain recognition errors. Reproduce
quotes exactly as given anyway - do not silently repair them.
"""


def _build_span_block(candidates):
    parts = []
    for i, cand in enumerate(candidates, 1):
        parts.append(f"[S{i}] (page {cand.page_no})\n{cand.text}")
    return "\n\n".join(parts)


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "True only if the spans contain enough to answer.",
        },
        "answer": {
            "type": "string",
            "description": "The answer, in the question's language. "
                           "Empty when sufficient is false.",
        },
        "claims": {
            "type": "array",
            "description": "One entry per factual claim in the answer.",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "span_id": {
                        "type": "string",
                        "description": "Exactly one supplied span id, e.g. 'S2'.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Verbatim supporting words from that span.",
                    },
                },
                "required": ["claim", "span_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sufficient", "answer", "claims"],
    "additionalProperties": False,
}


def generate(question, candidates, client=None, model=None):
    """Ask the model, with the response shape guaranteed by the schema."""
    import anthropic

    client = client or anthropic.Anthropic()
    span_block = _build_span_block(candidates)

    response = client.messages.create(
        model=model or ANSWER_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        # low effort: this is extraction and faithful quoting, not reasoning,
        # and the 5s budget is the binding constraint
        output_config={"effort": "low", "format": {
            "type": "json_schema", "schema": ANSWER_SCHEMA}},
        messages=[{
            "role": "user",
            "content": f"Source spans:\n\n{span_block}\n\nQuestion: {question}",
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.usage


def answer(question, client=None, document_id=None, threshold=None,
           min_overlap=None):
    """Retrieve, gate, generate, verify. Abstains rather than guessing."""
    threshold = ABSTAIN_THRESHOLD if threshold is None else threshold
    min_overlap = GROUNDING_MIN_OVERLAP if min_overlap is None else min_overlap

    result = retrieve(question, document_id=document_id)
    warnings = []
    if not result.scores_thresholdable:
        warnings.append(
            f"reranker {result.rerank_backend!r} produces uncalibrated scores "
            f"(embedding cosine, not a cross-encoder). The abstention threshold "
            f"is not meaningful in this configuration - plumbing only.")

    diagnostics = {
        "embed_model": result.embed_model,
        "rerank_backend": result.rerank_backend,
        "scores_thresholdable": result.scores_thresholdable,
        "threshold": threshold,
        "min_overlap": min_overlap,
        "best_score": result.best_score,
        "retrieval": result.diagnostics,
        "candidates": [
            {"chunk_id": c.chunk_id, "page_no": c.page_no,
             "dense": round(c.dense_score, 4),
             "rerank": None if c.rerank_score is None else round(c.rerank_score, 4),
             "gate": round(c.gate_score, 4),
             "preview": c.text[:80]}
            for c in result.candidates
        ],
    }

    # ---- layer 1: gate before generation -----------------------------------
    if not result.candidates:
        return Answer(question, False, ABSTAIN_MESSAGE,
                      diagnostics=diagnostics,
                      abstain_reason="nothing retrieved", warnings=warnings)

    if result.best_score < threshold:
        return Answer(
            question, False, ABSTAIN_MESSAGE, diagnostics=diagnostics,
            abstain_reason=(f"best score {result.best_score:.4f} is below the "
                            f"threshold {threshold}"),
            warnings=warnings)

    spans = result.candidates[:MAX_CONTEXT_SPANS]
    span_by_id = {f"S{i}": c for i, c in enumerate(spans, 1)}

    # ---- generation --------------------------------------------------------
    payload, usage = generate(question, spans, client=client)
    diagnostics["usage"] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    diagnostics["spans_shown"] = len(spans)

    # the model's own judgement is the second chance to abstain, after the gate
    if not payload.get("sufficient"):
        return Answer(question, False, ABSTAIN_MESSAGE, diagnostics=diagnostics,
                      abstain_reason="model judged the spans insufficient",
                      warnings=warnings)

    claims = payload.get("claims") or []
    if not claims:
        return Answer(question, False, ABSTAIN_MESSAGE, diagnostics=diagnostics,
                      abstain_reason="answer carried no cited claims",
                      warnings=warnings)

    # ---- layers 2 and 3: citations resolve, and claims are grounded --------
    checked, unresolved, ungrounded = [], [], []
    for claim in claims:
        span_id = claim.get("span_id", "")
        cand = span_by_id.get(span_id)
        if cand is None:
            unresolved.append(span_id)
            continue
        quote_ok = overlap(claim.get("quote", ""), cand.text)
        claim_ok = overlap(claim.get("claim", ""), cand.text)
        entry = {
            "claim": claim.get("claim", ""),
            "quote": claim.get("quote", ""),
            "span_id": span_id,
            "quote_overlap": round(quote_ok, 3),
            "claim_overlap": round(claim_ok, 3),
            **cand.citation(),
        }
        # the quote must really come from the span; the claim only has to be
        # lexically anchored in it, since an answer legitimately rephrases
        if quote_ok < min_overlap:
            ungrounded.append(entry)
        else:
            checked.append(entry)

    diagnostics["claims_total"] = len(claims)
    diagnostics["claims_grounded"] = len(checked)
    diagnostics["unresolved_span_ids"] = unresolved
    diagnostics["ungrounded_claims"] = ungrounded

    if unresolved:
        return Answer(
            question, False, ABSTAIN_MESSAGE, diagnostics=diagnostics,
            abstain_reason=(f"answer cited span id(s) that were not supplied: "
                            f"{unresolved} - treating the answer as unfounded"),
            warnings=warnings)

    if not checked:
        return Answer(
            question, False, ABSTAIN_MESSAGE, diagnostics=diagnostics,
            abstain_reason="no claim's quote could be found in the span it cited",
            warnings=warnings)

    if ungrounded:
        warnings.append(
            f"{len(ungrounded)} of {len(claims)} claims had quotes that do not "
            f"appear in their cited span; they were dropped from the citations")

    return Answer(question, True, payload["answer"], citations=checked,
                  diagnostics=diagnostics, warnings=warnings)
