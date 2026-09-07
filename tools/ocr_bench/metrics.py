"""Scoring for the OCR benchmark.

Two regimes, because Bengali OCR ground truth is expensive:

  WITH gold text  -> CER / WER, the real numbers. Needs hand transcription, so
                     only worth doing for a handful of representative pages.
  WITHOUT gold    -> cross-engine consensus plus the orthographic validity check
                     from tools.bengali. Free, and it localises disagreement so
                     you only transcribe the pages where engines actually differ.

Consensus is a proxy, not truth: engines can agree and both be wrong, most
plausibly on the same rare conjuncts. Treat a high-agreement page as "probably
fine, spot-check one" and a low-agreement page as "transcribe this one".
"""
from ..bengali import bengali_ratio, normalize, orthographic_report


def _levenshtein(a, b):
    """Edit distance over any two sequences. Two-row DP, no dependencies."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """Character error rate. 0.0 is perfect; can exceed 1.0 on insertions."""
    ref = normalize(reference)
    if not ref:
        return None
    return _levenshtein(ref, normalize(hypothesis)) / len(ref)


def wer(reference, hypothesis):
    """Word error rate over whitespace-separated tokens."""
    ref = normalize(reference).split()
    if not ref:
        return None
    return _levenshtein(ref, normalize(hypothesis).split()) / len(ref)


def similarity(a, b):
    """1.0 = identical after normalisation. Symmetric, unlike CER."""
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    denom = max(len(na), len(nb)) or 1
    return 1.0 - _levenshtein(na, nb) / denom


def quality_signals(text):
    """Ground-truth-free per-engine signals.

    `initial_dependent_rate` is the one to read first: nonzero means the engine
    emitted Bengali words starting with a dependent vowel sign, which is
    orthographically impossible and therefore proof of garbage.

    `chars` matters too — an engine that silently drops half a page looks fine
    on validity checks but is useless. Compare it across engines.
    """
    orth = orthographic_report(text)
    return {
        "chars": len(normalize(text)),
        "tokens": orth["tokens"],
        "bengali_ratio": bengali_ratio(text),
        "initial_dependent_rate": orth["initial_dependent_rate"],
        "triple_letter_runs": orth["triple_letter_runs"],
        "suspect_samples": orth["initial_dependent_samples"],
    }


def consensus_matrix(texts_by_engine):
    """Pairwise similarity between every pair of engines on one page."""
    names = sorted(texts_by_engine)
    return {
        (a, b): similarity(texts_by_engine[a], texts_by_engine[b])
        for i, a in enumerate(names) for b in names[i + 1:]
    }


def mean_agreement(texts_by_engine, exclude=("textlayer",)):
    """Average pairwise similarity per engine, excluding the corrupt baseline.

    An engine that disagrees with everyone is either much better or much worse
    than the rest — the report flags it, you look at the text to decide which.
    """
    names = [n for n in texts_by_engine if n not in exclude]
    if len(names) < 2:
        return {}
    out = {}
    for a in names:
        scores = [similarity(texts_by_engine[a], texts_by_engine[b])
                  for b in names if b != a]
        out[a] = sum(scores) / len(scores)
    return out
