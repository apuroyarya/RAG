"""Bengali text validity helpers, shared by the extraction probe and OCR bench.

The orthographic check here is the one quality signal that needs no ground
truth: Bengali has hard rules about what can start a cluster, so violations are
proof of corruption whether they came from a broken PDF font or a bad OCR pass.
"""
import re
import unicodedata

BENGALI_BLOCK = range(0x0980, 0x0A00)
VIRAMA = "\u09cd"

# Dependent vowel signs and marks that cannot open a syllable cluster.
DEPENDENT_SIGNS = set(
    "\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4"
    "\u09c7\u09c8\u09cb\u09cc\u09d7\u09bc"
) | {VIRAMA}

# Legacy ASCII-mapped Bengali font families (Bijoy/SutonnyMJ lineage).
LEGACY_FONT_HINT = re.compile(r"(MJ\b|SutonnyMJ|Bijoy|Boishakhi)", re.I)

_TRIPLE_LETTER = re.compile(r"([\u0985-\u09b9])\1\1")
_ORPHANED_SIGN = re.compile(r"\s[\u09be-\u09cc\u09cd]")
_WS = re.compile(r"\s+")


def is_bengali(ch):
    return ord(ch) in BENGALI_BLOCK


def bengali_ratio(text):
    """Share of visible characters that are Bengali codepoints.

    Deliberately NOT a quality signal: both known-bad sample PDFs score 92-96%
    here while being systematically wrong. Reported for context only.
    """
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    return sum(1 for c in visible if is_bengali(c)) / len(visible)


def normalize(text, collapse_whitespace=True):
    """NFC-normalize. PDF and OCR output both emit Indic conjuncts in visual
    rather than logical order, so without this `ক্ষ` will not match a typed query.
    """
    out = unicodedata.normalize("NFC", text)
    if collapse_whitespace:
        out = _WS.sub(" ", out).strip()
    return out


def orthographic_report(text):
    """Count sequences that are illegal or near-impossible in Bengali.

    `initial_dependent_rate` is the load-bearing number: a Bengali word cannot
    begin with a dependent vowel sign or virama, so any nonzero rate means the
    text is corrupt. No reference text required.
    """
    tokens = [t for t in text.split() if any(is_bengali(c) for c in t)]
    initial_dep = [t for t in tokens if t and t[0] in DEPENDENT_SIGNS]
    return {
        "tokens": len(tokens),
        "initial_dependent": len(initial_dep),
        "initial_dependent_rate": len(initial_dep) / max(len(tokens), 1),
        "initial_dependent_samples": initial_dep[:8],
        "triple_letter_runs": len(_TRIPLE_LETTER.findall(text)),
        "orphaned_signs": len(_ORPHANED_SIGN.findall(text)),
    }


def looks_corrupt(text, threshold=0.005):
    """Ground-truth-free verdict. True means do not index this text."""
    r = orthographic_report(text)
    return r["initial_dependent_rate"] > threshold or r["triple_letter_runs"] > 0
