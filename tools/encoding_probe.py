"""Bengali PDF text-layer triage.

Decides, per document, whether the embedded text layer is TRUSTWORTHY or must be
bypassed via render+OCR.

Why this exists: a Bengali PDF can have a text layer that is 97% Bengali
codepoints and still be systematically WRONG, because the embedded font lacks a
correct /ToUnicode cmap. The text embeds, indexes and retrieves cleanly, and
produces confident nonsense. A codepoint-ratio check does NOT catch this.

Three layers of detection, cheapest first:
  1. STRUCTURAL  - font encoding objects (free, caught both known-bad samples)
  2. ORTHOGRAPHIC - sequences illegal in Bengali (free)
  3. OCR CROSS-CHECK - render sampled pages, OCR, compare (costs money; truth)

Layers 1-2 run here. Layer 3 is the pipeline's normalize-stage gate.
"""
import sys, re, unicodedata
from os.path import basename
from collections import Counter
import pymupdf

BENG = lambda c: 0x0980 <= ord(c) < 0x0A00
VIRAMA = "\u09cd"
# dependent vowel signs + signs that cannot open a cluster
DEPENDENT = set("\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4\u09c7\u09c8"
                "\u09cb\u09cc\u09d7\u09bc") | {VIRAMA}
LEGACY_FONT_HINT = re.compile(r"(MJ|SutonnyMJ|Bijoy|Boishakhi)\b", re.I)


def font_structure(doc, pages):
    """Layer 1: inspect font objects for encodings that cannot round-trip."""
    findings, seen = [], set()
    for i in pages:
        for xref, _, _, name, *_ in doc[i].get_fonts():
            if xref in seen:
                continue
            seen.add(xref)
            obj = doc.xref_object(xref)
            has_tounicode = "/ToUnicode" in obj
            enc = "WinAnsiEncoding" if "WinAnsiEncoding" in obj else (
                  "Identity-H" if "Identity-H" in obj else "other")
            bad = (not has_tounicode) or bool(LEGACY_FONT_HINT.search(name))
            findings.append((name, enc, has_tounicode, bad))
    return findings


def orthographic(text):
    """Layer 2: count sequences that are illegal or near-impossible in Bengali."""
    tokens = [t for t in text.split() if any(BENG(c) for c in t)]
    initial_dep = [t for t in tokens if t and t[0] in DEPENDENT]     # illegal
    triples = re.findall(r"([\u0985-\u09b9])\1\1", text)             # 3x same letter
    dep_after_space = re.findall(r"\s[\u09be-\u09cc\u09cd]", text)   # orphaned sign
    return {
        "tokens": len(tokens),
        "token_initial_dependent_sign": len(initial_dep),
        "samples": initial_dep[:8],
        "triple_letter_runs": len(triples),
        "orphaned_signs": len(dep_after_space),
    }


def probe(path, sample=6):
    doc = pymupdf.open(path)
    pages = list(range(min(sample, doc.page_count)))
    text = "\n".join(doc[i].get_text() for i in pages)

    n_beng = sum(1 for c in text if BENG(c))
    n_vis = sum(1 for c in text if not c.isspace())
    ratio = n_beng / max(n_vis, 1)

    fonts = font_structure(doc, pages)
    orth = orthographic(text)
    empty_pages = sum(1 for i in pages if len(doc[i].get_text().strip()) < 20)

    bad_fonts = [f for f in fonts if f[3]]
    illegal_rate = orth["token_initial_dependent_sign"] / max(orth["tokens"], 1)

    if bad_fonts or illegal_rate > 0.005 or orth["triple_letter_runs"]:
        verdict = "TEXT LAYER UNTRUSTWORTHY -> render + OCR"
    elif ratio < 0.4 and n_vis > 50:
        verdict = "NOT BENGALI / legacy ASCII -> inspect"
    elif empty_pages == len(pages):
        verdict = "NO TEXT LAYER -> OCR"
    else:
        verdict = "text layer plausible -> OCR cross-check to confirm"

    print(f"\n{'='*78}\n{basename(path)}\n{'='*78}")
    print(f"pages={doc.page_count}  sampled={len(pages)}  bengali_codepoint_ratio={ratio:.1%}"
          f"  empty_pages={empty_pages}")
    print(f"\n[1] fonts ({len(bad_fonts)}/{len(fonts)} problematic):")
    for name, enc, tu, bad in fonts:
        print(f"    {'BAD ' if bad else 'ok  '} {name:42s} {enc:16s} ToUnicode={tu}")
    print(f"\n[2] orthographic:")
    for k, v in orth.items():
        print(f"    {k:32s} {v}")
    print(f"    token_initial_dependent_sign rate  {illegal_rate:.2%}  (Bengali words cannot"
          f" begin with a vowel sign -> any hit is corruption)")
    print(f"\nVERDICT: {verdict}")
    doc.close()
    return verdict


if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
