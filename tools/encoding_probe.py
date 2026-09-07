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
import sys
from os.path import abspath, basename, dirname

import pymupdf

# Runs as `python -m tools.encoding_probe` or as a bare script path.
try:
    from .bengali import LEGACY_FONT_HINT, bengali_ratio, is_bengali, orthographic_report
except ImportError:
    sys.path.insert(0, dirname(dirname(abspath(__file__))))
    from tools.bengali import (LEGACY_FONT_HINT, bengali_ratio, is_bengali,
                               orthographic_report)

if hasattr(sys.stdout, "reconfigure"):  # Bengali dies under cp1252 otherwise
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENG = is_bengali


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


def probe(path, sample=6):
    doc = pymupdf.open(path)
    pages = list(range(min(sample, doc.page_count)))
    text = "\n".join(doc[i].get_text() for i in pages)

    n_vis = sum(1 for c in text if not c.isspace())
    ratio = bengali_ratio(text)

    fonts = font_structure(doc, pages)
    orth = orthographic_report(text)  # layer 2, shared with the OCR bench
    empty_pages = sum(1 for i in pages if len(doc[i].get_text().strip()) < 20)

    bad_fonts = [f for f in fonts if f[3]]
    illegal_rate = orth["initial_dependent_rate"]

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
    print(f"    -> initial_dependent_rate {illegal_rate:.2%}: Bengali words cannot begin"
          f" with a dependent vowel sign, so any hit is corruption")
    print(f"\nVERDICT: {verdict}")
    doc.close()
    return verdict


if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
