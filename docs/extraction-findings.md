# Extraction findings — Bengali sample PDFs

Date: 2026-09-07. Tool: `tools/encoding_probe.py`. PyMuPDF 1.28.2.

## Result: both samples' text layers are UNUSABLE. Do not index them.

| | `দৈনিক দেবযজ্ঞ বিধি.pdf` | `কৃষ্ণচরিত্র … শাস্ত্রপৃষ্ঠা.pdf` |
|---|---|---|
| pages | 8 | 354 |
| Bengali codepoint ratio | 92.0% | 95.8% |
| fonts missing `/ToUnicode` | 3 / 7 | 11 / 17 |
| token-initial vowel signs (illegal) | 9.27% | 0.66% |
| verdict | untrustworthy | untrustworthy |

## Evidence — rendered glyphs vs extracted text

File 1 (verified against a 140-dpi render of page 2):

| Rendered (truth) | Extracted |
|---|---|
| দেবযজ্ঞ | দৈবযজ্ঞ |
| বিধিঃ | নবনিিঃ |
| প্রার্থনোপাসনা | প্রাথথথিাপাসিা |
| মন্ত্রাঃ | মন্ত্ািঃ |
| পৃষ্ঠা নং ১ | পৃষ্ঠা িং 1 |

Systematic: `ব→নব`, `ধ→ন`, `ত→ন`, `ে→ৈ`, `ং→িং`, `র` dropped.

File 2 (verified against the OS filename, which is correct Bengali):

| Truth | Extracted |
|---|---|
| ঋষি বঙ্কিমচন্দ্র চট্টোপাধ্যায় | ঋষি বষিমচন্দ্র চট্টোপোধ্যোয় |

Different substitutions (`ঙ্ক→ষ`, `া→ো`) because different fonts.

## Root cause

Not legacy ASCII encoding (the predicted failure). The PDFs embed subsetted
Bengali fonts declared `TrueType / WinAnsiEncoding, FirstChar 32, LastChar 95`
with **no `/ToUnicode` cmap**. Bengali glyphs sit in ASCII slots with no reverse
map, so extraction guesses and guesses wrongly. The pages *render* correctly —
the glyphs are right, only the text mapping is broken.

Both files mix good and bad fonts *within a single document*: e.g. file 1 has
`BCDEEE+NikoshBAN` as `Identity-H` with `ToUnicode` (extracts fine) alongside
`BCDGEE+NikoshBAN` as `WinAnsiEncoding` without it (garbage). So "mixed encoding"
is per-font-run, not per-document.

## Why a codepoint-ratio check is not enough

Both files are 92–96% Bengali codepoints and both are wrong. The naive detector
in the first draft of the design passed both. Corrupted text embeds cleanly,
retrieves cleanly, and yields confident nonsense that the abstention gate cannot
detect — because the bad text *is* the corpus.

## Detectors that do work (both free)

1. **Structural** — font declares no `/ToUnicode`, or its name matches the legacy
   family pattern (`*MJ`, Bijoy, Boishakhi). Caught both samples.
2. **Orthographic** — a Bengali word cannot begin with a dependent vowel sign
   (U+09BE–U+09CC) or a virama. Any occurrence is corruption. Caught both
   (9.27% and 0.66% of tokens).

Ground truth remains layer 3: render sampled pages, OCR them, compare.

## Consequence for the design

Per-font correction tables are not viable — the mapping differs per font subset,
and file 2 alone has 11 bad subsets. **Render + Bengali OCR is the primary
extraction path for this corpus**, not the fallback. The text layer is used only
where layers 1–2 pass *and* an OCR cross-check on sampled pages agrees.

This reverses the earlier plan, which demoted OCR on the grounds that the PDFs
were digital text. They are digital text; the text is simply wrong.

## Open

- [ ] Confirm whether file 2's doubled letters (`োগগ`, `ারীী`) come from two
      interleaved text runs (good subset + bad subset drawing the same content).
      If so, deduplication may recover some pages without OCR.
- [ ] Choose and validate a Bengali OCR engine on these two files before
      committing to it.
- [ ] Re-run the probe over ~20 more real documents to get the trustworthy /
      untrustworthy split, which sets the OCR budget.
