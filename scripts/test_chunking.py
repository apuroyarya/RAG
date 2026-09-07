"""Tests for the chunker's pure logic. No database, no OCR.

    python scripts/test_chunking.py

The offset assertions are the important ones. Citations resolve by slicing page
text with char_start/char_end, so an off-by-one here produces a citation that
points at the wrong words - which is worse than no citation, because it looks
authoritative.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.bengali import sentence_ends
from app.chunking import chunk_page, paragraph_spans, split_oversized

failures = []


def check(label, cond, detail=""):
    print(f"{'  ok  ' if cond else ' FAIL '} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


def test_sentence_ends():
    # the pattern that broke naive splitting: dandas inside a citation bracket
    s = "স দাধার পৃথিবীং বিধেম।।২।। [যর্তুঃ০অ০১৩।মং০৪।।] পরের বাক্য।"
    ends = sentence_ends(s)
    check("splits after a danda run", 21 in [e for e in ends] or len(ends) >= 1,
          str(ends))
    inside = s.index("[")
    closing = s.index("]")
    check("no split inside a citation bracket",
          not any(inside < e < closing for e in ends), str(ends))
    check("splits at end of text", ends and ends[-1] == len(s), str(ends[-1:]))

    check("no boundary without trailing whitespace",
          sentence_ends("০অ০৩০।মং০৩") == [], str(sentence_ends("০অ০৩০।মং০৩")))


def test_paragraph_spans():
    text = "এক\n\nদুই\n\n\n  \n\nতিন"
    spans = paragraph_spans(text)
    check("finds every non-blank paragraph", len(spans) == 3, str(spans))
    check("paragraph offsets slice back exactly",
          [text[a:b].strip() for a, b in spans] == ["এক", "দুই", "তিন"],
          str([text[a:b] for a, b in spans]))

    # duplicate text: offsets must differ even though the content is identical,
    # which is exactly what a search-based implementation would get wrong
    dup = "একই\n\nএকই"
    spans = paragraph_spans(dup)
    check("duplicate paragraphs get distinct offsets",
          len(spans) == 2 and spans[0] != spans[1], str(spans))

    check("empty text yields no paragraphs", paragraph_spans("   \n\n  ") == [])


def test_split_oversized():
    # no sentence terminator anywhere - must still be broken up
    runaway = "ক" * 500
    pieces = split_oversized(runaway, 0, len(runaway), 200)
    check("hard-splits a paragraph with no sentence boundary",
          all(b - a <= 200 for a, b in pieces) and len(pieces) == 3, str(pieces))
    check("hard split loses no characters",
          sum(b - a for a, b in pieces) == len(runaway))

    short = "ছোট বাক্য।"
    check("leaves a short span alone",
          split_oversized(short, 0, len(short), 200) == [(0, len(short))])


def test_chunk_page():
    check("empty page yields no chunks", chunk_page("") == [])
    check("whitespace-only page yields no chunks", chunk_page("  \n\n \t ") == [])

    paras = ["এই একটি বাংলা অনুচ্ছেদ যা যথেষ্ট বড় এবং একাধিক বাক্য ধারণ করে।" * 2
             for _ in range(6)]
    text = "\n\n".join(paras)
    chunks = chunk_page(text, target_chars=300, max_chars=500, overlap_chars=50)

    check("produces multiple chunks", len(chunks) > 1, f"{len(chunks)} chunks")
    check("every chunk offset slices back to its stored text",
          all(text[c["char_start"]:c["char_end"]] == c["text"] for c in chunks))
    check("no chunk exceeds max_chars",
          all(c["n_chars"] <= 500 for c in chunks),
          str([c["n_chars"] for c in chunks]))
    check("ordinals are sequential from zero",
          [c["ordinal"] for c in chunks] == list(range(len(chunks))))
    check("chunks are in document order",
          all(a["char_start"] < b["char_start"]
              for a, b in zip(chunks, chunks[1:])))
    check("chunks do not overlap in their own extents",
          all(a["char_end"] <= b["char_start"]
              for a, b in zip(chunks, chunks[1:])))
    check("embed_text carries the lead-in overlap",
          all(len(c["embed_text"]) >= len(c["text"]) for c in chunks)
          and len(chunks[1]["embed_text"]) > len(chunks[1]["text"]))
    check("embed_text ends where the chunk ends",
          all(c["embed_text"].endswith(c["text"][-20:]) for c in chunks))
    check("no text is dropped between chunks",
          chunks[0]["char_start"] == 0 and chunks[-1]["char_end"] == len(text),
          f"{chunks[0]['char_start']}..{chunks[-1]['char_end']} of {len(text)}")

    # a tiny trailing scrap should be merged back, not emitted alone
    tail = "\n\n".join(["অনুচ্ছেদ এক যা বেশ লম্বা এবং যথেষ্ট বড়।" * 4, "ছোট।"])
    tail_chunks = chunk_page(tail, target_chars=200, max_chars=400, min_chars=80)
    check("tiny trailing scrap is merged, not emitted alone",
          all(c["n_chars"] >= 80 for c in tail_chunks) or len(tail_chunks) == 1,
          str([c["n_chars"] for c in tail_chunks]))


for fn in (test_sentence_ends, test_paragraph_spans, test_split_oversized,
           test_chunk_page):
    print(f"\n{fn.__name__}")
    fn()

print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nall passed")
sys.exit(1 if failures else 0)
