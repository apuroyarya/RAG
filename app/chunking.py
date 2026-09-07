"""Splitting page text into retrievable chunks.

Pure functions, no database, so the behaviour can be tested directly.

Every chunk carries exact character offsets into the page text it came from.
That is not bookkeeping for its own sake: the abstention design requires the
model to cite spans, and a citation is only verifiable if it resolves back to
the precise characters a claim came from. Offsets are the mechanism.

Three decisions, each measured against the sample corpus rather than assumed:

  Chunks never cross a page boundary. Pages here average ~1000 characters, so
  the cost is small, and it means every chunk cites exactly one page. A chunk
  spanning pages would make its citation ambiguous for no retrieval benefit.

  Paragraphs are the primary unit, accumulated up to a target size. Paragraph
  splitting alone gives fragments of 18-50 characters on this corpus, too small
  to embed meaningfully; whole pages as single chunks are too coarse for a text
  where one page holds several independent mantras, and coarse chunks dilute the
  relevance scores the abstention gate reads.

  Sentence splitting is the fallback for an oversized paragraph, and it is
  bracket-aware. See `sentence_ends` in bengali.py for why.

Sizes are in characters, not tokens. Bengali tokenizes far heavier than English,
so an English token default would be wrong - but the true ratio depends on the
embedding model's tokenizer, which is not chosen yet. Characters are honest and
model-independent; `scripts/measure_tokens.py` converts once an embedder exists.
"""
import re

from .bengali import sentence_ends

_PARA_BREAK = re.compile(r"\n\s*\n")


def paragraph_spans(text):
    """[(start, end)] of non-blank paragraphs, with exact offsets.

    Uses finditer rather than split() so offsets stay accurate - split() loses
    the positions, and reconstructing them by searching is wrong when the same
    paragraph text appears twice on a page.
    """
    spans, pos = [], 0
    for match in _PARA_BREAK.finditer(text):
        if text[pos:match.start()].strip():
            spans.append((pos, match.start()))
        pos = match.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def _tighten(text, start, end):
    """Trim surrounding whitespace without losing offset accuracy."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_oversized(text, start, end, max_chars):
    """Break one too-long span at sentence boundaries; hard-split if it has none.

    A paragraph with no usable sentence terminator still has to be broken, or a
    single runaway block would exceed the embedder's input limit. Falling back
    to a hard character split is lossy at the seam, which is why it comes last.
    """
    if end - start <= max_chars:
        return [(start, end)]

    boundaries = [start + e for e in sentence_ends(text[start:end])]
    pieces, piece_start = [], start
    for boundary in boundaries:
        if boundary - piece_start >= max_chars:
            pieces.append((piece_start, boundary))
            piece_start = boundary
    if piece_start < end:
        pieces.append((piece_start, end))

    out = []
    for a, b in pieces:
        while b - a > max_chars:          # no sentence boundary to use
            out.append((a, a + max_chars))
            a += max_chars
        if b > a:
            out.append((a, b))
    return out


def chunk_page(text, target_chars=400, max_chars=700, min_chars=80,
               overlap_chars=60):
    """Split one page into chunks. Returns dicts with text and offsets.

    `overlap_chars` repeats the tail of the previous chunk at the start of the
    next so a sentence split across a boundary is still retrievable from both
    sides. The overlap is reported separately from the chunk's own span, so a
    citation still points at the chunk's real extent.
    """
    if not text or not text.strip():
        return []

    units = []
    for start, end in paragraph_spans(text):
        units.extend(split_oversized(text, start, end, max_chars))

    # accumulate adjacent units up to the target
    groups, current = [], None
    for start, end in units:
        if current is None:
            current = [start, end]
            continue
        if end - current[0] <= target_chars:
            current[1] = end
        else:
            groups.append(tuple(current))
            current = [start, end]
    if current is not None:
        groups.append(tuple(current))

    # a trailing scrap is better merged back than emitted on its own
    if len(groups) > 1 and (groups[-1][1] - groups[-1][0]) < min_chars:
        last = groups.pop()
        prev = groups.pop()
        groups.append((prev[0], last[1]))

    chunks = []
    for ordinal, (start, end) in enumerate(groups):
        start, end = _tighten(text, start, end)
        if start >= end:
            continue
        overlap_start = start
        if overlap_chars and chunks:
            overlap_start = max(chunks[-1]["char_start"], start - overlap_chars)
        chunks.append({
            "ordinal": ordinal,
            "char_start": start,
            "char_end": end,
            "text": text[start:end],
            # what actually gets embedded: the chunk plus its lead-in overlap
            "embed_text": text[overlap_start:end],
            "n_chars": end - start,
        })
    return chunks
