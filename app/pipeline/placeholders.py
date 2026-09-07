"""chunk / embed / index - registered but not implemented.

These are deliberately stubs rather than guesses. Each is blocked on a decision
that has not been made with evidence yet, and writing plausible-looking versions
now would bury those decisions in code:

  chunk  - needs the real text to size against. Bengali tokenizes 3-4x heavier
           than English, so an English 512-token default is wrong, and
           structure-aware chunking depends on what OCR output actually looks
           like (headings and layout survive the text layer but not plain OCR).
  embed  - BGE-M3 is the design's choice, for cross-lingual coverage in phases
           2-3 without reindexing. Needs the inference host decided.
  index  - Qdrant collection layout depends on whether hybrid search uses
           Qdrant sparse vectors or a separate analyzer.

They appear in the admin stage table as pending, which is honest: the pipeline
shows exactly how far a document can currently get.
"""
from .stages import Stage, StageResult, register


class NotYetImplemented(Stage):
    reason = ""

    def run(self, doc):
        raise NotImplementedError(
            f"the {self.name!r} stage is not implemented yet. {self.reason}")


@register
class Chunk(NotYetImplemented):
    name = "chunk"
    description = "Split normalized page text into retrievable chunks."
    reason = ("Blocked on real extracted text: chunk sizing must be measured "
              "against Bengali token counts, not copied from English defaults.")


@register
class Embed(NotYetImplemented):
    name = "embed"
    description = "Embed chunks with BGE-M3."
    reason = ("Blocked on choosing where BGE-M3 runs (hosted endpoint vs "
              "self-hosted). Chunk text is stored separately from vectors so "
              "changing this later is a re-embed, not a re-extract.")


@register
class Index(NotYetImplemented):
    name = "index"
    description = "Upsert vectors into Qdrant."
    reason = ("Blocked on the hybrid-search decision: Qdrant sparse vectors vs "
              "a separate Bengali analyzer determines the collection layout.")
