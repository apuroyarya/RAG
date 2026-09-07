"""index - registered but not implemented.

These are deliberately stubs rather than guesses. Each is blocked on a decision
that has not been made with evidence yet, and writing plausible-looking versions
now would bury those decisions in code:

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
class Index(NotYetImplemented):
    name = "index"
    description = "Upsert vectors into Qdrant."
    reason = ("Blocked on the hybrid-search decision: Qdrant sparse vectors vs "
              "a separate Bengali analyzer determines the collection layout.")
