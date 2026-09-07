"""Stage contract and registry.

A stage takes a document, does one thing, and returns metrics plus an optional
artifact path. It does not know how it was triggered, whether anything ran
before it, or what runs next - the runner owns all of that. That separation is
what lets manual triggers and an auto worker share one implementation.

Adding a stage means: append to STAGE_ORDER, add the value to the stage_name
enum in a migration, and register the class here.
"""
from dataclasses import dataclass, field

#: Execution order. A stage may only run once every stage before it has
#: succeeded or been skipped.
STAGE_ORDER = ["extract", "ocr", "normalize", "chunk", "embed", "index"]


@dataclass
class StageResult:
    """What a stage reports back.

    status is usually 'succeeded'. A stage returns:
      'skipped' when it had nothing to do (e.g. ocr on a document whose text
                layer passed triage everywhere)
      'held'    when it finished but a human must look before advancing - this
                is the review queue, and it deliberately blocks downstream work
    """
    status: str = "succeeded"
    metrics: dict = field(default_factory=dict)
    output_ref: str | None = None
    note: str | None = None


class Stage:
    name = "base"
    #: human-readable, shown in the admin stage table
    description = ""

    def run(self, doc) -> StageResult:
        """`doc` is the documents row as a dict. Raise to fail the stage;
        the runner records the traceback in stage_runs.error."""
        raise NotImplementedError


_REGISTRY: dict[str, Stage] = {}


def register(stage_cls):
    inst = stage_cls()
    if inst.name not in STAGE_ORDER:
        raise ValueError(f"{inst.name!r} is not in STAGE_ORDER")
    _REGISTRY[inst.name] = inst
    return stage_cls


def get(name) -> Stage:
    if name not in _REGISTRY:
        raise KeyError(
            f"stage {name!r} is not implemented yet. "
            f"implemented: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def implemented():
    return sorted(_REGISTRY)


def load_all():
    """Import the stage modules so their @register decorators run."""
    from . import (chunk, embed, extract, normalize, ocr,  # noqa: F401
                   placeholders)
    return implemented()
