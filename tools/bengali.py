"""Re-export of app.bengali, kept so the standalone tools keep their import path.

The canonical module is app/bengali.py: the ingestion pipeline depends on these
rules, and the app package must not import from tools/. One source of truth
matters here - if the probe and the pipeline ever disagree about what counts as
valid Bengali, the pipeline will index text the probe would have rejected.
"""
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from app.bengali import (  # noqa: E402,F401
    BENGALI_BLOCK,
    DEPENDENT_SIGNS,
    LEGACY_FONT_HINT,
    VIRAMA,
    bengali_ratio,
    is_bengali,
    looks_corrupt,
    normalize,
    orthographic_report,
)
