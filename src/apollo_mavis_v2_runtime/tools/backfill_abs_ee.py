"""CLI entry point: ``python -m apollo_mavis_v2_runtime.tools.backfill_abs_ee <dataset dir>``.

The implementation lives in :mod:`apollo_mavis_v2_runtime.recorder.backfill_abs_ee` (the
recorder package is the sanctioned parquet reader / writer - 14-dora §1 import confinement);
this module only re-exports it so the tool keeps its documented ``tools.`` address.
"""

from __future__ import annotations

from ..recorder.backfill_abs_ee import *  # noqa: F403
from ..recorder.backfill_abs_ee import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
