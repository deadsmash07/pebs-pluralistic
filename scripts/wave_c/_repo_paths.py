"""Output-path configuration for the wave_c experiments.

Results are written under RESULTS_ROOT (override with the
PEBS_RESULTS_DIR environment variable).
"""
import os
from pathlib import Path

RESULTS_ROOT = Path(os.environ.get("PEBS_RESULTS_DIR", "results"))
STANDALONE_RESULTS = RESULTS_ROOT
