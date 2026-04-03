"""Shared experiment configuration for the MIMIC-IV pipeline.

All scope parameters live here so every phase uses the same subset.
Adjust N_CONDITIONS for fast pilot runs (e.g. 5) vs full experiments (100).
"""

import os

# Number of top conditions (by frequency x comorbidity richness) to include.
# Controls scope across all pipeline phases.
N_CONDITIONS = int(os.environ.get('MIMIC_N_CONDITIONS', '5'))
