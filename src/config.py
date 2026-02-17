"""
Configuration for the PLGA pipeline and reproduction scripts.

Paths are relative to the repository root. Set DATA_DIR or OUTPUT_DIR
via environment variables to override (e.g. for different machines).
"""

import os
from pathlib import Path

# Repository root (directory containing config.py)
REPO_ROOT = Path(__file__).resolve().parent

# Data and output directories (relative to repo root)
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "outputs"))

# Input filenames (expected under DATA_DIR)
RAW_DATASET = "mp_dataset_processed.xlsx"
INITIAL_DATASET = "mp_dataset_initial.xlsx"

# Reproducibility: fixed seed for all random operations (numpy, random, sklearn, xgboost)
RANDOM_SEED = 42


def set_seeds() -> None:
    """Set global random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# Feature column list (used in pipeline and validation)
FEATURE_COLS = [
    "Drug MW", "Drug LogP", "Drug TPSA", "MolLogP", "TPSA", "ExactMolWt",
    "NumHDonors", "NumHAcceptors", "RotatableBonds",
    "Polymer MW", "LA_GA_numeric", "Hydrophilicity_Index",
    "Particle Size", "Drug Loading Capacity", "Drug Encapsulation Efficiency",
]
