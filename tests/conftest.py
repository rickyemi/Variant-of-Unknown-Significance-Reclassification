"""Shared pytest fixtures."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402


@pytest.fixture(scope="session")
def splits():
    """Train / test / VUS frames, building them from the raw CSV if needed."""
    from src.data import make_dataset
    if not cfg.TRAIN_FILE.exists():
        make_dataset.main()
    return {n: make_dataset.load_split(n) for n in ["train", "test", "vus"]}


@pytest.fixture(scope="session")
def trained():
    """The three saved model pipelines and their metadata (skip if not trained)."""
    if not all(p.exists() for p in cfg.MODEL_FILES.values()):
        pytest.skip("models not trained yet - run `make train`")
    from src.models.train_model import load_models
    return load_models()
