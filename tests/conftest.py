"""Shared fixtures. Tests use a temporary data dir so they never touch data/."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def small_sim():
    from aiids.simulator import simulate
    return simulate(days=4, seed=3, scale=0.12)


@pytest.fixture(scope="session")
def small_bundle(small_sim):
    from aiids.models import train_bundle
    flows, _, _ = small_sim
    return train_bundle(flows[flows.split == "train"], flows[flows.split == "val"], log=lambda *a: None)


@pytest.fixture()
def tmp_data(tmp_path, monkeypatch):
    from aiids import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DATASET_DIR", tmp_path / "datasets")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.sqlite")
    config.ensure_dirs()
    return tmp_path
