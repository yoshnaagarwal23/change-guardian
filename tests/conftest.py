"""Shared fixtures. Tests run against the real client sample data (it is small)."""
from pathlib import Path

import pytest

from tools import TriageTools

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tools() -> TriageTools:
    return TriageTools.from_disk(ROOT / "data/processed", ROOT / "data/sops")


@pytest.fixture
def sample_sop_text() -> str:
    return (ROOT / "data/sops/SOP-004.md").read_text()
