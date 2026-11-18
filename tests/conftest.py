"""Test defaults: headless plots, and no pretrained downloads."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


@pytest.fixture(autouse=True)
def _block_pretrained_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a weight file would be fetched.

    Model tests pass ``weights=None`` or replace the constructor. This fixture
    is the backstop so a forgotten argument cannot reach the network.
    """

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not download pretrained weights")

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", blocked)
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", blocked)
