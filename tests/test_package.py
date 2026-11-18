"""Packaging, teaching copy, and a boot of the Streamlit script."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import imagepro
from imagepro.data import MAX_ROTATION_DEGREES
from imagepro.explanations import CONCEPT_ORDER, CONCEPTS
from imagepro.training import DEFAULT_LEARNING_RATE, LEARNING_RATES, MIN_BATCH_SIZE

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONCEPTS = {
    "transfer_learning",
    "learning_rate",
    "batch_size",
    "epochs",
    "validation",
    "overfitting",
    "augmentation",
    "confidence",
    "seed",
}


def test_version_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == imagepro.__version__
    assert "John Amodeo" in project["project"]["authors"][0]["name"]


def test_license_and_changelog_name_the_release() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 John Amodeo" in license_text
    assert "Permission is hereby granted" in license_text
    assert "## 1.0.0" in changelog
    assert "Novel mode" in changelog
    assert "Augmentation" in changelog


def test_concepts_cover_the_teaching_goals() -> None:
    assert set(CONCEPT_ORDER) >= REQUIRED_CONCEPTS
    for key in CONCEPT_ORDER:
        concept = CONCEPTS[key]
        assert concept.title
        assert concept.summary
        assert len(concept.body) > 80
    assert f"{MAX_ROTATION_DEGREES:g}" in CONCEPTS["augmentation"].body
    assert str(MIN_BATCH_SIZE) in CONCEPTS["batch_size"].body
    assert DEFAULT_LEARNING_RATE in LEARNING_RATES
    assert all(rate > 0 for rate in LEARNING_RATES)


def test_readme_embeds_committed_screenshots() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# How Neural Networks Learn: An Image Classifier Lab\n")
    assert "actions/workflows/ci.yml/badge.svg" in readme
    images = re.findall(r"!\[[^\]]*\]\((docs/[^)]+)\)", readme)
    assert len(images) >= 5
    for relative in images:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 10_000


def test_app_script_boots() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=60)
    app.run()
    assert not app.exception
    titles = [title.value for title in app.title]
    assert "How Neural Networks Learn: An Image Classifier Lab" in titles
