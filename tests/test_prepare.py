from __future__ import annotations

import json

import pandas as pd
import pytest

from synthetic_replication.config import load_config, project_root
from synthetic_replication.papers import prepare_paper


def require_raw_inputs() -> None:
    raw_dir = project_root() / "data/raw/experience-based-discrimination"
    required = [raw_dir / "data.dta", raw_dir / "dataworker.dta"]
    if not all(path.is_file() for path in required):
        pytest.skip("official source data are not bundled; see data/README.md")


def test_prepare_outputs_and_counts() -> None:
    require_raw_inputs()
    config = load_config()
    manifest = prepare_paper(config, project_root())
    assert manifest["sample_sizes"] == {
        "baseline_employers": 297,
        "control_employers": 135,
        "rounds_per_employer": 15,
    }

    profiles = pd.read_csv(project_root() / "data/normalized/experience-based-discrimination/employer_profiles.csv")
    outcomes = pd.read_csv(project_root() / "data/normalized/experience-based-discrimination/observed_core_outcomes.csv")

    assert profiles["treatment"].value_counts().to_dict() == {"baseline": 297, "control": 135}
    assert outcomes["employer_key"].nunique() == 432
    assert len(outcomes) == 432 * 15


def test_prepare_reproduces_reference_means() -> None:
    require_raw_inputs()
    config = load_config()
    prepare_paper(config, project_root())
    targets = json.loads(
        (project_root() / "data/normalized/experience-based-discrimination/observed_targets.json").read_text(encoding="utf-8")
    )

    assert targets["paper_level_reference"]["figure3_period15_final_belief_mean_nonzero_b_hires"]["baseline"] == 8.624812
    assert targets["paper_level_reference"]["figure3_period15_final_belief_mean_nonzero_b_hires"]["control"] == 9.09037
    assert targets["paper_level_reference"]["control_minus_baseline_final_belief_nonzero_b_hires"] == 0.465558
