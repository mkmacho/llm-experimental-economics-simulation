from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_replication.config import load_config, project_root
import synthetic_replication.papers.experience_based_discrimination as ebd
from synthetic_replication.papers.experience_based_discrimination import (
    HeuristicAgent,
    _load_prepared,
    _paths,
    _profile_model,
    _select_audit_profiles,
    _sample_profiles,
    _sample_group_b_draws,
    _simulate_one_employer,
    evaluate_simulations,
    prepare_paper,
    simulate_paper,
)
from synthetic_replication.types import Action


RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw/experience-based-discrimination"
pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "data.dta").is_file() or not (RAW_DIR / "dataworker.dta").is_file(),
    reason="official source data are not bundled; see data/README.md",
)


def test_group_b_draws_are_without_replacement() -> None:
    worker_pool = pd.DataFrame(
        {
            "group": ["g"] * 30 + ["b"] * 30,
            "canonical_group": ["group_a"] * 30 + ["group_b"] * 30,
            "productivity": list(range(30)) + list(range(100, 130)),
        }
    )
    draws = _sample_group_b_draws(worker_pool, 15, rng=np.random.default_rng(1))
    assert len(draws) == 15
    assert len(set(draws)) == 15


def test_control_treatment_forces_group_b() -> None:
    config = load_config()
    prepare_paper(config, project_root())
    paths = _paths(config, project_root())
    spec, profiles, worker_pool, _observed, _targets = _load_prepared(paths)
    profile_row = profiles.loc[profiles["treatment"] == "control"].iloc[0]
    profile = _profile_model(profile_row)
    rows, _logs = _simulate_one_employer(
        spec=spec,
        profile=profile,
        agent=HeuristicAgent("always_A"),
        worker_pool=worker_pool,
        replicate_id=0,
        simulation_employer_id="test-control",
        replicate_seed=123,
        label_variant="canonical",
        fixed_a_productivity=spec.group_a_known_productivity,
    )
    assert {row["action"] for row in rows} == {Action.HIRE_B.value}


def test_sample_profiles_can_limit_per_treatment() -> None:
    config = load_config()
    prepare_paper(config, project_root())
    paths = _paths(config, project_root())
    _spec, profiles, _worker_pool, _observed, _targets = _load_prepared(paths)
    sampled = _sample_profiles(profiles, np.random.default_rng(7), sample_size_per_treatment=1)

    counts = sampled.groupby("treatment")["simulation_profile_index"].count().to_dict()
    assert counts == {"baseline": 1, "control": 1}


def test_select_audit_profiles_respects_total_sample_size() -> None:
    config = load_config()
    prepare_paper(config, project_root())
    paths = _paths(config, project_root())
    _spec, profiles, _worker_pool, _observed, _targets = _load_prepared(paths)

    sampled = _select_audit_profiles(profiles, np.random.default_rng(11), audit_sample_size=24)
    counts = sampled["treatment"].value_counts().sort_index().to_dict()

    assert len(sampled) == 24
    assert counts == {"baseline": 12, "control": 12}
    assert sampled["employer_key"].is_unique


def test_select_audit_profiles_smoke_mode_keeps_one_per_treatment() -> None:
    config = load_config()
    prepare_paper(config, project_root())
    paths = _paths(config, project_root())
    _spec, profiles, _worker_pool, _observed, _targets = _load_prepared(paths)

    sampled = _select_audit_profiles(profiles, np.random.default_rng(11), audit_sample_size=0)
    counts = sampled["treatment"].value_counts().sort_index().to_dict()

    assert len(sampled) == 2
    assert counts == {"baseline": 1, "control": 1}


def test_end_to_end_baseline_only(tmp_path) -> None:
    config = load_config()
    config.runtime.replicates = 1
    config.runtime.bootstrap_iterations = 25
    config.runtime.audit_replicates = 1
    config.runtime.audit_sample_size = 6

    prepare_paper(config, project_root())
    simulate_manifest = simulate_paper(config, project_root(), replicates=1, baseline_only=True)
    evaluation_summary = evaluate_simulations(config, project_root())

    assert simulate_manifest["replicates"] == 1
    assert evaluation_summary["comparison_report"].endswith("outputs/experience-based-discrimination/evaluation/comparison.md")


def test_evaluate_reuses_observed_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    config.runtime.replicates = 1
    config.runtime.bootstrap_iterations = 10
    config.runtime.audit_replicates = 1
    config.runtime.audit_sample_size = 6

    prepare_paper(config, project_root())
    simulate_paper(config, project_root(), replicates=1, baseline_only=True)

    calls = {"count": 0}
    original = ebd._bootstrap_observed_effects

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ebd, "_bootstrap_observed_effects", counted)
    evaluate_simulations(config, project_root())

    assert calls["count"] == 1


def test_smoke_test_forces_minimal_runtime() -> None:
    config = load_config()
    config.runtime.replicates = 20
    config.runtime.audit_replicates = 2
    config.runtime.audit_sample_size = 24

    prepare_paper(config, project_root())
    simulate_manifest = simulate_paper(config, project_root(), replicates=5, baseline_only=True, smoke_test=True)
    simulation_frame = pd.read_csv(project_root() / simulate_manifest["simulation_records"])

    assert simulate_manifest["replicates"] == 1
    assert simulate_manifest["smoke_test"] is True
    assert simulate_manifest["effective_runtime"]["audit_replicates"] == 1
    assert simulate_manifest["effective_runtime"]["audit_sample_size"] == 0
    assert simulate_manifest["effective_runtime"]["sample_size_per_treatment"] == 1
    counts = simulation_frame.groupby("agent_name")["simulation_employer_id"].nunique().to_dict()
    assert counts == {"always_A": 2, "always_B": 2, "uniform_random": 2}
