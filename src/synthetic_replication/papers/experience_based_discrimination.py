from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from synthetic_replication.config import PipelineConfig
from synthetic_replication.llm_clients import AnthropicMessagesClient, OpenAIResponsesClient
from synthetic_replication.render import (
    render_decision_prompt,
    render_recall_probe,
    render_update_prompt,
)
from synthetic_replication.types import (
    Action,
    BeliefUpdateResponse,
    DecisionResponse,
    EmployerProfile,
    PaperRecallProbeResponse,
    PaperSpec,
    Treatment,
    TreatmentSpec,
)
from synthetic_replication.utils import (
    clamp,
    ensure_directory,
    format_markdown_table,
    stable_seed,
    write_json,
    write_jsonl,
)

PAPER_ID = "experience-based-discrimination"


@dataclass
class RunPaths:
    raw_dir: Path
    normalized_dir: Path
    output_dir: Path


def build_paper_spec(config: PipelineConfig) -> PaperSpec:
    return PaperSpec(
        paper_id=PAPER_ID,
        paper_title="Experience-based Discrimination",
        rounds=15,
        group_a_known_productivity=9.0,
        group_b_minority_share=0.25,
        group_a_majority_share=0.75,
        belief_min=1.0,
        belief_max=18.0,
        positive_experience_cutoff=9.0,
        group_b_without_replacement=True,
        baseline_treatment=TreatmentSpec(
            name=Treatment.BASELINE,
            description="Choose between the fixed group-A option and the uncertain group-B option each round.",
            allowed_actions=[Action.HIRE_A, Action.HIRE_B],
        ),
        control_treatment=TreatmentSpec(
            name=Treatment.CONTROL,
            description="Forced-exposure treatment: each round hires group B.",
            allowed_actions=[Action.HIRE_B],
            forced_action=Action.HIRE_B,
        ),
        version_pins=config.versions.model_dump(),
        notes=[
            "v1 covers the Baseline and Control treatments only.",
            "Worker-side productivity is resampled empirically without replacement within simulated employers.",
            "Employer heterogeneity is represented by empirical personas sampled from the cleaned core treatments.",
        ],
    )


def _paths(config: PipelineConfig, project_root: Path) -> RunPaths:
    raw_dir = project_root / config.paths.raw_root / PAPER_ID
    normalized_dir = project_root / config.paths.normalized_root / PAPER_ID
    output_dir = project_root / config.paths.output_root / PAPER_ID
    ensure_directory(normalized_dir)
    ensure_directory(output_dir)
    return RunPaths(raw_dir=raw_dir, normalized_dir=normalized_dir, output_dir=output_dir)


def _load_raw_tables(paths: RunPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = paths.raw_dir / "data.dta"
    worker_path = paths.raw_dir / "dataworker.dta"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw employer data: {raw_path}")
    if not worker_path.exists():
        raise FileNotFoundError(f"Missing raw worker data: {worker_path}")
    raw = pd.read_stata(raw_path)
    workers = pd.read_stata(worker_path)
    return raw, workers


def _group_key(frame: pd.DataFrame) -> pd.Series:
    numeric_id = frame["id"].round().astype(int)
    return frame["empgroup"].astype(str) + ":" + numeric_id.astype(str)


def _groupwise_max(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("employer_key")[column].transform("max")


def _clean_employer_data(raw: pd.DataFrame, spec: PaperSpec) -> pd.DataFrame:
    frame = raw.copy()
    frame["id"] = frame["id"].round().astype(int)
    frame["round"] = frame["round"].astype(int)
    frame["employer_key"] = _group_key(frame)
    frame = frame.sort_values(["employer_key", "round"]).reset_index(drop=True)

    uncertain_labels = {
        "Purple - ? Credits",
        "Gray - ? Credits",
        "Green - ? Credits",
        "Female - ? Credits",
    }
    frame["purple"] = 0
    frame.loc[frame["playerselect"].isin(uncertain_labels), "purple"] = 1
    frame.loc[frame["empgroup"] == "NA", "purple"] = 1

    frame["belowmean"] = np.where(frame["purple"].eq(1), (frame["playerdraw"] < 9).astype(float), np.nan)
    frame["abovemean"] = np.where(frame["purple"].eq(1), (frame["playerdraw"] > 9).astype(float), np.nan)
    frame["purpcount"] = frame.groupby("employer_key")["purple"].cumsum()
    frame["totpurp"] = frame.groupby("employer_key")["purpcount"].transform("max")

    first_b = frame["purpcount"].eq(1) & frame["purple"].eq(1)
    frame["pos1"] = np.where(first_b & frame["playerdraw"].gt(9), 1.0, np.where(first_b, 0.0, np.nan))
    frame["neg1"] = np.where(first_b & frame["playerdraw"].lt(9), 1.0, np.where(first_b, 0.0, np.nan))
    frame["p1"] = _groupwise_max(frame, "pos1")
    frame["n1"] = _groupwise_max(frame, "neg1")

    first_signal_specs = {
        "n18": np.where(first_b & frame["playerdraw"].eq(8), 1.0, np.where(first_b & frame["playerdraw"].ge(9), 0.0, np.nan)),
        "n17": np.where(first_b & frame["playerdraw"].eq(7), 1.0, np.where(first_b & frame["playerdraw"].ge(8), 0.0, np.nan)),
        "n16": np.where(first_b & frame["playerdraw"].le(6), 1.0, np.where(first_b & frame["playerdraw"].ge(7), 0.0, np.nan)),
    }
    for name, values in first_signal_specs.items():
        frame[name] = values
        frame[name] = _groupwise_max(frame, name)

    rescale_mask = frame["playerbeliefs"] > 1000
    frame.loc[rescale_mask, "playerbeliefs"] = frame.loc[rescale_mask, "playerbeliefs"] / 220

    invalid_upper = ((frame["playerbeliefs"] > 18) & (frame["empgroup"] != "AB")) | (
        (frame["playerbeliefs"] > 18) & (frame["round"] == spec.rounds) & (frame["empgroup"] == "AB")
    )
    frame = frame.loc[~invalid_upper].copy()
    frame = frame.loc[~(frame["playerbeliefs"] < 1)].copy()

    frame["avgdraw"] = frame["playerdraw"].where(frame["purple"].eq(1))
    frame["avgdraw"] = frame.groupby("employer_key")["avgdraw"].transform("mean")
    frame["o7"] = frame["avgdraw"] + 7
    frame["u7"] = frame["avgdraw"] - 7
    frame["upsd"] = frame.groupby("employer_key")["playerbeliefs"].transform("std")
    frame["prev_belief"] = frame.groupby("employer_key")["playerbeliefs"].shift(1)
    frame["wrongdirection"] = 0.0
    mask = frame["purple"].eq(1) & frame["prev_belief"].notna()
    frame.loc[
        mask & (frame["prev_belief"] > frame["playerbeliefs"]) & (frame["playerdraw"] > frame["prev_belief"]),
        "wrongdirection",
    ] = 1.0
    frame.loc[
        mask & (frame["prev_belief"] < frame["playerbeliefs"]) & (frame["playerdraw"] < frame["prev_belief"]),
        "wrongdirection",
    ] = 1.0
    frame["meanwrong"] = frame.groupby("employer_key")["wrongdirection"].transform("mean")
    frame["nround"] = frame.groupby("employer_key")["round"].transform("count")

    frame = frame.loc[~((frame["round"] == spec.rounds) & frame["totpurp"].ne(0) & (frame["playerbeliefs"] > frame["o7"]))].copy()
    frame = frame.loc[~((frame["round"] == spec.rounds) & frame["totpurp"].ne(0) & (frame["playerbeliefs"] < frame["u7"]))].copy()
    frame = frame.loc[~((frame["upsd"] == 0) & frame["totpurp"].ge(1))].copy()
    frame = frame.loc[~(frame["meanwrong"] > 0.75)].copy()
    frame = frame.loc[
        ~(
            (frame["nround"] != spec.rounds)
            & frame["mf"].isna()
            & frame["exp"].isna()
            & frame["learn"].isna()
            & frame["equal"].isna()
        )
    ].copy()

    frame = frame.sort_values(["employer_key", "round"]).reset_index(drop=True)
    frame["groupswitch"] = frame.groupby("employer_key")["purple"].diff().abs()
    frame.loc[frame.groupby("employer_key").cumcount().eq(0), "groupswitch"] = np.nan
    frame["finalbel"] = frame.groupby("employer_key")["playerbeliefs"].transform("last")
    frame["remh"] = frame["totpurp"] - frame["purpcount"]
    frame["playerprior_num"] = pd.to_numeric(frame["playerprior"], errors="coerce")
    frame["prior"] = frame.groupby("employer_key")["playerprior_num"].transform("mean")
    frame["playerswitch_num"] = pd.to_numeric(frame["playerswitch"], errors="coerce")
    frame["amb"] = frame.groupby("employer_key")["playerswitch_num"].transform("mean")

    complete_counts = frame.groupby("employer_key")["round"].transform("size")
    complete_unique_rounds = frame.groupby("employer_key")["round"].transform("nunique")
    complete_max_round = frame.groupby("employer_key")["round"].transform("max")
    complete_min_round = frame.groupby("employer_key")["round"].transform("min")
    frame["complete_rounds"] = (
        (complete_counts == spec.rounds)
        & (complete_unique_rounds == spec.rounds)
        & (complete_max_round == spec.rounds)
        & (complete_min_round == 1)
    )

    core = frame.loc[
        (frame["learn"] != 1)
        & (frame["equal"] != 1)
        & (frame["mf"] != 1)
        & (frame["exp"] != 1)
        & frame["empgroup"].isin(["A", "NA"])
    ].copy()
    core = core.loc[core["complete_rounds"]].copy()
    core["treatment"] = core["empgroup"].map({"A": Treatment.BASELINE.value, "NA": Treatment.CONTROL.value})
    core["hired_b"] = core["purple"].astype(int)
    core["action"] = np.where(core["hired_b"].eq(1), Action.HIRE_B.value, Action.HIRE_A.value)
    core["experience_sign"] = np.where(
        core["hired_b"].eq(1) & core["playerdraw"].gt(spec.positive_experience_cutoff),
        "positive",
        np.where(
            core["hired_b"].eq(1) & core["playerdraw"].lt(spec.positive_experience_cutoff),
            "negative",
            np.where(core["hired_b"].eq(1), "neutral", ""),
        ),
    )
    core["next_hired_b"] = core.groupby("employer_key")["hired_b"].shift(-1)
    core["source_numeric_id"] = core["id"].astype(int)
    return core


def _prepare_profiles(core: pd.DataFrame, spec: PaperSpec) -> pd.DataFrame:
    profile_cols = [
        "employer_key",
        "source_numeric_id",
        "empgroup",
        "treatment",
        "playerage",
        "prior",
        "amb",
        "totpurp",
        "finalbel",
    ]
    profiles = core.sort_values(["employer_key", "round"]).groupby("employer_key", as_index=False).first()[profile_cols].copy()
    profiles["prior_b_raw"] = profiles["prior"].astype(float)
    profiles["prior_b_prompt"] = profiles["prior_b_raw"].clip(spec.belief_min, spec.belief_max)
    profiles["prior_b_was_clipped"] = profiles["prior_b_raw"] != profiles["prior_b_prompt"]
    profiles["treatment"] = profiles["treatment"].map(
        {
            Treatment.BASELINE.value: Treatment.BASELINE.value,
            Treatment.CONTROL.value: Treatment.CONTROL.value,
        }
    )
    profiles = profiles.rename(
        columns={
            "empgroup": "source_group_code",
            "playerage": "age_years",
            "amb": "ambiguity_switch",
            "totpurp": "observed_total_b_hires",
            "finalbel": "observed_final_belief",
        }
    )
    return profiles[
        [
            "employer_key",
            "source_numeric_id",
            "source_group_code",
            "treatment",
            "age_years",
            "prior_b_raw",
            "prior_b_prompt",
            "prior_b_was_clipped",
            "ambiguity_switch",
            "observed_total_b_hires",
            "observed_final_belief",
        ]
    ].copy()


def _prepare_worker_pool(workers: pd.DataFrame) -> pd.DataFrame:
    frame = workers.copy()
    counts = frame["group"].value_counts()
    minority_group = counts.idxmin()
    majority_group = counts.idxmax()
    frame["canonical_group"] = frame["group"].map(
        {
            majority_group: "group_a",
            minority_group: "group_b",
        }
    )
    frame["productivity"] = frame["npuz"].astype(float)
    return frame[["group", "canonical_group", "productivity"]].copy()


def _prepare_observed_outcomes(core: pd.DataFrame) -> pd.DataFrame:
    frame = core.copy()
    frame = frame.rename(
        columns={
            "playerdraw": "observed_productivity",
            "playerbeliefs": "belief_after_round",
            "prior": "prior_b_raw",
            "purpcount": "cumulative_b_hires",
            "totpurp": "total_b_hires",
            "finalbel": "final_belief",
        }
    )
    frame["prior_b_prompt"] = frame["prior_b_raw"].clip(1, 18)
    return frame[
        [
            "employer_key",
            "source_numeric_id",
            "empgroup",
            "treatment",
            "round",
            "action",
            "hired_b",
            "observed_productivity",
            "belief_after_round",
            "prior_b_raw",
            "prior_b_prompt",
            "cumulative_b_hires",
            "total_b_hires",
            "remh",
            "experience_sign",
            "next_hired_b",
            "p1",
            "n1",
            "groupswitch",
            "final_belief",
            "playerselect",
        ]
    ].copy()


def _observed_targets(profiles: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    round15 = outcomes.loc[outcomes["round"] == 15].copy()
    nonzero = round15.loc[round15["total_b_hires"] > 0].copy()
    baseline_follow = outcomes.loc[(outcomes["treatment"] == Treatment.BASELINE.value) & (outcomes["round"] < 15) & (outcomes["hired_b"] == 1)].copy()
    first_b = outcomes.loc[(outcomes["treatment"] == Treatment.BASELINE.value) & (outcomes["cumulative_b_hires"] == 1)].copy()
    return {
        "sample_sizes": {
            "baseline_employers": int((profiles["treatment"] == Treatment.BASELINE.value).sum()),
            "control_employers": int((profiles["treatment"] == Treatment.CONTROL.value).sum()),
            "rounds_per_employer": int(outcomes.groupby("employer_key")["round"].count().iloc[0]),
        },
        "paper_level_reference": {
            "figure3_period15_final_belief_mean_nonzero_b_hires": {
                "baseline": round(nonzero.loc[nonzero["treatment"] == Treatment.BASELINE.value, "belief_after_round"].mean(), 6),
                "control": round(nonzero.loc[nonzero["treatment"] == Treatment.CONTROL.value, "belief_after_round"].mean(), 6),
            },
            "control_minus_baseline_final_belief_nonzero_b_hires": round(
                nonzero.loc[nonzero["treatment"] == Treatment.CONTROL.value, "belief_after_round"].mean()
                - nonzero.loc[nonzero["treatment"] == Treatment.BASELINE.value, "belief_after_round"].mean(),
                6,
            ),
        },
        "micro_targets": {
            "initial_prior_mean": {
                treatment: round(group["prior_b_prompt"].mean(), 6)
                for treatment, group in profiles.groupby("treatment")
            },
            "next_round_hire_rate_after_positive": round(
                baseline_follow.loc[baseline_follow["experience_sign"] == "positive", "next_hired_b"].mean(),
                6,
            ),
            "next_round_hire_rate_after_negative": round(
                baseline_follow.loc[baseline_follow["experience_sign"] == "negative", "next_hired_b"].mean(),
                6,
            ),
            "future_b_hires_after_positive_first_experience": round(
                first_b.loc[first_b["experience_sign"] == "positive", "remh"].mean(),
                6,
            ),
            "future_b_hires_after_negative_first_experience": round(
                first_b.loc[first_b["experience_sign"] == "negative", "remh"].mean(),
                6,
            ),
        },
    }


def prepare_paper(config: PipelineConfig, project_root: Path) -> dict[str, Any]:
    spec = build_paper_spec(config)
    paths = _paths(config, project_root)
    raw, workers = _load_raw_tables(paths)
    core = _clean_employer_data(raw, spec)
    profiles = _prepare_profiles(core, spec)
    worker_pool = _prepare_worker_pool(workers)
    observed_outcomes = _prepare_observed_outcomes(core)
    targets = _observed_targets(profiles, observed_outcomes)

    profiles_path = paths.normalized_dir / "employer_profiles.csv"
    worker_path = paths.normalized_dir / "worker_pool_empirical.csv"
    outcomes_path = paths.normalized_dir / "observed_core_outcomes.csv"
    spec_path = paths.normalized_dir / "paper_spec.json"
    targets_path = paths.normalized_dir / "observed_targets.json"
    manifest_path = paths.normalized_dir / "prepare_manifest.json"

    profiles.to_csv(profiles_path, index=False)
    worker_pool.to_csv(worker_path, index=False)
    observed_outcomes.to_csv(outcomes_path, index=False)
    write_json(spec_path, spec.model_dump(mode="json"))
    write_json(targets_path, targets)

    manifest = {
        "paper_id": PAPER_ID,
        "raw_files": ["data.dta", "dataworker.dta"],
        "normalized_files": [
            str(profiles_path.relative_to(project_root)),
            str(worker_path.relative_to(project_root)),
            str(outcomes_path.relative_to(project_root)),
            str(spec_path.relative_to(project_root)),
            str(targets_path.relative_to(project_root)),
        ],
        "sample_sizes": targets["sample_sizes"],
        "version_pins": config.versions.model_dump(),
        "notes": [
            "Normalized core data are rebuilt from data.dta using the published cleaning logic, with employer_key = empgroup:id to avoid cross-treatment id collisions.",
            "Only complete 15-round Baseline and Control employers are retained for the v1 simulation target.",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_prepared(paths: RunPaths) -> tuple[PaperSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = PaperSpec.model_validate(json.loads((paths.normalized_dir / "paper_spec.json").read_text(encoding="utf-8")))
    profiles = pd.read_csv(paths.normalized_dir / "employer_profiles.csv")
    worker_pool = pd.read_csv(paths.normalized_dir / "worker_pool_empirical.csv")
    observed = pd.read_csv(paths.normalized_dir / "observed_core_outcomes.csv")
    targets = json.loads((paths.normalized_dir / "observed_targets.json").read_text(encoding="utf-8"))
    return spec, profiles, worker_pool, observed, targets


def _profile_model(row: pd.Series) -> EmployerProfile:
    return EmployerProfile(
        employer_key=str(row["employer_key"]),
        source_numeric_id=int(row["source_numeric_id"]),
        source_group_code=str(row["source_group_code"]),
        treatment=Treatment(str(row["treatment"])),
        age_years=float(row["age_years"]) if pd.notna(row["age_years"]) else None,
        prior_b_raw=float(row["prior_b_raw"]),
        prior_b_prompt=float(row["prior_b_prompt"]),
        prior_b_was_clipped=bool(row["prior_b_was_clipped"]),
        ambiguity_switch=float(row["ambiguity_switch"]) if pd.notna(row["ambiguity_switch"]) else None,
        observed_total_b_hires=int(row["observed_total_b_hires"]),
        observed_final_belief=float(row["observed_final_belief"]),
    )


class BaseAgent:
    name: str
    provider: str

    def decide(
        self,
        *,
        spec: PaperSpec,
        profile: EmployerProfile,
        round_number: int,
        current_belief: float,
        history: list[dict[str, object]],
        fixed_a_productivity: float,
        label_variant: str,
        rng: np.random.Generator,
    ) -> tuple[DecisionResponse, dict[str, Any]]:
        raise NotImplementedError

    def update_belief(
        self,
        *,
        spec: PaperSpec,
        current_belief: float,
        observed_productivity: float,
        previous_b_hires: int,
        rng: np.random.Generator,
    ) -> tuple[BeliefUpdateResponse, dict[str, Any]]:
        raise NotImplementedError

    def run_recall_probe(self) -> dict[str, Any]:
        return {"skipped": True, "reason": "not_supported"}


class HeuristicAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        self.name = name
        self.provider = "baseline"

    def decide(
        self,
        *,
        spec: PaperSpec,
        profile: EmployerProfile,
        round_number: int,
        current_belief: float,
        history: list[dict[str, object]],
        fixed_a_productivity: float,
        label_variant: str,
        rng: np.random.Generator,
    ) -> tuple[DecisionResponse, dict[str, Any]]:
        prompt = render_decision_prompt(
            spec=spec,
            profile=profile,
            treatment=profile.treatment,
            round_number=round_number,
            current_belief=current_belief,
            history=history,
            fixed_a_productivity=fixed_a_productivity,
            label_variant=label_variant,
        )
        if profile.treatment == Treatment.CONTROL:
            action = Action.HIRE_B
        elif self.name == "always_A":
            action = Action.HIRE_A
        elif self.name == "always_B":
            action = Action.HIRE_B
        elif self.name == "uniform_random":
            action = Action.HIRE_B if rng.random() < 0.5 else Action.HIRE_A
        else:  # pragma: no cover - guarded by construction
            raise ValueError(f"Unsupported heuristic agent {self.name}")
        response = DecisionResponse(action=action, belief_b=float(current_belief))
        return response, {
            "stage": "decision",
            "prompt_text": prompt,
            "raw_response": response.model_dump_json(),
            "parsed_response": response.model_dump(mode="json"),
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd_estimate": 0.0,
        }

    def update_belief(
        self,
        *,
        spec: PaperSpec,
        current_belief: float,
        observed_productivity: float,
        previous_b_hires: int,
        rng: np.random.Generator,
    ) -> tuple[BeliefUpdateResponse, dict[str, Any]]:
        prompt = render_update_prompt(
            spec=spec,
            observed_productivity=observed_productivity,
            current_belief=current_belief,
        )
        learning_rate = 1.0 / (previous_b_hires + 2.0)
        posterior = clamp(current_belief + learning_rate * (observed_productivity - current_belief), 1.0, 18.0)
        response = BeliefUpdateResponse(belief_b=round(float(posterior), 4))
        return response, {
            "stage": "update",
            "prompt_text": prompt,
            "raw_response": response.model_dump_json(),
            "parsed_response": response.model_dump(mode="json"),
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd_estimate": 0.0,
        }


def _normalize_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty model response")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        candidate = text[start : end + 1]
        json.loads(candidate)
        return candidate


def _preview_text(raw_text: str, limit: int = 240) -> str:
    if not raw_text:
        return ""
    flattened = " ".join(raw_text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


def _anthropic_prefill_for_schema(schema: Any) -> str | None:
    if schema is DecisionResponse:
        return '{"action":"'
    if schema is BeliefUpdateResponse:
        return '{"belief_b":'
    if schema is PaperRecallProbeResponse:
        return '{"recognized":'
    return None


def _parse_model_json_response(
    *,
    raw_text: str,
    stage: str,
    provider: str,
    model_id: str,
    response_meta: dict[str, Any],
    schema: Any,
) -> Any:
    try:
        return schema.model_validate_json(_normalize_json_response(raw_text))
    except Exception as exc:
        debug_meta = {key: value for key, value in response_meta.items() if value not in (None, [], {})}
        raise RuntimeError(
            f"{provider} model '{model_id}' returned invalid JSON during {stage}. "
            f"raw_text_length={len(raw_text or '')}; "
            f"raw_text_preview={_preview_text(raw_text)!r}; "
            f"response_meta={json.dumps(debug_meta, sort_keys=True)}"
        ) from exc


class LLMAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        provider: str,
        model_id: str,
        max_output_tokens: int,
        cost_model: Any,
        api_key_env: str | None = None,
        timeout_seconds: int = 90,
        max_retries: int = 4,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 12.0,
    ) -> None:
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.cost_model = cost_model
        if provider == "openai":
            self.client = OpenAIResponsesClient(
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                initial_backoff_seconds=initial_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
        elif provider == "anthropic":
            self.client = AnthropicMessagesClient(
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                initial_backoff_seconds=initial_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    def is_available(self) -> bool:
        return self.client.is_configured and bool(self.model_id)

    def _request(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema: Any | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        assistant_prefill = _anthropic_prefill_for_schema(schema) if self.provider == "anthropic" else None
        payload = self.client.create_json_response(
            model=self.model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=self.max_output_tokens,
            assistant_prefill=assistant_prefill,
        )
        raw_text = self.client.extract_output_text(payload)
        if assistant_prefill:
            raw_text = assistant_prefill + raw_text
        usage = self.client.extract_usage(payload)
        response_meta = self.client.describe_payload(payload)
        return raw_text, {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost_usd_estimate": self.cost_model.cost_estimate(usage["input_tokens"], usage["output_tokens"]),
        }, response_meta

    def decide(
        self,
        *,
        spec: PaperSpec,
        profile: EmployerProfile,
        round_number: int,
        current_belief: float,
        history: list[dict[str, object]],
        fixed_a_productivity: float,
        label_variant: str,
        rng: np.random.Generator,
    ) -> tuple[DecisionResponse, dict[str, Any]]:
        prompt = render_decision_prompt(
            spec=spec,
            profile=profile,
            treatment=profile.treatment,
            round_number=round_number,
            current_belief=current_belief,
            history=history,
            fixed_a_productivity=fixed_a_productivity,
            label_variant=label_variant,
        )
        system_prompt = "\n".join(
            [
                "You are participating in an economic decision task.",
                "Use only the information on the task card.",
                "Do not mention papers, authors, journals, archives, or outside knowledge.",
                "Do not provide analysis, explanations, or markdown.",
                "Return only valid JSON.",
            ]
        )
        if self.provider == "anthropic":
            system_prompt = "\n".join(
                [
                    system_prompt,
                    'Continue the JSON object started by the assistant message.',
                    'Return only the remaining characters needed to complete that JSON object.',
                ]
            )
        raw_text, usage, response_meta = self._request(system_prompt, prompt, schema=DecisionResponse)
        parsed = _parse_model_json_response(
            raw_text=raw_text,
            stage="decision",
            provider=self.provider,
            model_id=self.model_id,
            response_meta=response_meta,
            schema=DecisionResponse,
        )
        if profile.treatment == Treatment.CONTROL:
            parsed = DecisionResponse(action=Action.HIRE_B, belief_b=parsed.belief_b)
        return parsed, {
            "stage": "decision",
            "prompt_text": prompt,
            "raw_response": raw_text,
            "parsed_response": parsed.model_dump(mode="json"),
            **usage,
        }

    def update_belief(
        self,
        *,
        spec: PaperSpec,
        current_belief: float,
        observed_productivity: float,
        previous_b_hires: int,
        rng: np.random.Generator,
    ) -> tuple[BeliefUpdateResponse, dict[str, Any]]:
        prompt = render_update_prompt(
            spec=spec,
            observed_productivity=observed_productivity,
            current_belief=current_belief,
        )
        system_prompt = "\n".join(
            [
                "You are participating in an economic decision task.",
                "Use only the information on the outcome card.",
                "Do not mention papers, authors, journals, archives, or outside knowledge.",
                "Do not provide analysis, explanations, or markdown.",
                "Return only valid JSON.",
            ]
        )
        if self.provider == "anthropic":
            system_prompt = "\n".join(
                [
                    system_prompt,
                    'Continue the JSON object started by the assistant message.',
                    'Return only the remaining characters needed to complete that JSON object.',
                ]
            )
        raw_text, usage, response_meta = self._request(system_prompt, prompt, schema=BeliefUpdateResponse)
        parsed = _parse_model_json_response(
            raw_text=raw_text,
            stage="belief update",
            provider=self.provider,
            model_id=self.model_id,
            response_meta=response_meta,
            schema=BeliefUpdateResponse,
        )
        return parsed, {
            "stage": "update",
            "prompt_text": prompt,
            "raw_response": raw_text,
            "parsed_response": parsed.model_dump(mode="json"),
            **usage,
        }

    def run_recall_probe(self) -> dict[str, Any]:
        prompt = render_recall_probe()
        system_prompt = "\n".join(
            [
                "Answer the recognition probe honestly.",
                "Do not provide analysis, explanations, or markdown.",
                "Return only valid JSON.",
            ]
        )
        if self.provider == "anthropic":
            system_prompt = "\n".join(
                [
                    system_prompt,
                    'Continue the JSON object started by the assistant message.',
                    'Return only the remaining characters needed to complete that JSON object.',
                ]
            )
        raw_text, usage, response_meta = self._request(system_prompt, prompt, schema=PaperRecallProbeResponse)
        probe = _parse_model_json_response(
            raw_text=raw_text,
            stage="recognition probe",
            provider=self.provider,
            model_id=self.model_id,
            response_meta=response_meta,
            schema=PaperRecallProbeResponse,
        )
        return {
            "prompt_text": prompt,
            "raw_response": raw_text,
            "parsed_response": probe.model_dump(mode="json"),
            **usage,
        }


def _build_agents(config: PipelineConfig, baseline_only: bool) -> list[BaseAgent]:
    agents: list[BaseAgent] = [
        HeuristicAgent("always_A"),
        HeuristicAgent("always_B"),
        HeuristicAgent("uniform_random"),
    ]
    if baseline_only:
        return agents
    for slot_name, slot in config.resolved_model_slots().items():
        model_id = slot.resolved_model_id()
        if not model_id:
            continue
        agent = LLMAgent(
            slot_name,
            slot.provider,
            model_id,
            slot.max_output_tokens,
            slot,
            api_key_env=slot.resolved_api_key_env(),
            timeout_seconds=slot.timeout_seconds,
            max_retries=slot.max_retries,
            initial_backoff_seconds=slot.initial_backoff_seconds,
            max_backoff_seconds=slot.max_backoff_seconds,
        )
        if agent.is_available:
            agents.append(agent)
    return agents


def _sample_profiles(
    profiles: pd.DataFrame,
    rng: np.random.Generator,
    *,
    sample_size_per_treatment: int | None = None,
) -> pd.DataFrame:
    sampled: list[pd.DataFrame] = []
    for treatment, group in profiles.groupby("treatment"):
        target_size = len(group) if sample_size_per_treatment is None else min(sample_size_per_treatment, len(group))
        choice = rng.choice(group.index.to_numpy(), size=target_size, replace=True)
        picked = group.loc[choice].copy().reset_index(drop=True)
        picked["simulation_profile_index"] = range(len(picked))
        picked["treatment"] = treatment
        sampled.append(picked)
    return pd.concat(sampled, ignore_index=True)


def _select_audit_profiles(
    profiles: pd.DataFrame,
    rng: np.random.Generator,
    *,
    audit_sample_size: int,
) -> pd.DataFrame:
    if profiles.empty:
        return profiles.copy()

    grouped = {treatment: group.copy() for treatment, group in profiles.groupby("treatment", sort=True)}
    treatments = list(grouped)
    if not treatments:
        return profiles.iloc[0:0].copy()

    if audit_sample_size <= 0:
        target_total = len(treatments)
    else:
        target_total = min(audit_sample_size, sum(len(group) for group in grouped.values()))

    counts = {treatment: 0 for treatment in treatments}
    if audit_sample_size <= 0:
        for treatment in treatments:
            counts[treatment] = min(1, len(grouped[treatment]))
    else:
        base = target_total // len(treatments)
        remainder = target_total % len(treatments)
        for index, treatment in enumerate(treatments):
            counts[treatment] = min(len(grouped[treatment]), base + (1 if index < remainder else 0))
        allocated = sum(counts.values())
        while allocated < target_total:
            updated = False
            for treatment in treatments:
                if counts[treatment] < len(grouped[treatment]):
                    counts[treatment] += 1
                    allocated += 1
                    updated = True
                    if allocated == target_total:
                        break
            if not updated:
                break

    sampled: list[pd.DataFrame] = []
    for treatment in treatments:
        count = counts[treatment]
        if count <= 0:
            continue
        group = grouped[treatment]
        chosen_index = rng.choice(group.index.to_numpy(), size=count, replace=False)
        sampled.append(group.loc[chosen_index].copy())
    if not sampled:
        return profiles.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True).reset_index(drop=True)


def _sample_group_b_draws(worker_pool: pd.DataFrame, rounds: int, rng: np.random.Generator) -> list[float]:
    group_b = worker_pool.loc[worker_pool["canonical_group"] == "group_b", "productivity"].to_numpy()
    if len(group_b) < rounds:
        raise ValueError("Group B empirical pool is smaller than the requested rounds.")
    return rng.choice(group_b, size=rounds, replace=False).astype(float).tolist()


def _simulate_one_employer(
    *,
    spec: PaperSpec,
    profile: EmployerProfile,
    agent: BaseAgent,
    worker_pool: pd.DataFrame,
    replicate_id: int,
    simulation_employer_id: str,
    replicate_seed: int,
    label_variant: str,
    fixed_a_productivity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(stable_seed(replicate_seed, simulation_employer_id, label_variant, fixed_a_productivity))
    group_b_draws = _sample_group_b_draws(worker_pool, spec.rounds, rng)
    current_belief = float(profile.prior_b_prompt)
    b_index = 0
    round_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    history: list[dict[str, object]] = []

    for round_number in range(1, spec.rounds + 1):
        decision_response, decision_log = agent.decide(
            spec=spec,
            profile=profile,
            round_number=round_number,
            current_belief=current_belief,
            history=history,
            fixed_a_productivity=fixed_a_productivity,
            label_variant=label_variant,
            rng=rng,
        )
        action = decision_response.action
        forced = False
        if profile.treatment == Treatment.CONTROL:
            action = Action.HIRE_B
            forced = True
        belief_before = current_belief
        if action == Action.HIRE_A:
            observed_productivity = fixed_a_productivity
            belief_after = current_belief
        else:
            observed_productivity = float(group_b_draws[b_index])
            update_response, update_log = agent.update_belief(
                spec=spec,
                current_belief=current_belief,
                observed_productivity=observed_productivity,
                previous_b_hires=b_index,
                rng=rng,
            )
            belief_after = float(update_response.belief_b)
            b_index += 1
            interaction_rows.append(
                {
                    "paper_id": PAPER_ID,
                    "agent_name": agent.name,
                    "replicate_id": replicate_id,
                    "simulation_employer_id": simulation_employer_id,
                    "round": round_number,
                    "stage": "update",
                    "prompt_text": update_log["prompt_text"],
                    "raw_response": update_log["raw_response"],
                    "parsed_response": json.dumps(update_log["parsed_response"], sort_keys=True),
                    "input_tokens": update_log["input_tokens"],
                    "output_tokens": update_log["output_tokens"],
                    "total_tokens": update_log["total_tokens"],
                    "cost_usd_estimate": update_log["cost_usd_estimate"],
                    "replicate_seed": replicate_seed,
                    "label_variant": label_variant,
                    "fixed_a_productivity": fixed_a_productivity,
                }
            )
        current_belief = float(belief_after)
        experience_sign = ""
        if action == Action.HIRE_B:
            if observed_productivity > spec.positive_experience_cutoff:
                experience_sign = "positive"
            elif observed_productivity < spec.positive_experience_cutoff:
                experience_sign = "negative"
            else:
                experience_sign = "neutral"

        round_row = {
            "paper_id": PAPER_ID,
            "agent_name": agent.name,
            "provider": agent.provider,
            "replicate_id": replicate_id,
            "simulation_employer_id": simulation_employer_id,
            "source_employer_key": profile.employer_key,
            "treatment": profile.treatment.value,
            "round": round_number,
            "action": action.value,
            "forced_action": forced,
            "belief_before_round": round(float(belief_before), 6),
            "belief_after_round": round(float(belief_after), 6),
            "observed_productivity": round(float(observed_productivity), 6),
            "hired_b": int(action == Action.HIRE_B),
            "cumulative_b_hires": b_index,
            "experience_sign": experience_sign,
            "persona_age": profile.age_years,
            "persona_prior_b_raw": profile.prior_b_raw,
            "persona_prior_b_prompt": profile.prior_b_prompt,
            "persona_ambiguity_switch": profile.ambiguity_switch,
            "label_variant": label_variant,
            "fixed_a_productivity": fixed_a_productivity,
            "prompt_version": spec.version_pins["prompt"],
            "renderer_version": spec.version_pins["renderer"],
            "parser_version": spec.version_pins["parser"],
            "persona_version": spec.version_pins["persona"],
            "resampling_version": spec.version_pins["resampling"],
            "replicate_seed": replicate_seed,
        }
        round_rows.append(round_row)
        history.append(
            {
                "round": round_number,
                "action": action.value,
                "productivity": round(float(observed_productivity), 1),
                "belief_after": round(float(belief_after), 1),
            }
        )
        interaction_rows.append(
            {
                "paper_id": PAPER_ID,
                "agent_name": agent.name,
                "replicate_id": replicate_id,
                "simulation_employer_id": simulation_employer_id,
                "round": round_number,
                "stage": "decision",
                "prompt_text": decision_log["prompt_text"],
                "raw_response": decision_log["raw_response"],
                "parsed_response": json.dumps(decision_log["parsed_response"], sort_keys=True),
                "input_tokens": decision_log["input_tokens"],
                "output_tokens": decision_log["output_tokens"],
                "total_tokens": decision_log["total_tokens"],
                "cost_usd_estimate": decision_log["cost_usd_estimate"],
                "replicate_seed": replicate_seed,
                "label_variant": label_variant,
                "fixed_a_productivity": fixed_a_productivity,
            }
        )
    return round_rows, interaction_rows


def _finalize_simulation_records(records: pd.DataFrame, spec: PaperSpec) -> pd.DataFrame:
    frame = records.copy()
    frame["total_b_hires"] = frame.groupby(["agent_name", "replicate_id", "simulation_employer_id"])["hired_b"].transform("sum")
    frame["remaining_b_hires"] = frame["total_b_hires"] - frame["cumulative_b_hires"]
    first_mask = frame["hired_b"].eq(1) & frame["cumulative_b_hires"].eq(1)
    first_sign = frame.loc[first_mask, ["agent_name", "replicate_id", "simulation_employer_id", "experience_sign"]].rename(
        columns={"experience_sign": "first_experience_sign"}
    )
    frame = frame.merge(first_sign, on=["agent_name", "replicate_id", "simulation_employer_id"], how="left")
    frame["next_hired_b"] = frame.groupby(["agent_name", "replicate_id", "simulation_employer_id"])["hired_b"].shift(-1)
    frame["final_belief"] = frame.groupby(["agent_name", "replicate_id", "simulation_employer_id"])["belief_after_round"].transform("last")
    return frame


def _run_label_audit(
    *,
    spec: PaperSpec,
    agent: BaseAgent,
    profiles: pd.DataFrame,
    worker_pool: pd.DataFrame,
    audit_replicates: int,
    audit_sample_size: int,
    base_seed: int,
) -> dict[str, Any]:
    subset = _select_audit_profiles(
        profiles,
        np.random.default_rng(stable_seed(base_seed, "label-audit-subset")),
        audit_sample_size=audit_sample_size,
    )
    if subset.empty:
        return {"status": "skipped", "reason": "no_profiles"}
    deltas: list[float] = []
    for audit_id in range(audit_replicates):
        canonical_rates: list[float] = []
        neutral_rates: list[float] = []
        for label_variant in ("canonical", "neutral"):
            audit_rows: list[dict[str, Any]] = []
            for offset, row in subset.reset_index(drop=True).iterrows():
                profile = _profile_model(row)
                round_rows, _ = _simulate_one_employer(
                    spec=spec,
                    profile=profile,
                    agent=agent,
                    worker_pool=worker_pool,
                    replicate_id=audit_id,
                    simulation_employer_id=f"audit-{audit_id}-{offset}-{label_variant}",
                    replicate_seed=stable_seed(base_seed, "label-audit", agent.name, audit_id),
                    label_variant=label_variant,
                    fixed_a_productivity=spec.group_a_known_productivity,
                )
                audit_rows.extend(round_rows)
            rate = pd.DataFrame(audit_rows)["hired_b"].mean()
            if label_variant == "canonical":
                canonical_rates.append(float(rate))
            else:
                neutral_rates.append(float(rate))
        deltas.append(abs(neutral_rates[-1] - canonical_rates[-1]))
    return {
        "status": "completed",
        "mean_absolute_hire_b_rate_delta": round(float(np.mean(deltas)), 6),
        "replicates": audit_replicates,
        "sample_size_total": int(len(subset)),
        "sample_size_by_treatment": {
            str(key): int(value) for key, value in subset["treatment"].value_counts().sort_index().to_dict().items()
        },
    }


def _run_payoff_audit(
    *,
    spec: PaperSpec,
    agent: BaseAgent,
    profiles: pd.DataFrame,
    worker_pool: pd.DataFrame,
    audit_replicates: int,
    audit_sample_size: int,
    base_seed: int,
) -> dict[str, Any]:
    subset = _select_audit_profiles(
        profiles,
        np.random.default_rng(stable_seed(base_seed, "payoff-audit-subset")),
        audit_sample_size=audit_sample_size,
    )
    if subset.empty:
        return {"status": "skipped", "reason": "no_profiles"}
    rates: dict[str, list[float]] = {"baseline_payoff": [], "perturbed_payoff": []}
    for audit_id in range(audit_replicates):
        for scenario, fixed_a in {"baseline_payoff": spec.group_a_known_productivity, "perturbed_payoff": spec.group_a_known_productivity + 1}.items():
            rows: list[dict[str, Any]] = []
            for offset, row in subset.reset_index(drop=True).iterrows():
                profile = _profile_model(row)
                round_rows, _ = _simulate_one_employer(
                    spec=spec,
                    profile=profile,
                    agent=agent,
                    worker_pool=worker_pool,
                    replicate_id=audit_id,
                    simulation_employer_id=f"payoff-{audit_id}-{offset}-{scenario}",
                    replicate_seed=stable_seed(base_seed, "payoff-audit", agent.name, audit_id),
                    label_variant="canonical",
                    fixed_a_productivity=fixed_a,
                )
                rows.extend(round_rows)
            rates[scenario].append(float(pd.DataFrame(rows)["hired_b"].mean()))
    baseline_mean = float(np.mean(rates["baseline_payoff"]))
    perturbed_mean = float(np.mean(rates["perturbed_payoff"]))
    return {
        "status": "completed",
        "baseline_hire_b_rate": round(baseline_mean, 6),
        "perturbed_hire_b_rate": round(perturbed_mean, 6),
        "delta": round(perturbed_mean - baseline_mean, 6),
        "replicates": audit_replicates,
        "sample_size_total": int(len(subset)),
        "sample_size_by_treatment": {
            str(key): int(value) for key, value in subset["treatment"].value_counts().sort_index().to_dict().items()
        },
    }


def simulate_paper(
    config: PipelineConfig,
    project_root: Path,
    *,
    replicates: int | None = None,
    baseline_only: bool = False,
    smoke_test: bool = False,
) -> dict[str, Any]:
    paths = _paths(config, project_root)
    spec, profiles, worker_pool, _observed, _targets = _load_prepared(paths)
    run_replicates = 1 if smoke_test else (replicates or config.runtime.replicates)
    audit_replicates = 1 if smoke_test else config.runtime.audit_replicates
    audit_sample_size = 0 if smoke_test else config.runtime.audit_sample_size
    sample_size_per_treatment = 1 if smoke_test else None
    agents = _build_agents(config, baseline_only=baseline_only)
    if not agents:
        raise RuntimeError("No agents are available. Configure model env vars or run with --baseline-only.")

    simulation_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for agent in agents:
        recall_probe = agent.run_recall_probe()
        label_audit = _run_label_audit(
            spec=spec,
            agent=agent,
            profiles=profiles,
            worker_pool=worker_pool,
            audit_replicates=audit_replicates,
            audit_sample_size=audit_sample_size,
            base_seed=config.runtime.base_seed,
        )
        payoff_audit = _run_payoff_audit(
            spec=spec,
            agent=agent,
            profiles=profiles,
            worker_pool=worker_pool,
            audit_replicates=audit_replicates,
            audit_sample_size=audit_sample_size,
            base_seed=config.runtime.base_seed,
        )
        for replicate_id in range(run_replicates):
            replicate_seed = stable_seed(config.runtime.base_seed, agent.name, replicate_id)
            sampled_profiles = _sample_profiles(
                profiles,
                np.random.default_rng(replicate_seed),
                sample_size_per_treatment=sample_size_per_treatment,
            )
            for row_index, row in sampled_profiles.iterrows():
                profile = _profile_model(row)
                simulation_employer_id = f"{agent.name}:rep{replicate_id}:{row_index}"
                round_rows, logs = _simulate_one_employer(
                    spec=spec,
                    profile=profile,
                    agent=agent,
                    worker_pool=worker_pool,
                    replicate_id=replicate_id,
                    simulation_employer_id=simulation_employer_id,
                    replicate_seed=replicate_seed,
                    label_variant="canonical",
                    fixed_a_productivity=spec.group_a_known_productivity,
                )
                simulation_rows.extend(round_rows)
                interaction_rows.extend(logs)
        manifests.append(
            {
                "agent_name": agent.name,
                "provider": agent.provider,
                "recall_probe": recall_probe,
                "label_audit": label_audit,
                "payoff_audit": payoff_audit,
            }
        )

    simulation_frame = _finalize_simulation_records(pd.DataFrame(simulation_rows), spec)
    simulations_path = paths.output_dir / "simulation_records.csv"
    interactions_path = paths.output_dir / "interaction_logs.jsonl"
    manifest_path = paths.output_dir / "simulation_manifest.json"
    simulation_frame.to_csv(simulations_path, index=False)
    write_jsonl(interactions_path, interaction_rows)
    manifest = {
        "paper_id": PAPER_ID,
        "replicates": run_replicates,
        "smoke_test": smoke_test,
        "effective_runtime": {
            "replicates": run_replicates,
            "audit_replicates": audit_replicates,
            "audit_sample_size": audit_sample_size,
            "sample_size_per_treatment": sample_size_per_treatment,
        },
        "agents": manifests,
        "version_pins": config.versions.model_dump(),
        "simulation_records": str(simulations_path.relative_to(project_root)),
        "interaction_logs": str(interactions_path.relative_to(project_root)),
    }
    write_json(manifest_path, manifest)
    return manifest


def _distribution_metrics(observed_profiles: pd.DataFrame, observed_outcomes: pd.DataFrame, simulated: pd.DataFrame, agent_name: str) -> dict[str, Any]:
    agent_rows = simulated.loc[simulated["agent_name"] == agent_name].copy()
    round15_sim = agent_rows.loc[agent_rows["round"] == 15].copy()
    round15_obs = observed_outcomes.loc[observed_outcomes["round"] == 15].copy()
    metrics: dict[str, Any] = {}
    for treatment in [Treatment.BASELINE.value, Treatment.CONTROL.value]:
        observed_prior = observed_profiles.loc[observed_profiles["treatment"] == treatment, "prior_b_prompt"].to_numpy()
        simulated_prior = (
            agent_rows.sort_values(["replicate_id", "simulation_employer_id", "round"])
            .groupby(["replicate_id", "simulation_employer_id"], as_index=False)
            .first()
            .loc[lambda frame: frame["treatment"] == treatment, "persona_prior_b_prompt"]
            .to_numpy()
        )
        observed_final = round15_obs.loc[round15_obs["treatment"] == treatment, "belief_after_round"].to_numpy()
        simulated_final = round15_sim.loc[round15_sim["treatment"] == treatment, "belief_after_round"].to_numpy()
        metrics[f"{treatment}_prior_belief_ks"] = round(float(stats.ks_2samp(observed_prior, simulated_prior).statistic), 6)
        metrics[f"{treatment}_prior_belief_wasserstein"] = round(float(stats.wasserstein_distance(observed_prior, simulated_prior)), 6)
        metrics[f"{treatment}_final_belief_ks"] = round(float(stats.ks_2samp(observed_final, simulated_final).statistic), 6)
        metrics[f"{treatment}_final_belief_wasserstein"] = round(float(stats.wasserstein_distance(observed_final, simulated_final)), 6)

        obs_round_rate = observed_outcomes.loc[observed_outcomes["treatment"] == treatment].groupby("round")["hired_b"].mean().sort_index()
        sim_round_rate = (
            agent_rows.loc[agent_rows["treatment"] == treatment]
            .groupby(["replicate_id", "round"])["hired_b"]
            .mean()
            .groupby("round")
            .mean()
            .sort_index()
        )
        metrics[f"{treatment}_hire_b_rate_rmse"] = round(float(np.sqrt(np.mean((obs_round_rate.to_numpy() - sim_round_rate.to_numpy()) ** 2))), 6)
    return metrics


def _replicate_effects(frame: pd.DataFrame) -> dict[str, float]:
    round15 = frame.loc[frame["round"] == 15].copy()
    nonzero = round15.loc[round15["total_b_hires"] > 0].copy()
    first_b = frame.loc[(frame["treatment"] == Treatment.BASELINE.value) & (frame["cumulative_b_hires"] == 1)].copy()
    follow = frame.loc[(frame["treatment"] == Treatment.BASELINE.value) & (frame["hired_b"] == 1) & (frame["round"] < 15)].copy()
    remaining_column = "remaining_b_hires" if "remaining_b_hires" in frame.columns else "remh"
    final_belief_column = "final_belief" if "final_belief" in frame.columns else "belief_after_round"

    def _safe_mean(series: pd.Series) -> float:
        return float(series.mean()) if not series.empty else float("nan")

    return {
        "control_minus_baseline_final_belief_nonzero_b_hires": _safe_mean(
            nonzero.loc[nonzero["treatment"] == Treatment.CONTROL.value, "belief_after_round"]
        ) - _safe_mean(nonzero.loc[nonzero["treatment"] == Treatment.BASELINE.value, "belief_after_round"]),
        "positive_minus_negative_future_b_hires": _safe_mean(first_b.loc[first_b["experience_sign"] == "positive", remaining_column])
        - _safe_mean(first_b.loc[first_b["experience_sign"] == "negative", remaining_column]),
        "positive_minus_negative_final_belief": _safe_mean(first_b.loc[first_b["experience_sign"] == "positive", final_belief_column])
        - _safe_mean(first_b.loc[first_b["experience_sign"] == "negative", final_belief_column]),
        "positive_minus_negative_next_round_hire_rate": _safe_mean(follow.loc[follow["experience_sign"] == "positive", "next_hired_b"])
        - _safe_mean(follow.loc[follow["experience_sign"] == "negative", "next_hired_b"]),
    }


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _observed_effect_tables(observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    remaining_column = "remaining_b_hires" if "remaining_b_hires" in observed.columns else "remh"
    final_belief_column = "final_belief" if "final_belief" in observed.columns else "belief_after_round"
    round15 = observed.loc[observed["round"] == 15, ["employer_key", "treatment", "belief_after_round", "total_b_hires"]].drop_duplicates()
    first_b = observed.loc[
        (observed["treatment"] == Treatment.BASELINE.value) & (observed["cumulative_b_hires"] == 1),
        ["employer_key", "experience_sign", remaining_column, final_belief_column],
    ].drop_duplicates()
    first_b = first_b.rename(columns={remaining_column: "remaining_b_hires", final_belief_column: "final_belief"})
    follow = observed.loc[
        (observed["treatment"] == Treatment.BASELINE.value) & (observed["hired_b"] == 1) & (observed["round"] < 15),
        ["employer_key", "experience_sign", "next_hired_b"],
    ].copy()
    return round15, first_b, follow


def _bootstrap_effect_from_tables(
    round15: pd.DataFrame,
    first_b: pd.DataFrame,
    follow: pd.DataFrame,
    sampled_keys: pd.Series,
) -> dict[str, float]:
    weights = sampled_keys.value_counts().rename_axis("employer_key").reset_index(name="weight")
    weighted_round15 = round15.merge(weights, on="employer_key", how="inner")
    weighted_first = first_b.merge(weights, on="employer_key", how="inner")
    weighted_follow = follow.merge(weights, on="employer_key", how="inner")

    control_nonzero = weighted_round15.loc[
        (weighted_round15["treatment"] == Treatment.CONTROL.value) & (weighted_round15["total_b_hires"] > 0)
    ]
    baseline_nonzero = weighted_round15.loc[
        (weighted_round15["treatment"] == Treatment.BASELINE.value) & (weighted_round15["total_b_hires"] > 0)
    ]
    positive_first = weighted_first.loc[weighted_first["experience_sign"] == "positive"]
    negative_first = weighted_first.loc[weighted_first["experience_sign"] == "negative"]
    positive_follow = weighted_follow.loc[weighted_follow["experience_sign"] == "positive"]
    negative_follow = weighted_follow.loc[weighted_follow["experience_sign"] == "negative"]

    return {
        "control_minus_baseline_final_belief_nonzero_b_hires": _weighted_mean(
            control_nonzero["belief_after_round"], control_nonzero["weight"]
        )
        - _weighted_mean(baseline_nonzero["belief_after_round"], baseline_nonzero["weight"]),
        "positive_minus_negative_future_b_hires": _weighted_mean(
            positive_first["remaining_b_hires"], positive_first["weight"]
        )
        - _weighted_mean(negative_first["remaining_b_hires"], negative_first["weight"]),
        "positive_minus_negative_final_belief": _weighted_mean(
            positive_first["final_belief"], positive_first["weight"]
        )
        - _weighted_mean(negative_first["final_belief"], negative_first["weight"]),
        "positive_minus_negative_next_round_hire_rate": _weighted_mean(
            positive_follow["next_hired_b"], positive_follow["weight"]
        )
        - _weighted_mean(negative_follow["next_hired_b"], negative_follow["weight"]),
    }


def _bootstrap_observed_effects(observed: pd.DataFrame, iterations: int, seed: int) -> dict[str, dict[str, float]]:
    employer_keys = pd.Series(observed["employer_key"].drop_duplicates().to_numpy())
    round15, first_b, follow = _observed_effect_tables(observed)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    observed_effects = _bootstrap_effect_from_tables(round15, first_b, follow, employer_keys)
    for _ in range(iterations):
        sampled = pd.Series(rng.choice(employer_keys.to_numpy(), size=len(employer_keys), replace=True))
        for metric_name, value in _bootstrap_effect_from_tables(round15, first_b, follow, sampled).items():
            draws.setdefault(metric_name, []).append(value)
    summary: dict[str, dict[str, float]] = {}
    for metric_name, values in draws.items():
        summary[metric_name] = {
            "estimate": round(float(observed_effects[metric_name]), 6),
            "ci_low": round(float(np.nanquantile(values, 0.025)), 6),
            "ci_high": round(float(np.nanquantile(values, 0.975)), 6),
        }
    return summary


def _intervals_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return max(a_low, b_low) <= min(a_high, b_high)


def _safe_nanmean(values: list[float]) -> float:
    filtered = [value for value in values if pd.notna(value)]
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def _inferential_metrics(
    observed_summary: dict[str, dict[str, float]],
    simulated: pd.DataFrame,
    agent_name: str,
) -> dict[str, Any]:
    agent_rows = simulated.loc[simulated["agent_name"] == agent_name].copy()
    replicate_values = {
        replicate_id: _replicate_effects(frame)
        for replicate_id, frame in agent_rows.groupby("replicate_id")
    }
    metrics: dict[str, Any] = {}
    for name in next(iter(replicate_values.values())).keys():
        values = np.array([replicate_values[rep_id][name] for rep_id in sorted(replicate_values)], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            sim_estimate = float("nan")
            sim_low = float("nan")
            sim_high = float("nan")
        else:
            sim_estimate = float(np.mean(finite))
            sim_low = float(np.quantile(finite, 0.025))
            sim_high = float(np.quantile(finite, 0.975))
        obs = observed_summary[name]
        metrics[name] = {
            "observed_estimate": obs["estimate"],
            "observed_ci_low": obs["ci_low"],
            "observed_ci_high": obs["ci_high"],
            "simulated_estimate": round(sim_estimate, 6),
            "simulated_ci_low": round(sim_low, 6),
            "simulated_ci_high": round(sim_high, 6),
            "sign_match": bool(np.sign(obs["estimate"]) == np.sign(sim_estimate) or (obs["estimate"] == 0 and sim_estimate == 0)),
            "effect_size_error": round(abs(sim_estimate - obs["estimate"]), 6),
            "ci_overlap": bool(_intervals_overlap(obs["ci_low"], obs["ci_high"], sim_low, sim_high)),
        }
    return metrics


def _write_agent_report(
    *,
    agent_name: str,
    provider: str,
    distribution_metrics: dict[str, Any],
    inferential_metrics: dict[str, Any],
    audits: dict[str, Any],
    report_path: Path,
) -> None:
    distribution_rows = [{"metric": key, "value": value} for key, value in distribution_metrics.items()]
    inferential_rows = [
        {
            "metric": key,
            "obs": value["observed_estimate"],
            "sim": value["simulated_estimate"],
            "error": value["effect_size_error"],
            "sign_match": value["sign_match"],
            "ci_overlap": value["ci_overlap"],
        }
        for key, value in inferential_metrics.items()
    ]
    report = "\n".join(
        [
            f"# Evaluation Report: {agent_name}",
            "",
            f"- Provider: {provider}",
            "",
            "## Distributional Fit",
            format_markdown_table(distribution_rows, ["metric", "value"]),
            "",
            "## Inferential Fit",
            format_markdown_table(inferential_rows, ["metric", "obs", "sim", "error", "sign_match", "ci_overlap"]),
            "",
            "## Contamination Safeguards",
            "```json",
            json.dumps(audits, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")


def evaluate_simulations(config: PipelineConfig, project_root: Path) -> dict[str, Any]:
    paths = _paths(config, project_root)
    spec, observed_profiles, _worker_pool, observed_outcomes, _targets = _load_prepared(paths)
    simulations_path = paths.output_dir / "simulation_records.csv"
    manifest_path = paths.output_dir / "simulation_manifest.json"
    if not simulations_path.exists():
        raise FileNotFoundError(f"Missing simulation records: {simulations_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing simulation manifest: {manifest_path}")
    simulated = pd.read_csv(simulations_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_summary = _bootstrap_observed_effects(
        observed_outcomes,
        config.runtime.bootstrap_iterations,
        stable_seed(config.runtime.base_seed, "obs-bootstrap"),
    )

    evaluation_dir = ensure_directory(paths.output_dir / "evaluation")
    summary_rows: list[dict[str, Any]] = []
    full_summary: dict[str, Any] = {"paper_id": PAPER_ID, "agents": {}}
    for agent in manifest["agents"]:
        agent_name = agent["agent_name"]
        distribution_metrics = _distribution_metrics(observed_profiles, observed_outcomes, simulated, agent_name)
        inferential_metrics = _inferential_metrics(
            observed_summary,
            simulated,
            agent_name,
        )
        agent_report_path = evaluation_dir / f"{agent_name}.md"
        _write_agent_report(
            agent_name=agent_name,
            provider=agent["provider"],
            distribution_metrics=distribution_metrics,
            inferential_metrics=inferential_metrics,
            audits={
                "recall_probe": agent["recall_probe"],
                "label_audit": agent["label_audit"],
                "payoff_audit": agent["payoff_audit"],
            },
            report_path=agent_report_path,
        )
        mean_distribution_error = round(_safe_nanmean(list(distribution_metrics.values())), 6)
        mean_effect_error = round(_safe_nanmean([metric["effect_size_error"] for metric in inferential_metrics.values()]), 6)
        summary_rows.append(
            {
                "agent": agent_name,
                "provider": agent["provider"],
                "mean_distribution_error": mean_distribution_error,
                "mean_effect_error": mean_effect_error,
            }
        )
        full_summary["agents"][agent_name] = {
            "provider": agent["provider"],
            "distribution_metrics": distribution_metrics,
            "inferential_metrics": inferential_metrics,
            "audits": {
                "recall_probe": agent["recall_probe"],
                "label_audit": agent["label_audit"],
                "payoff_audit": agent["payoff_audit"],
            },
            "report_path": str(agent_report_path.relative_to(project_root)),
        }

    comparison_md = "\n".join(
        [
            "# Evaluation Comparison",
            "",
            format_markdown_table(summary_rows, ["agent", "provider", "mean_distribution_error", "mean_effect_error"]),
            "",
            "Composite scores are intentionally omitted; the per-agent reports retain the raw metric tables.",
            "",
        ]
    )
    comparison_path = evaluation_dir / "comparison.md"
    comparison_path.write_text(comparison_md, encoding="utf-8")
    write_json(evaluation_dir / "summary.json", full_summary)
    return {
        "paper_id": PAPER_ID,
        "comparison_report": str(comparison_path.relative_to(project_root)),
        "agent_reports": {
            agent_name: info["report_path"]
            for agent_name, info in full_summary["agents"].items()
        },
    }
