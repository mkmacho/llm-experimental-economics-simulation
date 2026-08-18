from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


class VersionPins(BaseModel):
    pipeline: str
    prompt: str
    renderer: str
    parser: str
    persona: str
    resampling: str


class RuntimeConfig(BaseModel):
    replicates: int = 20
    base_seed: int = 20260317
    bootstrap_iterations: int = 300
    audit_replicates: int = 2
    audit_sample_size: int = 24


class PathsConfig(BaseModel):
    raw_root: str = "data/raw"
    normalized_root: str = "data/normalized"
    output_root: str = "outputs"


class ModelSlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_name: str | None = None
    provider: str = "openai"
    model_env: str | None = None
    model_id: str | None = None
    api_key_env: str | None = None
    max_output_tokens: int = 180
    timeout_seconds: int = 90
    max_retries: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 12.0
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    @field_validator("slot_name", "model_env", "model_id", "api_key_env", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> str:
        normalized = str(value or "openai").strip().lower()
        if normalized not in {"openai", "anthropic"}:
            raise ValueError("provider must be 'openai' or 'anthropic'")
        return normalized

    @model_validator(mode="after")
    def validate_model_source(self) -> ModelSlotConfig:
        if not self.model_env and not self.model_id:
            raise ValueError("Either model_env or model_id must be configured for each model slot")
        return self

    def resolved_model_id(self) -> str | None:
        env_value = _blank_to_none(os.getenv(self.model_env)) if self.model_env else None
        return env_value or self.model_id

    def resolved_slot_name(self, fallback: str = "default") -> str:
        return self.slot_name or fallback

    def resolved_api_key_env(self) -> str:
        if self.api_key_env:
            return self.api_key_env
        if self.provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        return "OPENAI_API_KEY"

    def cost_estimate(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return round(
            (input_tokens / 1_000_000.0) * self.input_cost_per_million
            + (output_tokens / 1_000_000.0) * self.output_cost_per_million,
            8,
        )


class PipelineConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    default_paper: str = "experience-based-discrimination"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    versions: VersionPins
    paths: PathsConfig = Field(default_factory=PathsConfig)
    single_model: ModelSlotConfig | None = Field(default=None, alias="model")
    models: dict[str, ModelSlotConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model_names(self) -> PipelineConfig:
        if self.single_model:
            single_name = self.single_model.resolved_slot_name()
            if single_name in self.models:
                raise ValueError(f"Duplicate model slot name configured via [model] and [models.{single_name}]")
        return self

    def resolved_model_slots(self) -> dict[str, ModelSlotConfig]:
        slots: dict[str, ModelSlotConfig] = {}
        if self.single_model:
            slots[self.single_model.resolved_slot_name()] = self.single_model
        slots.update(self.models)
        return slots


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    path = Path(config_path) if config_path else project_root() / "config" / "pipeline.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return PipelineConfig.model_validate(data)
