from __future__ import annotations

from synthetic_replication.config import load_config
import pytest

from synthetic_replication.papers.experience_based_discrimination import (
    _anthropic_prefill_for_schema,
    _build_agents,
    _normalize_json_response,
    _parse_model_json_response,
)
from synthetic_replication.types import Action, BeliefUpdateResponse, DecisionResponse, PaperRecallProbeResponse


def test_single_model_config_supports_anthropic(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        """
default_paper = "experience-based-discrimination"

[versions]
pipeline = "v1"
prompt = "v1"
renderer = "v1"
parser = "v1"
persona = "v1"
resampling = "v1"

[model]
provider = "anthropic"
model_env = "ANTHROPIC_MODEL"
max_output_tokens = 180
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")

    config = load_config(config_path)
    slots = config.resolved_model_slots()

    assert list(slots) == ["default"]
    assert slots["default"].provider == "anthropic"
    assert slots["default"].resolved_model_id() == "claude-test"
    assert slots["default"].resolved_api_key_env() == "ANTHROPIC_API_KEY"


def test_multi_model_config_supports_named_slots(tmp_path) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        """
default_paper = "experience-based-discrimination"

[versions]
pipeline = "v1"
prompt = "v1"
renderer = "v1"
parser = "v1"
persona = "v1"
resampling = "v1"

[models.small]
provider = "openai"
model_id = "gpt-test"
max_output_tokens = 120

[models.frontier]
provider = "anthropic"
model_id = "claude-test"
max_output_tokens = 240
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    slots = config.resolved_model_slots()

    assert list(slots) == ["small", "frontier"]
    assert slots["small"].resolved_model_id() == "gpt-test"
    assert slots["frontier"].provider == "anthropic"
    assert slots["frontier"].resolved_model_id() == "claude-test"


def test_build_agents_includes_anthropic_agent_when_configured(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "pipeline.toml"
    config_path.write_text(
        """
default_paper = "experience-based-discrimination"

[versions]
pipeline = "v1"
prompt = "v1"
renderer = "v1"
parser = "v1"
persona = "v1"
resampling = "v1"

[model]
slot_name = "primary"
provider = "anthropic"
model_id = "claude-test"
max_output_tokens = 180
timeout_seconds = 120
max_retries = 5
initial_backoff_seconds = 2.0
max_backoff_seconds = 20.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    config = load_config(config_path)
    agents = _build_agents(config, baseline_only=False)

    assert [agent.name for agent in agents] == ["always_A", "always_B", "uniform_random", "primary"]
    assert [agent.provider for agent in agents][-1] == "anthropic"
    assert agents[-1].client.timeout_seconds == 120
    assert agents[-1].client.max_retries == 5


def test_normalize_json_response_handles_code_fences() -> None:
    raw_text = """```json
{"action":"hire_B","belief_b":7.5}
```"""

    assert _normalize_json_response(raw_text) == '{"action":"hire_B","belief_b":7.5}'


def test_normalize_json_response_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Empty model response"):
        _normalize_json_response("   \n\t ")


def test_parse_model_json_response_surfaces_provider_metadata() -> None:
    with pytest.raises(RuntimeError, match="anthropic model 'claude-test' returned invalid JSON during decision"):
        _parse_model_json_response(
            raw_text="",
            stage="decision",
            provider="anthropic",
            model_id="claude-test",
            response_meta={"stop_reason": "end_turn", "content_block_types": []},
            schema=DecisionResponse,
        )


def test_anthropic_prefill_matches_schema() -> None:
    assert _anthropic_prefill_for_schema(DecisionResponse) == '{"action":"'
    assert _anthropic_prefill_for_schema(BeliefUpdateResponse) == '{"belief_b":'
    assert _anthropic_prefill_for_schema(PaperRecallProbeResponse) == '{"recognized":'


def test_parse_model_json_response_coerces_object_action_shape() -> None:
    parsed = _parse_model_json_response(
        raw_text='{"action":{"hire_B":1},"belief_b":2.5}',
        stage="decision",
        provider="anthropic",
        model_id="claude-test",
        response_meta={"stop_reason": "end_turn", "content_block_types": ["text"]},
        schema=DecisionResponse,
    )

    assert parsed.action == Action.HIRE_B
    assert parsed.belief_b == 2.5
