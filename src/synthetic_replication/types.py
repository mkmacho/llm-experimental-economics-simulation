from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    HIRE_A = "hire_A"
    HIRE_B = "hire_B"


class Treatment(str, Enum):
    BASELINE = "baseline"
    CONTROL = "control"


class TreatmentSpec(BaseModel):
    name: Treatment
    description: str
    allowed_actions: list[Action]
    forced_action: Action | None = None


class PaperSpec(BaseModel):
    paper_id: str
    paper_title: str
    rounds: int
    group_a_known_productivity: float
    group_b_minority_share: float
    group_a_majority_share: float
    belief_min: float = 1.0
    belief_max: float = 18.0
    positive_experience_cutoff: float = 9.0
    group_b_without_replacement: bool = True
    baseline_treatment: TreatmentSpec
    control_treatment: TreatmentSpec
    version_pins: dict[str, str]
    notes: list[str] = Field(default_factory=list)


class EmployerProfile(BaseModel):
    employer_key: str
    source_numeric_id: int
    source_group_code: str
    treatment: Treatment
    age_years: float | None = None
    prior_b_raw: float
    prior_b_prompt: float
    prior_b_was_clipped: bool
    ambiguity_switch: float | None = None
    observed_total_b_hires: int
    observed_final_belief: float


class DecisionResponse(BaseModel):
    action: Action
    belief_b: float

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, value: object) -> object:
        if isinstance(value, dict):
            truthy_actions = [
                key
                for key, marker in value.items()
                if key in {action.value for action in Action} and bool(marker)
            ]
            if len(truthy_actions) == 1:
                return truthy_actions[0]
        return value

    @field_validator("belief_b")
    @classmethod
    def validate_belief(cls, value: float) -> float:
        if value < 1 or value > 18:
            raise ValueError("belief_b must lie within [1, 18]")
        return float(value)


class BeliefUpdateResponse(BaseModel):
    belief_b: float

    @field_validator("belief_b")
    @classmethod
    def validate_belief(cls, value: float) -> float:
        if value < 1 or value > 18:
            raise ValueError("belief_b must lie within [1, 18]")
        return float(value)


class PaperRecallProbeResponse(BaseModel):
    recognized: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""
