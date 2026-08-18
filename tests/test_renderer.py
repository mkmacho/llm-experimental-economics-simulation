from __future__ import annotations

from synthetic_replication.config import load_config
from synthetic_replication.papers import build_paper_spec
from synthetic_replication.render import render_decision_prompt
from synthetic_replication.types import EmployerProfile, Treatment


def test_decision_prompt_snapshot() -> None:
    config = load_config()
    spec = build_paper_spec(config)
    profile = EmployerProfile(
        employer_key="A:1",
        source_numeric_id=1,
        source_group_code="A",
        treatment=Treatment.BASELINE,
        age_years=25,
        prior_b_raw=6.0,
        prior_b_prompt=6.0,
        prior_b_was_clipped=False,
        ambiguity_switch=12.0,
        observed_total_b_hires=6,
        observed_final_belief=5.5,
    )
    prompt = render_decision_prompt(
        spec=spec,
        profile=profile,
        treatment=Treatment.BASELINE,
        round_number=1,
        current_belief=6.0,
        history=[],
        fixed_a_productivity=9.0,
    )
    expected = """Task Card

Role
You are an employer in a repeated hiring task.

Profile
- Age: 25
- Initial estimate of Group B mean productivity: 6.0
- Ambiguity-switch score from a separate task: 12.0

Round: 1 of 15
Treatment: baseline

Known facts
- Group A has 75% of workers.
- Hiring Group A yields exactly 9.0 solved puzzles.
- Group B has 25% of workers.
- Hiring Group B reveals one worker drawn without replacement from Group B.
- Valid productivity values run from 1 to 18.
- Your current carried-over estimate of Group B's mean productivity is 6.0.

History
Round | Choice | Observed productivity | Belief after round
No completed rounds yet.

Action requirement
- Allowed actions this round: `hire_A` or `hire_B`.

Response format
Return JSON only with keys "action" and "belief_b".
- "belief_b" should be your current estimate of Group B mean productivity before this round is resolved."""
    assert prompt == expected
