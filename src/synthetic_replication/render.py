from __future__ import annotations

from collections.abc import Iterable

from synthetic_replication.types import Action, EmployerProfile, PaperSpec, Treatment


def _persona_lines(profile: EmployerProfile) -> list[str]:
    lines = [
        f"- Age: {int(profile.age_years)}" if profile.age_years is not None else "- Age: unavailable",
        f"- Initial estimate of Group B mean productivity: {profile.prior_b_prompt:.1f}",
    ]
    if profile.ambiguity_switch is not None:
        lines.append(f"- Ambiguity-switch score from a separate task: {profile.ambiguity_switch:.1f}")
    else:
        lines.append("- Ambiguity-switch score from a separate task: unavailable")
    return lines


def _history_lines(history: Iterable[dict[str, object]]) -> list[str]:
    lines = ["Round | Choice | Observed productivity | Belief after round"]
    for row in history:
        lines.append(
            f"{row['round']:>5} | {row['action']:>6} | {row['productivity']:>19} | {row['belief_after']:>18}"
        )
    if len(lines) == 1:
        lines.append("No completed rounds yet.")
    return lines


def render_decision_prompt(
    spec: PaperSpec,
    profile: EmployerProfile,
    treatment: Treatment,
    round_number: int,
    current_belief: float,
    history: list[dict[str, object]],
    fixed_a_productivity: float,
    label_variant: str = "canonical",
) -> str:
    if label_variant == "neutral":
        group_a_name = "Group X"
        group_b_name = "Group Y"
    else:
        group_a_name = "Group A"
        group_b_name = "Group B"
    allowed_actions = (
        "`hire_A` or `hire_B`"
        if treatment == Treatment.BASELINE
        else "`hire_B` only"
    )
    return "\n".join(
        [
            "Task Card",
            "",
            "Role",
            "You are an employer in a repeated hiring task.",
            "",
            "Profile",
            *_persona_lines(profile),
            "",
            f"Round: {round_number} of {spec.rounds}",
            f"Treatment: {treatment.value}",
            "",
            "Known facts",
            f"- {group_a_name} has {int(spec.group_a_majority_share * 100)}% of workers.",
            f"- Hiring {group_a_name} yields exactly {fixed_a_productivity:.1f} solved puzzles.",
            f"- {group_b_name} has {int(spec.group_b_minority_share * 100)}% of workers.",
            f"- Hiring {group_b_name} reveals one worker drawn without replacement from {group_b_name}.",
            f"- Valid productivity values run from {spec.belief_min:.0f} to {spec.belief_max:.0f}.",
            f"- Your current carried-over estimate of {group_b_name}'s mean productivity is {current_belief:.1f}.",
            "",
            "History",
            *_history_lines(history),
            "",
            "Action requirement",
            f"- Allowed actions this round: {allowed_actions}.",
            "",
            "Response format",
            'Return JSON only with keys "action" and "belief_b".',
            '- "belief_b" should be your current estimate of Group B mean productivity before this round is resolved.',
        ]
    )


def render_update_prompt(
    spec: PaperSpec,
    observed_productivity: float,
    current_belief: float,
) -> str:
    return "\n".join(
        [
            "Outcome Card",
            "",
            "You hired Group B this round.",
            f"- Observed productivity: {observed_productivity:.1f}",
            f"- Your estimate just before seeing this outcome was {current_belief:.1f}",
            "",
            "Response format",
            'Return JSON only with key "belief_b".',
            f'- The updated belief must stay within [{spec.belief_min:.0f}, {spec.belief_max:.0f}].',
        ]
    )


def render_recall_probe() -> str:
    return "\n".join(
        [
            "Recognition probe",
            "",
            "A participant repeatedly hires from a known group with fixed productivity 9 or an uncertain minority group whose workers are drawn without replacement.",
            "The participant updates a belief about the uncertain group's mean productivity after hiring from that group.",
            "",
            'Return JSON only with keys "recognized", "confidence", and "notes".',
            '- "recognized" must be true or false.',
            '- "confidence" must be between 0 and 1.',
        ]
    )
