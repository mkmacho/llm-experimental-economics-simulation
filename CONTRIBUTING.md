# Contributing

Keep the public repository code-only: do not commit participant-level data,
archival experiment materials, generated interaction logs, API keys, or local
outputs. Changes to the simulation design should explain the experimental
mapping, the prompt information set, random seeds, expected model-call budget,
and the validation or contamination check that supports the change.

Before opening a pull request, run `uv run pytest -q` from the repository root.
Live-provider checks are opt-in and should never run in CI.
