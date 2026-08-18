# LLM Experimental Economics Simulation

This repository reconstructs the core experiment of [Experience-based Discrimination by Louis-Pierre Lepage](https://www.aeaweb.org/articles?id=10.1257/app.20220466) from public materials, simulates employer decisions with heuristic or LLM-based agents, and evaluates whether the simulated behavior matches both the observed micro-data and the paper's headline findings.

> **Public-release note:** this is a code-only project. It does not include
> subject-level data or archived experiment materials. Read
> [data/README.md](data/README.md) and [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)
> before acquiring or sharing external inputs.

It is a transparent, code-first framework for testing whether heuristic and LLM
agents can reproduce economically meaningful behavior without redistributing
the underlying participant-level data.

## What This Repository Does

The project has four concrete goals:

1. Prepare a reproducible local dataset from public archive files.
2. Simulate the experiment end to end with transparent prompts and logs.
3. Evaluate the simulation on both behavioral fit and paper-level conclusions.
4. Preserve a detailed record of decisions, caveats, and open questions.

The project reconstructs one experimental design from a public source archive.

Current paper scope:

* Paper: `Experience-based Discrimination`
* Treatments: `Baseline` and `Control`
* Horizon: `15` employer rounds
* Non-LLM baselines: `always_A`, `always_B`, `uniform_random`
* Optional LLM providers: OpenAI and Anthropic

The original repo guidelines are preserved in [GUIDELINES](GUIDELINES.md). The running decision record lives in [PLANNING](PLANNING.md) and [planning/decision_log.jsonl](planning/decision_log.jsonl).

## Current Status

Implemented:

* Paper selection with a primary target and backup
* Reproducible preparation of the `Experience-based Discrimination` core sample
* Simulation pipeline with heuristic agents and configurable LLM backends
* Evaluation reports for distributional fit, paper-level effects, and contamination audits

Normalized core sample:

* `297` Baseline employers
* `135` Control employers
* `15` rounds per employer

## Pipeline Overview

### Prepare

`prepare` reads locally acquired raw files from `data/raw/experience-based-discrimination/` and writes normalized artifacts:

* `employer_profiles.csv`
* `worker_pool_empirical.csv`
* `observed_core_outcomes.csv`
* `paper_spec.json`
* `observed_targets.json`
* `prepare_manifest.json`

The executable `prepare` stage reads `data.dta` and `dataworker.dta`. See
[data/README.md](data/README.md) for the official source and local file layout.

Key choices:

* employer identity is reconstructed with `empgroup:id`
* only complete `15`-round Baseline and Control traces are retained
* the worker side is treated as exogenous and resampled empirically during simulation

### Simulate

`simulate` samples employer personas with replacement within treatment, samples Group-B productivity without replacement within each simulated employer, renders a screen-like task card, and runs a per-round decision loop.

For contamination audits, `audit_sample_size` is interpreted as the total sampled employer count across treatments and is drawn without replacement within treatment.

The model sees only task-relevant information:

* role and instructions
* treatment and round
* fixed and uncertain payoff structure
* payoff-relevant history
* current belief and allowed actions

Simulation outputs:

* `outputs/experience-based-discrimination/simulation_records.csv`
* `outputs/experience-based-discrimination/interaction_logs.jsonl`
* `outputs/experience-based-discrimination/simulation_manifest.json`

These files are generated locally and intentionally git-ignored.

### Expected API Calls

`prepare` and `evaluate` make `0` model API calls. All live model traffic happens in `simulate`, and `run` makes the same model calls as `simulate` because its additional stages are local.

Per configured LLM model, the current default call budget is:

* recall probe: `1`
* label audit: `2160` to `2880`
* payoff audit: `2160` to `2880`
* main simulation: `8505` to `12960` per replicate

The ranges come from one decision call every round plus one extra belief-update call whenever the final action is `hire_B`.

With the current paper sample and default audit settings, a one-model run is approximately:

* `simulate --replicates 1`: minimum `12,826`, maximum `18,721`, observed-rate estimate about `15,985`
* each additional replicate adds `8,505` to `12,960` more calls, with observed-rate estimate about `10,894`
* `run --replicates R`: the same model-call count as `simulate --replicates R`

If you run `--baseline-only`, the agents are heuristic and external API calls drop to `0`.

### Smoke Test

A true low-cost smoke test is available for live LLM checks:

```bash
uv run llm-experimental-economics-simulation simulate --smoke-test
uv run llm-experimental-economics-simulation run --smoke-test
```

`--smoke-test` forces:

* `1` replicate
* `1` sampled employer per treatment in the main simulation
* `1` audit replicate
* the smallest non-zero audit sample

That reduces a one-model live run from roughly `16k` calls to roughly `226` to `301` calls. Smoke-test outputs are only for connectivity and pipeline sanity checks, not for empirical evaluation.

### Evaluate

`evaluate` measures both micro-level fit and paper-level conclusions.

Behavioral targets include:

* prior belief distributions
* final belief distributions
* round-level hire-B rates
* reactions to positive versus negative Group-B experience

Paper-level targets include:

* `Control - Baseline` final-belief gap among employers with at least one B hire
* downstream effects of positive versus negative first Group-B experience

Evaluation outputs are written to `outputs/experience-based-discrimination/evaluation/`.

### Contamination Safeguards

The repository includes explicit checks against paper-recognition artifacts:

* blind prompts in the main loop
* a separate paper-recall probe
* relabeled-group audits
* payoff-perturbation audits

## Deliberate Abstractions

Two pipeline abstractions are intentional and central to the current design:

* the renderer shows a structured text-card prompt instead of replaying the original oTree HTML screens verbatim
* the main prompts use canonical or neutral group labels rather than the archive's original color labels

Those choices are deliberate because they preserve the economically relevant state the employer sees, including treatment, round, allowed actions, payoff-relevant history, and current belief, while reducing prompt brittleness and supporting leakage-control audits. The detailed audit note is in [docs/archive-reconstruction-audit.md](docs/archive-reconstruction-audit.md).

## Installation

### Requirements

* Python `3.11+`
* `uv` recommended, but not required

### Install Dependencies

With `uv`:

```bash
uv sync --extra dev
```

With `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Runtime settings live in [config/pipeline.toml](config/pipeline.toml).

The model layer supports:

* one configured LLM via `[model]`
* multiple configured LLMs via `[models.<slot>]`
* `openai` or `anthropic` per slot
* model IDs supplied directly or via environment variables

If no usable LLM slots are configured, the repository still runs the heuristic baselines.

### API Keys

Only the provider you use needs a key:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

### Single-Model Setup

Example:

```toml
[model]
slot_name = "default"
provider = "anthropic"
model_env = "ANTHROPIC_MODEL"
max_output_tokens = 180
timeout_seconds = 90
max_retries = 4
initial_backoff_seconds = 1.0
max_backoff_seconds = 12.0
```

```bash
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="claude-sonnet-4-5"
```

### Three-Model Ladder Setup

Example:

```toml
[models.small]
provider = "openai"
model_env = "OPENAI_MODEL_SMALL"
max_output_tokens = 180
timeout_seconds = 90

[models.mid]
provider = "anthropic"
model_env = "ANTHROPIC_MODEL_MID"
max_output_tokens = 180
timeout_seconds = 90

[models.frontier]
provider = "openai"
model_env = "OPENAI_MODEL_FRONTIER"
max_output_tokens = 180
timeout_seconds = 90
```

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_MODEL_SMALL="gpt-5-mini"
export ANTHROPIC_MODEL_MID="claude-sonnet-4-5"
export OPENAI_MODEL_FRONTIER="gpt-5"
```

### Config Notes

* `provider` must be `openai` or `anthropic`
* each slot must define either `model_env` or `model_id`
* `slot_name` is optional for `[model]` and defaults to `default`
* `timeout_seconds`, `max_retries`, `initial_backoff_seconds`, and `max_backoff_seconds` control HTTP resilience per model slot
* retryable provider failures include transient overloads, timeouts, and connection resets; request-format and auth errors still fail immediately
* `max_output_tokens = 180` is already ample for the repository's tiny JSON responses and will not fix `529 overloaded` or timeout failures; only increase it if errors show truncation such as `stop_reason=max_tokens`
* `api_key_env` is optional; the defaults are `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`

## Running the Pipeline

Full run with `uv`:

```bash
uv run llm-experimental-economics-simulation run
```

Full run without `uv`:

```bash
PYTHONPATH=src python3 -m synthetic_replication.cli run
```

Stage-by-stage:

```bash
uv run llm-experimental-economics-simulation prepare
uv run llm-experimental-economics-simulation simulate --replicates 2
uv run llm-experimental-economics-simulation evaluate
```

Heuristic-only smoke test:

```bash
uv run llm-experimental-economics-simulation run --baseline-only --replicates 1
```

Low-cost live LLM smoke test:

```bash
uv run llm-experimental-economics-simulation simulate --smoke-test
```

## Outputs

Main artifact locations:

* `outputs/experience-based-discrimination/`
* `outputs/experience-based-discrimination/evaluation/`

## Repository Layout

```text
config/                        Runtime and model configuration
data/                          Local, ignored source and prepared data
docs/                          Technical design notes
outputs/                       Simulation and evaluation outputs
src/synthetic_replication/     Pipeline implementation
tests/                         Automated tests
README.md                      Repository documentation
```

## Development and Testing

Run the test suite:

```bash
uv run pytest -q
```

Useful local validation command:

```bash
uv run llm-experimental-economics-simulation run --baseline-only --replicates 1
```

## Limitations

Current known limits:

* v1 covers only the `Baseline` and `Control` treatments
* the prompt renderer is a structured text-card surrogate, not a literal reproduction of the original oTree screens
* the main prompts use anonymized group labels instead of the archive's color labels so relabel audits can test leakage sensitivity
* the archived Control oTree apps use `180/90` incentives, while the vendored instructions PDF and current reconstruction use `220/110`
* the archived oTree apps sample worker-productivity sequences with `random.choices`, while the pipeline keeps without-replacement resampling to match the written fixed-pool worker design
* the evaluation suite is lighter than a full appendix-style econometric replication
* provider outputs are normalized to JSON, but model behavior can still drift across releases

## References

Primary references:

* Louis-Pierre Lepage, [*Experience-based Discrimination*](https://www.aeaweb.org/articles?id=10.1257/app.20220466).
* [Official replication archive](https://www.openicpsr.org/openicpsr/project/192292/version/V1/view).
* [Pipeline design note](docs/llm-simulation-pipeline.md).
