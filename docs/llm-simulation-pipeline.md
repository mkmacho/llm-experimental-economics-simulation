# LLM Simulation Pipeline: `Experience-based Discrimination`

## Scope

This repository now implements the v1 pipeline for the selected paper `Experience-based Discrimination`.

v1 scope is intentionally narrow:

* Treatments in scope: `Baseline` and `Control`
* Horizon: `15` employer hiring rounds
* Worker side: empirical resampling from the observed worker pool
* Employer side: empirical personas sampled from the cleaned core-treatment employer sample
* Benchmarks in scope: `always_A`, `always_B`, `uniform_random`, plus one or more configured OpenAI or Anthropic model slots when configured

Out of scope for v1:

* `Exploration`
* `Information`
* `Equal`
* `Gender`
* `Elicitation-only` extensions

## Design Summary

The implemented pipeline follows the project constraints recorded during its
development. The original private brief is intentionally not included in this
public-facing repository.

* It prepares frozen local tables from public archive data.
* It runs end to end from one CLI namespace with `prepare`, `simulate`, `evaluate`, and `run`.
* It keeps prompt inputs blind to paper metadata.
* It evaluates both micro-behavior and paper-level conclusions.
* It records project decisions in both markdown and JSONL form.

## Prepare Stage

`prepare` reads locally acquired raw files in `data/raw/experience-based-discrimination`:

* `data.dta`
* `dataworker.dta`
* `archive-docs/Experimental Instructions.pdf`
* `archive-docs/README.pdf`
* `archive-docs/Python/`
* `archive-docs/Stata and R/`

The executable prepare step consumes `data.dta` and `dataworker.dta`. The vendored PDFs, oTree code, and Stata/R scripts are retained as audit references for the reconstruction and for later appendix-level extensions.

It rebuilds the cleaned core sample from `data.dta` using the published `data_clean.do` logic, with one deliberate correction:

* employers are keyed by `empgroup:id`, not by the numeric `id` alone

This correction matters because the raw archive reuses numeric ids across treatments. Using `empgroup:id` avoids cross-treatment leakage in derived fields like final beliefs.

The frozen outputs are:

* `data/normalized/experience-based-discrimination/employer_profiles.csv`
* `data/normalized/experience-based-discrimination/worker_pool_empirical.csv`
* `data/normalized/experience-based-discrimination/observed_core_outcomes.csv`
* `data/normalized/experience-based-discrimination/paper_spec.json`
* `data/normalized/experience-based-discrimination/observed_targets.json`
* `data/normalized/experience-based-discrimination/prepare_manifest.json`

The current normalized core sample is:

* `297` Baseline employers
* `135` Control employers
* `15` rounds per employer

## Simulate Stage

`simulate` samples a fresh experimental replicate by:

1. Sampling employer personas with replacement within treatment.
2. Sampling a without-replacement Group-B productivity sequence for each simulated employer from the empirical worker pool.
3. Running a per-round state machine.
4. Logging rendered prompts, raw responses, parsed outputs, seeds, and metadata.

The model-visible stimulus is a hybrid text card:

* structured internally
* screen-like externally
* restricted to what a real employer would know

The prompt surface never includes:

* the paper title
* author names
* DOI or journal metadata
* archive filenames
* reported coefficients or headline findings

Current outputs:

* `outputs/experience-based-discrimination/simulation_records.csv`
* `outputs/experience-based-discrimination/interaction_logs.jsonl`
* `outputs/experience-based-discrimination/simulation_manifest.json`

These files are generated locally and intentionally not version-controlled.

## Deliberate Abstractions

Two abstractions are intentional in the current renderer and are now part of the documented design:

* The model sees a structured text card rather than a literal replay of the original oTree HTML screens.
* The model sees canonical or neutral group labels rather than the archive's original color labels.

These choices are deliberate because the simulation target is the economically relevant information state, not the exact screen chrome. The renderer preserves:

* role and treatment
* round number and allowed actions
* fixed versus uncertain productivity structure
* current carried belief
* payoff-relevant history visible to the employer

The renderer intentionally drops:

* HTML layout and styling
* color-language details that are not necessary for the task logic
* comprehension-question scaffolding
* paper metadata that would increase leakage risk

The neutral relabeling path is also used directly in the contamination audits.

## Archive Reconstruction Audit

The newly vendored archive `Python` and `Stata and R` directories were used to audit the current reconstruction. The main results are:

* The Baseline oTree app, templates, and vendored instructions all support the current core structure: `15` rounds, `75/25` group split, known group productivity of `9`, and belief elicitation only after minority-group hires.
* The Control oTree app supports the current forced-exposure interpretation and round-by-round belief elicitation, which matches the simulation state machine.
* The Stata cleaning and analysis scripts support the current evaluation targets, including cumulative uncertain-group hires, first positive and negative experience indicators, `remh`, `finalbel`, `prior`, `amb`, and the Baseline versus Control comparison.
* The archived Control oTree apps use `180/90` incentives, while the vendored participant instructions and current reconstruction use `220/110`. The pipeline currently keeps the participant-facing `220/110` reconstruction and documents the discrepancy rather than silently switching constants.
* The archived oTree apps generate worker-productivity sequences with `random.choices(..., k=16)`. The pipeline intentionally keeps without-replacement worker resampling because the written task describes a fixed worker pool where each worker is hired at most once.
* The published Stata cleaning logic mostly groups by `id empgroup`, but some later steps use `id` alone. The repository keeps `empgroup:id` throughout because the raw archive reuses numeric ids across treatments.

The main repository README documents the source archive and data-access boundary.

## Evaluate Stage

`evaluate` reports distributional fit and inferential fit separately.

Distributional targets include:

* initial prior distribution by treatment
* final belief distribution by treatment
* round-level hire-B rates

Inferential targets include:

* `Control - Baseline` final-belief gap among employers with at least one B hire
* first positive versus first negative experience effect on future B hiring
* first positive versus first negative experience effect on final beliefs
* positive versus negative experience effect on next-round B hiring

Evaluation outputs:

* per-agent markdown reports in `outputs/experience-based-discrimination/evaluation/`
* `outputs/experience-based-discrimination/evaluation/comparison.md`
* `outputs/experience-based-discrimination/evaluation/summary.json`

## Contamination Safeguards

The current implementation includes four safeguards:

* blind prompts in the main simulation loop
* a separate paper-recall probe for configured LLM agents
* relabel audits comparing canonical and neutral group names
* payoff-perturbation audits that change the fixed Group-A benchmark

These audits are recorded in the simulation manifest and copied into each evaluation report.

## CLI

With `uv` or a local Python environment:

```bash
uv run llm-experimental-economics-simulation prepare
uv run llm-experimental-economics-simulation simulate --baseline-only --replicates 2
uv run llm-experimental-economics-simulation evaluate
uv run llm-experimental-economics-simulation run --baseline-only --replicates 2
```

Without `uv`, the equivalent local command is:

```bash
PYTHONPATH=src python3 -m synthetic_replication.cli run --baseline-only --replicates 2
```

## Current Caveats

* The implemented prompt remains a structured economic surrogate rather than a literal oTree screen replay, by design.
* The archived Control oTree code conflicts with the vendored participant instructions on whether the control incentives are `180/90` or `220/110`; the current pipeline keeps the participant-facing `220/110` version and documents that choice in the audit note.
* The provider layer now supports OpenAI and Anthropic, but only configured slots with a matching API key will run.
* The current evaluation reports are honest but still lightweight compared with the full paper appendix; Table-2-style regression replications are deferred to a later pass.
