from __future__ import annotations

from pathlib import Path

import click

from synthetic_replication.config import load_config, project_root
from synthetic_replication.papers import PAPER_ID, evaluate_simulations, prepare_paper, simulate_paper


def _resolve_root() -> Path:
    return project_root()


@click.group()
def main() -> None:
    """Synthetic replication pipeline."""


@main.command()
@click.option("--config-path", default=None, help="Path to pipeline TOML config.")
@click.option("--paper-id", default=PAPER_ID, show_default=True, help="Paper identifier.")
def prepare(config_path: str | None, paper_id: str) -> None:
    if paper_id != PAPER_ID:
        raise click.ClickException(f"Unsupported paper id: {paper_id}")
    config = load_config(config_path)
    manifest = prepare_paper(config, _resolve_root())
    click.echo(f"Prepared {paper_id}: {manifest['sample_sizes']}")


@main.command()
@click.option("--config-path", default=None, help="Path to pipeline TOML config.")
@click.option("--paper-id", default=PAPER_ID, show_default=True, help="Paper identifier.")
@click.option("--replicates", type=int, default=None, help="Override the configured replicate count.")
@click.option("--baseline-only", is_flag=True, help="Run only heuristic baselines.")
@click.option(
    "--smoke-test",
    is_flag=True,
    help="Minimize API workload: force one replicate, one sampled employer per treatment, and smallest audits.",
)
def simulate(
    config_path: str | None,
    paper_id: str,
    replicates: int | None,
    baseline_only: bool,
    smoke_test: bool,
) -> None:
    if paper_id != PAPER_ID:
        raise click.ClickException(f"Unsupported paper id: {paper_id}")
    config = load_config(config_path)
    manifest = simulate_paper(
        config,
        _resolve_root(),
        replicates=replicates,
        baseline_only=baseline_only,
        smoke_test=smoke_test,
    )
    click.echo(f"Simulated {paper_id}: {manifest['simulation_records']}")


@main.command()
@click.option("--config-path", default=None, help="Path to pipeline TOML config.")
@click.option("--paper-id", default=PAPER_ID, show_default=True, help="Paper identifier.")
def evaluate(config_path: str | None, paper_id: str) -> None:
    if paper_id != PAPER_ID:
        raise click.ClickException(f"Unsupported paper id: {paper_id}")
    config = load_config(config_path)
    summary = evaluate_simulations(config, _resolve_root())
    click.echo(f"Evaluated {paper_id}: {summary['comparison_report']}")


@main.command()
@click.option("--config-path", default=None, help="Path to pipeline TOML config.")
@click.option("--paper-id", default=PAPER_ID, show_default=True, help="Paper identifier.")
@click.option("--replicates", type=int, default=None, help="Override the configured replicate count.")
@click.option("--baseline-only", is_flag=True, help="Run only heuristic baselines.")
@click.option(
    "--smoke-test",
    is_flag=True,
    help="Minimize API workload: force one replicate, one sampled employer per treatment, and smallest audits.",
)
def run(
    config_path: str | None,
    paper_id: str,
    replicates: int | None,
    baseline_only: bool,
    smoke_test: bool,
) -> None:
    if paper_id != PAPER_ID:
        raise click.ClickException(f"Unsupported paper id: {paper_id}")
    config = load_config(config_path)
    prepare_paper(config, _resolve_root())
    simulate_paper(
        config,
        _resolve_root(),
        replicates=replicates,
        baseline_only=baseline_only,
        smoke_test=smoke_test,
    )
    summary = evaluate_simulations(config, _resolve_root())
    click.echo(f"Completed {paper_id}: {summary['comparison_report']}")


if __name__ == "__main__":
    main()
