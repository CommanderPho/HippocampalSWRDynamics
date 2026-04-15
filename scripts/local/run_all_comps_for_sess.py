import click
from replay_structure.metadata import SESSION_RATDAY
from replay_structure.pipelines.session_pipeline import (
    DEFAULT_LIKELIHOOD,
    DEFAULT_TIME_WINDOW_MS,
    DEFAULT_TRAJECTORY_SD_METERS,
    run_session_pipeline,
)


@click.command()
@click.option("--session", type=click.IntRange(0, len(SESSION_RATDAY) - 1), required=True)
@click.option("--bin_size_cm", type=click.INT, default=4, show_default=True)
@click.option("--time_window_ms", type=click.INT, default=DEFAULT_TIME_WINDOW_MS, show_default=True)
@click.option("--likelihood_function", type=click.Choice(["poisson", "negbinomial"]), default=DEFAULT_LIKELIHOOD, show_default=True)
@click.option("--filename_ext", type=click.STRING, default="", show_default=True)
@click.option("--trajectory_sd_meters", type=click.FLOAT, default=DEFAULT_TRAJECTORY_SD_METERS, show_default=True)
@click.option("--skip-gridsearch", is_flag=True, help="Assume gridsearch result inputs already exist.")
@click.option("--force-ratday-preprocess", is_flag=True, help="Regenerate ratday data from OpenFieldData.mat even if a saved ratday .obj already exists.")
@click.option("--run-marginals", is_flag=True, help="Run get_marginals.py after the core model comparison pipeline.")
@click.option("--run-diffusion-constant", is_flag=True, help="Run run_diffusion_constant.py after trajectories are computed.")
@click.option("--continue-on-gridsearch-error", is_flag=True, help="Continue momentum gridsearch after per-spikemat failures when possible.")
@click.option("--strict", is_flag=True, help="Stop immediately on the first step failure.")
@click.option("--dry-run", is_flag=True, help="Print the planned commands and phases without executing them.")
def main(
    session: int,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: str,
    filename_ext: str,
    trajectory_sd_meters: float,
    skip_gridsearch: bool,
    force_ratday_preprocess: bool,
    run_marginals: bool,
    run_diffusion_constant: bool,
    continue_on_gridsearch_error: bool,
    strict: bool,
    dry_run: bool,
) -> None:
    """
    
    
    session: int = 1
    bin_size_cm: int = 4
    time_window_ms: int = DEFAULT_TIME_WINDOW_MS # 0.003
    likelihood_function: str = DEFAULT_LIKELIHOOD
    filename_ext: str = ''
    trajectory_sd_meters: float = DEFAULT_TRAJECTORY_SD_METERS
    skip_gridsearch: bool = True
    force_ratday_preprocess: bool = True
    run_marginals: bool = True
    run_diffusion_constant: bool = True
    continue_on_gridsearch_error: bool = False
    strict: bool = False
    dry_run: bool = False

    """
    run_session_pipeline(
        session=session,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        likelihood_function=likelihood_function,
        filename_ext=filename_ext,
        trajectory_sd_meters=trajectory_sd_meters,
        skip_gridsearch=skip_gridsearch,
        force_ratday_preprocess=force_ratday_preprocess,
        run_marginals_phase=run_marginals,
        run_diffusion_constant_phase=run_diffusion_constant,
        continue_on_gridsearch_error=continue_on_gridsearch_error,
        strict=strict,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
