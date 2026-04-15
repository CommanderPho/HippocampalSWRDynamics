from typing import Optional

import click

from replay_structure.metadata import string_to_data_type, string_to_session_indicator
from replay_structure.pipelines.modeling_pipeline import run_diffusion_constant_analysis


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(["ripples", "run_snippets", "poisson_simulated_ripples"]),
    required=True,
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option(
    "--trajectory_type", type=click.Choice(["true", "inferred"]), required=True
)
@click.option("--bin_space", is_flag=True)
@click.option("--filename_ext", type=click.STRING, default="")
def run(
    data_type: str,
    session: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    trajectory_type: str,
    bin_space: bool,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = None
    if session is not None:
        session_indicator = string_to_session_indicator(session)
    run_diffusion_constant_analysis(
        data_type=data_type_,
        session=session_indicator,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        trajectory_type=trajectory_type,
        bin_space=bin_space,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    run()
