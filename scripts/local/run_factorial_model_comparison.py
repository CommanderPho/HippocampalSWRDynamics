from typing import Optional

import click

from replay_structure.metadata import string_to_data_type, string_to_session_indicator
from replay_structure.pipelines.modeling_pipeline import run_factorial_model_comparison


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        ["ripples", "poisson_simulated_ripples", "negbinomial_simulated_ripples"]
    ),
    default="ripples",
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=3)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    data_type: str,
    session: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = None
    if session is not None:
        session_indicator = string_to_session_indicator(session)
    run_factorial_model_comparison(
        data_type=data_type_,
        session=session_indicator,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    main()
