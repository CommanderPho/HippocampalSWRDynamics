from typing import Optional

import click

from replay_structure.metadata import string_to_data_type, string_to_likelihood_function, string_to_session_indicator
from replay_structure.pipelines.modeling_pipeline import run_model_comparison


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        [
            "ripples",
            "run_snippets",
            "poisson_simulated_ripples",
            "negbinomial_simulated_ripples",
            "placefieldID_shuffle",
            "placefield_rotation",
            "high_synchrony_events",
        ]
    ),
    required=True,
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--likelihood_function", type=click.STRING, default=None)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    data_type: str,
    session: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Optional[str],
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = None
    likelihood_function_ = None if likelihood_function is None else string_to_likelihood_function(likelihood_function)
    if session is not None:
        session_indicator = string_to_session_indicator(session)
    run_model_comparison(
        data_type=data_type_,
        session=session_indicator,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        likelihood_function=likelihood_function_,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    main()
