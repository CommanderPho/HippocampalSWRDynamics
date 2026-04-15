from typing import Optional, Union

import click

from replay_structure.metadata import (
    Likelihood_Function,
    Session_Indicator,
    string_to_data_type,
    string_to_likelihood_function,
    string_to_session_indicator,
)
from replay_structure.pipelines.structure_analysis_pipeline import run_structure_analysis_reformat


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        [
            "ripples",
            "run_snippets",
            "poisson_simulated_ripples",
            "negbinomial_simulated_ripples",
            "ripples_pf",
            "placefieldID_shuffle",
            "placefield_rotation",
            "high_synchrony_events",
            "high_synchrony_events_pf",
        ]
    ),
    required=True,
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", default=None)
@click.option(
    "--likelihood_function", type=click.Choice(["poisson", "negbinomial"]), default=None
)
@click.option("--filename_ext", type=click.STRING, default="")
@click.option(
    "--external_source_format",
    type=click.Choice(
        [
            "canonical",
            "neuropy_epochs_spkcount",
            "pypho_decoded_epochs",
            "replayswitchinghmm",
        ]
    ),
    default=None,
)
@click.option("--external_source_path", type=click.Path(exists=True), default=None)
def main(
    data_type: str,
    session: Optional[Union[int, str]],
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: str,
    filename_ext: str,
    external_source_format: Optional[str],
    external_source_path: Optional[str],
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = None
    likelihood_function_: Optional[Likelihood_Function] = None
    if likelihood_function is not None:
        likelihood_function_ = string_to_likelihood_function(likelihood_function)
    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
    run_structure_analysis_reformat(
        data_type=data_type_,
        session=session_indicator,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        likelihood_function=likelihood_function_,
        filename_ext=filename_ext,
        external_source_format=external_source_format,
        external_source_path=external_source_path,
    )


if __name__ == "__main__":
    main()
