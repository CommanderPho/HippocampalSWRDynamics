from typing import Optional

import click

from replay_structure.metadata import MODELS_AS_STR, Data_Type, Simulated_Session_Name, string_to_data_type, string_to_session_indicator
from replay_structure.pipelines.modeling_pipeline import generate_model_recovery_data


@click.command()
@click.option("--model", type=click.Choice(MODELS_AS_STR), default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option(
    "--data_type",
    type=click.Choice(["poisson_simulated_ripples", "negbinomial_simulated_ripples"]),
    default="simulated_ripples",
)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    model: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    data_type: str,
    filename_ext: str,
):
    data_type_: Data_Type = string_to_data_type(data_type)
    session_indicator = None
    if model is not None:
        session_indicator = string_to_session_indicator(model)
        assert isinstance(session_indicator, Simulated_Session_Name)
    generate_model_recovery_data(
        data_type=data_type_,
        model=session_indicator,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    main()
