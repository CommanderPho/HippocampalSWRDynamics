from typing import Optional

import click

from replay_structure.metadata import string_to_data_type, string_to_session_indicator
from replay_structure.pipelines.modeling_pipeline import run_pf_analysis


@click.command()
@click.option("--session", type=click.INT, default=None)
@click.option(
    "--data_type",
    type=click.Choice(["ripples_pf", "high_synchrony_events_pf"]),
    required=True,
)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--filename_ext", type=click.STRING, default="")
@click.option("--decoding_type", type=click.Choice(["map", "mean"]), default="map")
def main(
    session: Optional[int],
    data_type: str,
    bin_size_cm: int,
    filename_ext: str,
    decoding_type: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = None
    if session is not None:
        session_indicator = string_to_session_indicator(session)
    run_pf_analysis(
        data_type=data_type_,
        session=session_indicator,
        bin_size_cm=bin_size_cm,
        decoding_type=decoding_type,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    main()
