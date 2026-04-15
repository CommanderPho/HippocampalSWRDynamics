from typing import Optional

import click

from replay_structure.metadata import Session_Name, string_to_data_type, string_to_session_indicator
from replay_structure.pipelines.preprocessing_pipeline import run_preprocess_spikemat


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        [
            "ripples",
            "run_snippets",
            "ripples_pf",
            "high_synchrony_events_pf",
            "placefieldID_shuffle",
            "placefield_rotation",
            "high_synchrony_events",
        ]
    ),
    required=True,
)
@click.option("--session", type=click.INT, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--filename_ext", default="")
def main(
    data_type: str,
    session: Optional[int],
    bin_size_cm: int,
    time_window_ms: Optional[int],
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = None
    if session is not None:
        session_indicator: Session_Name = string_to_session_indicator(session)
        assert isinstance(session_indicator, Session_Name)
    run_preprocess_spikemat(
        data_type=data_type_,
        session=session_indicator,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    main()
