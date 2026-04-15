from typing import Optional

import click

from replay_structure.metadata import string_to_data_type, string_to_session_indicator
from replay_structure.pipelines.modeling_pipeline import run_marginals


@click.command()
@click.option(
    "--data_type", type=click.Choice(["ripples", "run_snippets"]), default="ripples"
)
@click.option("--session", default="0")
@click.option("--spikemat_ind", type=click.INT, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--diffusion_only", is_flag=True)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    data_type: str,
    session: str,
    spikemat_ind: Optional[int],
    bin_size_cm: int,
    time_window_ms: int,
    diffusion_only: bool,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator = string_to_session_indicator(session)
    run_marginals(
        data_type=data_type_,
        session=session_indicator,
        spikemat_ind=spikemat_ind,
        bin_size_cm=bin_size_cm,
        time_window_ms=time_window_ms,
        diffusion_only=diffusion_only,
        filename_ext=filename_ext,
    )


if __name__ == "__main__":
    main()
