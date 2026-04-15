from typing import Optional

import click

from replay_structure.metadata import Session_Name, string_to_session_indicator
from replay_structure.pipelines.preprocessing_pipeline import run_preprocess_ratday


@click.command()
@click.option("--session", type=click.INT, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--filename_ext", default="")
@click.option("--rotate_placefields", type=click.BOOL, default=False)
def main(
    session: Optional[int],
    bin_size_cm: int,
    filename_ext: str,
    rotate_placefields: bool,
):
    session_indicator = None
    if session is not None:
        session_indicator: Session_Name = string_to_session_indicator(session)
        assert isinstance(session_indicator, Session_Name)
    run_preprocess_ratday(session=session_indicator, bin_size_cm=bin_size_cm, filename_ext=filename_ext, rotate_placefields=rotate_placefields)


if __name__ == "__main__":
    main()
