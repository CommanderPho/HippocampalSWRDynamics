import click

from replay_structure.metadata import string_to_data_type
from replay_structure.pipelines.modeling_pipeline import run_predictive_analysis


@click.command()
@click.option(
    "--trajectory_type",
    type=click.Choice(["viterbi", "map", "mean"]),
    default="viterbi",
)
@click.option(
    "--data_type",
    type=click.Choice(["ripples", "high_synchrony_events"]),
    default="ripples",
)
def main(trajectory_type: str, data_type: str, bin_size_cm: int = 4):
    data_type_ = string_to_data_type(data_type)
    run_predictive_analysis(data_type=data_type_, trajectory_type=trajectory_type, bin_size_cm=bin_size_cm)


if __name__ == "__main__":
    main()
