import click

from replay_structure.metadata import string_to_data_type
from replay_structure.pipelines.modeling_pipeline import run_descriptive_stats


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(["ripples", "high_synchrony_events"]),
    default="ripples",
)
def main(data_type: str, bin_size_cm: int = 4):
    data_type_ = string_to_data_type(data_type)
    run_descriptive_stats(data_type=data_type_, bin_size_cm=bin_size_cm)


if __name__ == "__main__":
    main()
