import click
from typing import Optional, Union

from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.simulated_neural_data import Simulated_Data_Preprocessing
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.read_write import (
    load_spikemat_data,
    load_data,
    save_structure_data,
    load_structure_data,
)
from replay_structure.external_event_adapters import (
    build_payload_from_neuropy_epochs_spkcount,
    build_payload_from_pypho_decoded_epochs_result,
    build_payload_from_replayswitchinghmm,
)
from replay_structure.metadata import (
    Likelihood_Function,
    Session_Indicator,
    Session_List,
    Data_Type,
    Ripples,
    PlaceField_Rotation,
    PlaceFieldID_Shuffle,
    Ripples_PF,
    Run_Snippets,
    HighSynchronyEvents,
    HighSynchronyEvents_PF,
    Poisson_Simulated_Ripples,
    NegBinomial_Simulated_Ripples,
    string_to_data_type,
    string_to_session_indicator,
    string_to_likelihood_function,
)


def run_structure_analysis_preprocessing(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    filename_ext: str,
) -> None:
    print(
        f"running session {session_indicator} with {bin_size_cm}cm bins"
        f"and {time_window_ms}ms time window."
    )
    spikemat_data = load_spikemat_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )

    if (
        isinstance(data_type.name, Ripples)
        or isinstance(data_type.name, PlaceFieldID_Shuffle)
        or isinstance(data_type.name, PlaceField_Rotation)
    ):
        assert isinstance(spikemat_data, Ripple_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_ripple_data(
            spikemat_data, likelihood_function
        )
    elif isinstance(data_type.name, Run_Snippets):
        assert isinstance(spikemat_data, Run_Snippet_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_run_snippet_data(
            spikemat_data, likelihood_function
        )
    elif isinstance(data_type.name, HighSynchronyEvents):
        assert isinstance(spikemat_data, HighSynchronyEvents_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_highsynchrony_data(
            spikemat_data, likelihood_function
        )
    elif isinstance(data_type.name, Ripples_PF):
        assert isinstance(spikemat_data, Ripple_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_pfanalysis_data(
            spikemat_data, likelihood_function, select_population_burst=False
        )
    elif isinstance(data_type.name, HighSynchronyEvents_PF):
        assert isinstance(spikemat_data, HighSynchronyEvents_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_highsynchronypf_data(
            spikemat_data, likelihood_function
        )
    elif isinstance(data_type.name, Poisson_Simulated_Ripples) or isinstance(
        data_type.name, NegBinomial_Simulated_Ripples
    ):
        assert isinstance(spikemat_data, Simulated_Data_Preprocessing)
        assert data_type.simulated_data_name is not None
        # get likelihood function params from real data session 0
        structure_data = load_structure_data(
            Session_List[0],
            data_type.default_time_window_ms,
            data_type.simulated_data_name,
            likelihood_function,
        )
        likelihood_function_params = structure_data.params.likelihood_function_params
        structure_analysis_input = Structure_Analysis_Input.reformat_simulated_data(
            spikemat_data, likelihood_function_params
        )
    else:
        raise AttributeError("Invalid data_type.")

    save_structure_data(
        structure_analysis_input,
        session_indicator,
        structure_analysis_input.params.time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=structure_analysis_input.params.bin_size_cm,
        ext=filename_ext,
    )


def _build_external_payload(external_source: dict, external_source_format: str) -> dict:
    if external_source_format == "canonical":
        return external_source
    if external_source_format == "neuropy_epochs_spkcount":
        return build_payload_from_neuropy_epochs_spkcount(**external_source)
    if external_source_format == "pypho_decoded_epochs":
        return build_payload_from_pypho_decoded_epochs_result(**external_source)
    if external_source_format == "replayswitchinghmm":
        return build_payload_from_replayswitchinghmm(**external_source)
    raise ValueError(f"invalid external_source_format: {external_source_format}")


def run_external_structure_analysis_preprocessing(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: Optional[int],
    likelihood_function: Likelihood_Function,
    filename_ext: str,
    external_source_path: str,
    external_source_format: str,
) -> None:
    print(
        f"running external session {session_indicator} on {data_type.name} data "
        f"from {external_source_format}"
    )
    external_source = load_data(external_source_path)
    external_payload = _build_external_payload(external_source, external_source_format)
    structure_analysis_input = Structure_Analysis_Input.reformat_external_data(
        external_payload, likelihood_function=likelihood_function
    )
    if (
        (time_window_ms is not None)
        and (time_window_ms != structure_analysis_input.params.time_window_ms)
    ):
        raise ValueError(
            "CLI time_window_ms does not match the external payload time window"
        )
    if bin_size_cm != structure_analysis_input.params.bin_size_cm:
        raise ValueError("CLI bin_size_cm does not match the external payload bin size")
    diagnostics = structure_analysis_input.source_metadata.get("diagnostics", {})
    print(f"external input diagnostics: {diagnostics}")
    save_structure_data(
        structure_analysis_input,
        session_indicator,
        structure_analysis_input.params.time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=structure_analysis_input.params.bin_size_cm,
        ext=filename_ext,
    )


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

    if (time_window_ms is None) and (external_source_path is None):
        time_window_ms = data_type_.default_time_window_ms

    if likelihood_function is None:
        likelihood_function_: Likelihood_Function = (
            data_type_.default_likelihood_function
        )

    else:
        likelihood_function_ = string_to_likelihood_function(likelihood_function)
    print(likelihood_function_)

    if (external_source_format is None) != (external_source_path is None):
        raise ValueError(
            "external_source_format and external_source_path must be provided together"
        )

    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
        if external_source_path is None:
            run_structure_analysis_preprocessing(
                data_type_,
                session_indicator,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                filename_ext,
            )
        else:
            assert external_source_format is not None
            run_external_structure_analysis_preprocessing(
                data_type_,
                session_indicator,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                filename_ext,
                external_source_path,
                external_source_format,
            )
    else:
        if external_source_path is not None:
            raise ValueError("external preprocessing requires an explicit session name")
        for session_indicator in data_type_.session_list:
            run_structure_analysis_preprocessing(
                data_type_,
                session_indicator,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                filename_ext,
            )


if __name__ == "__main__":
    main()
