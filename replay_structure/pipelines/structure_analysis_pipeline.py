from typing import Optional

from replay_structure.external_event_adapters import (
    build_payload_from_neuropy_epochs_spkcount,
    build_payload_from_pypho_decoded_epochs_result,
    build_payload_from_replayswitchinghmm,
)
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.metadata import (
    Data_Type,
    HighSynchronyEvents,
    HighSynchronyEvents_PF,
    Likelihood_Function,
    NegBinomial_Simulated_Ripples,
    PlaceField_Rotation,
    PlaceFieldID_Shuffle,
    Poisson_Simulated_Ripples,
    Ripples,
    Ripples_PF,
    Run_Snippets,
    Session_Indicator,
    Session_List,
)
from replay_structure.read_write import load_data, load_spikemat_data, load_structure_data, save_structure_data
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing
from replay_structure.simulated_neural_data import Simulated_Data_Preprocessing
from replay_structure.structure_analysis_input import Structure_Analysis_Input


def run_structure_analysis_preprocessing(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    filename_ext: str,
) -> None:
    print(f"running session {session_indicator} with {bin_size_cm}cm binsand {time_window_ms}ms time window.")
    spikemat_data = load_spikemat_data(
        session_indicator, time_window_ms, data_type.name, bin_size_cm=bin_size_cm, ext=filename_ext
    )
    if (
        isinstance(data_type.name, Ripples)
        or isinstance(data_type.name, PlaceFieldID_Shuffle)
        or isinstance(data_type.name, PlaceField_Rotation)
    ):
        assert isinstance(spikemat_data, Ripple_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_ripple_data(spikemat_data, likelihood_function)
    elif isinstance(data_type.name, Run_Snippets):
        assert isinstance(spikemat_data, Run_Snippet_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_run_snippet_data(spikemat_data, likelihood_function)
    elif isinstance(data_type.name, HighSynchronyEvents):
        assert isinstance(spikemat_data, HighSynchronyEvents_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_highsynchrony_data(spikemat_data, likelihood_function)
    elif isinstance(data_type.name, Ripples_PF):
        assert isinstance(spikemat_data, Ripple_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_pfanalysis_data(
            spikemat_data, likelihood_function, select_population_burst=False
        )
    elif isinstance(data_type.name, HighSynchronyEvents_PF):
        assert isinstance(spikemat_data, HighSynchronyEvents_Preprocessing)
        structure_analysis_input = Structure_Analysis_Input.reformat_highsynchronypf_data(spikemat_data, likelihood_function)
    elif isinstance(data_type.name, Poisson_Simulated_Ripples) or isinstance(data_type.name, NegBinomial_Simulated_Ripples):
        assert isinstance(spikemat_data, Simulated_Data_Preprocessing)
        assert data_type.simulated_data_name is not None
        structure_data = load_structure_data(
            Session_List[0], data_type.default_time_window_ms, data_type.simulated_data_name, likelihood_function
        )
        likelihood_function_params = structure_data.params.likelihood_function_params
        structure_analysis_input = Structure_Analysis_Input.reformat_simulated_data(spikemat_data, likelihood_function_params)
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
    print(f"running external session {session_indicator} on {data_type.name} data from {external_source_format}")
    external_source = load_data(external_source_path)
    external_payload = _build_external_payload(external_source, external_source_format)
    structure_analysis_input = Structure_Analysis_Input.reformat_external_data(
        external_payload, likelihood_function=likelihood_function
    )
    if (time_window_ms is not None) and (time_window_ms != structure_analysis_input.params.time_window_ms):
        raise ValueError("CLI time_window_ms does not match the external payload time window")
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


def run_structure_analysis_reformat(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    likelihood_function: Optional[Likelihood_Function] = None,
    filename_ext: str = "",
    external_source_format: Optional[str] = None,
    external_source_path: Optional[str] = None,
) -> None:
    if (time_window_ms is None) and (external_source_path is None):
        time_window_ms = data_type.default_time_window_ms
    if likelihood_function is None:
        likelihood_function = data_type.default_likelihood_function
    print(likelihood_function)
    if (external_source_format is None) != (external_source_path is None):
        raise ValueError("external_source_format and external_source_path must be provided together")
    if session is not None:
        if external_source_path is None:
            assert time_window_ms is not None
            run_structure_analysis_preprocessing(
                data_type, session, bin_size_cm, time_window_ms, likelihood_function, filename_ext
            )
            return
        assert external_source_format is not None
        run_external_structure_analysis_preprocessing(
            data_type,
            session,
            bin_size_cm,
            time_window_ms,
            likelihood_function,
            filename_ext,
            external_source_path,
            external_source_format,
        )
        return
    if external_source_path is not None:
        raise ValueError("external preprocessing requires an explicit session name")
    assert time_window_ms is not None
    for session_indicator in data_type.session_list:
        run_structure_analysis_preprocessing(
            data_type, session_indicator, bin_size_cm, time_window_ms, likelihood_function, filename_ext
        )
