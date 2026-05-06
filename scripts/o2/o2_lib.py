# import shutil
from typing import List, Optional, Tuple

import replay_structure.structure_models_gridsearch as gridsearch
from replay_structure.structure_trajectory import Most_Likely_Trajectories
import replay_structure.read_write as read_write
from replay_structure.config import (
    Structure_Model_Gridsearch_Parameters,
    MAX_LIKELIHOOD_SD_METERS_RIPPLES,
    MAX_LIKELIHOOD_SD_METERS_RUN_SNIPPETS,
)
from replay_structure.metadata import (
    Data_Type,
    PlaceField_Rotation,
    PlaceFieldID_Shuffle,
    Ripples,
    Run_Snippets,
    Diffusion,
    Momentum,
    SessionSpikemat_Name,
    Stationary_Gaussian,
    Session_Indicator,
    Session_Name,
    Simulated_Session_Name,
    External_Session_Name,
    Likelihood_Function,
    HighSynchronyEvents,
)

DEFAULT_FILENAME_EXT = ""


def _select_diffusion_params(session_indicator: Session_Indicator, data_type: Data_Type):
    if isinstance(session_indicator, Session_Name) or isinstance(session_indicator, External_Session_Name):
        if isinstance(data_type.name, Ripples) or isinstance(data_type.name, PlaceFieldID_Shuffle) or isinstance(data_type.name, PlaceField_Rotation) or isinstance(data_type.name, HighSynchronyEvents):
            return Structure_Model_Gridsearch_Parameters.ripple_diffusion_params()
        if isinstance(data_type.name, Run_Snippets):
            return Structure_Model_Gridsearch_Parameters.run_diffusion_params()
        raise Exception("Invalid Data_Type.name type")
    if isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            return Structure_Model_Gridsearch_Parameters.ripple_diffusion_params()
        if isinstance(data_type.simulated_data_name, Run_Snippets):
            return Structure_Model_Gridsearch_Parameters.run_diffusion_params()
        raise Exception("Invalid Data_Type.simulated_data_name type")
    raise Exception("Invalid Session_Indicator type")


def _select_stationary_gaussian_params(session_indicator: Session_Indicator, data_type: Data_Type):
    if isinstance(session_indicator, Session_Name) or isinstance(session_indicator, External_Session_Name):
        if isinstance(data_type.name, Ripples) or isinstance(data_type.name, PlaceFieldID_Shuffle) or isinstance(data_type.name, PlaceField_Rotation) or isinstance(data_type.name, HighSynchronyEvents):
            return Structure_Model_Gridsearch_Parameters.ripple_stationary_gaussian_params()
        if isinstance(data_type.name, Run_Snippets):
            return Structure_Model_Gridsearch_Parameters.run_stationary_gaussian_params()
        raise Exception("Invalid Data_Type.name type")
    if isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            return Structure_Model_Gridsearch_Parameters.ripple_stationary_gaussian_params()
        if isinstance(data_type.simulated_data_name, Run_Snippets):
            return Structure_Model_Gridsearch_Parameters.run_stationary_gaussian_params()
        raise Exception("Invalid Data_Type.simulated_data_name type")
    raise Exception("Invalid Session_Indicator type")


def _select_momentum_params(session_indicator: Session_Indicator, data_type: Data_Type) -> Tuple[object, bool]:
    if isinstance(session_indicator, Session_Name) or isinstance(session_indicator, External_Session_Name):
        if isinstance(data_type.name, Ripples) or isinstance(data_type.name, PlaceFieldID_Shuffle) or isinstance(data_type.name, PlaceField_Rotation) or isinstance(data_type.name, HighSynchronyEvents):
            return Structure_Model_Gridsearch_Parameters.ripple_momentum_params(), False
        if isinstance(data_type.name, Run_Snippets):
            return Structure_Model_Gridsearch_Parameters.run_momentum_params(), False
        raise Exception("Invalid Data_Type.name type")
    if isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            return Structure_Model_Gridsearch_Parameters.ripple_momentum_params(), False
        if isinstance(data_type.simulated_data_name, Run_Snippets):
            return Structure_Model_Gridsearch_Parameters.run_momentum_params(), False
        raise Exception("Invalid Data_Type.simulated_data_name type")
    raise Exception("Invalid Session_Indicator type")


def submit_diffusion_gridsearch(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    o2: bool = True,
    filename_ext: str = "",
):
    structure_data = read_write.load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )
    if isinstance(session_indicator, Session_Name) or isinstance(
        session_indicator, External_Session_Name
    ):
        if (
            isinstance(data_type.name, Ripples)
            or isinstance(data_type.name, PlaceFieldID_Shuffle)
            or isinstance(data_type.name, PlaceField_Rotation)
            or isinstance(data_type.name, HighSynchronyEvents)
        ):
            params = Structure_Model_Gridsearch_Parameters.ripple_diffusion_params()
        elif isinstance(data_type.name, Run_Snippets):
            params = Structure_Model_Gridsearch_Parameters.run_diffusion_params()
        else:
            raise Exception("Invalid Data_Type.name type")
    elif isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            params = Structure_Model_Gridsearch_Parameters.ripple_diffusion_params()
        elif isinstance(data_type.simulated_data_name, Run_Snippets):
            params = Structure_Model_Gridsearch_Parameters.run_diffusion_params()
        else:
            raise Exception("Invalid Data_Type.simulated_data_name type")
    else:
        raise Exception("Invalid Session_Indicator type")
    diffusion_gridsearch = gridsearch.Diffusion(structure_data, params)

    read_write.save_gridsearch_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        Diffusion(),
        diffusion_gridsearch,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )


def submit_momentum_gridsearch(
    session_indicator: Session_Indicator,
    spikemat_ind: int,
    time_window_ms: int,
    data_type: Data_Type,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    o2: bool = True,
    filename_ext: str = "",
):
    structure_data = read_write.load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )
    if isinstance(session_indicator, Session_Name) or isinstance(
        session_indicator, External_Session_Name
    ):
        if (
            isinstance(data_type.name, Ripples)
            or isinstance(data_type.name, PlaceFieldID_Shuffle)
            or isinstance(data_type.name, PlaceField_Rotation)
            or isinstance(data_type.name, HighSynchronyEvents)
        ):
            params = Structure_Model_Gridsearch_Parameters.ripple_momentum_params()
            adjust_params = False
        elif isinstance(data_type.name, Run_Snippets):
            params = Structure_Model_Gridsearch_Parameters.run_momentum_params()
            adjust_params = False
        else:
            raise Exception("Invalid Data_Type.name type")
    elif isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            params = Structure_Model_Gridsearch_Parameters.ripple_momentum_params()
            adjust_params = False
        elif isinstance(data_type.simulated_data_name, Run_Snippets):
            params = Structure_Model_Gridsearch_Parameters.run_momentum_params()
            adjust_params = False
        else:
            raise Exception("Invalid Data_Type.simulated_data_name type")
    else:
        raise Exception("Invalid Session_Indicator type")
    momentum_gridsearch = gridsearch.Momentum(
        structure_data, params, spikemat_ind, adjust_params=adjust_params
    )
    read_write.save_gridsearch_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        Momentum(),
        momentum_gridsearch,
        spikemat_ind=spikemat_ind,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )


def submit_stationary_gaussian_gridsearch(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    o2: bool = True,
    filename_ext: str = "",
):
    structure_data = read_write.load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )
    if isinstance(session_indicator, Session_Name) or isinstance(
        session_indicator, External_Session_Name
    ):
        if (
            isinstance(data_type.name, Ripples)
            or isinstance(data_type.name, PlaceFieldID_Shuffle)
            or isinstance(data_type.name, PlaceField_Rotation)
            or isinstance(data_type.name, HighSynchronyEvents)
        ):
            params = (
                Structure_Model_Gridsearch_Parameters.ripple_stationary_gaussian_params()
            )
        elif isinstance(data_type.name, Run_Snippets):
            params = (
                Structure_Model_Gridsearch_Parameters.run_stationary_gaussian_params()
            )
        else:
            raise Exception("Invalid Data_Type.name type")
    elif isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            params = (
                Structure_Model_Gridsearch_Parameters.ripple_stationary_gaussian_params()
            )
        elif isinstance(data_type.simulated_data_name, Run_Snippets):
            params = (
                Structure_Model_Gridsearch_Parameters.run_stationary_gaussian_params()
            )
        else:
            raise Exception("Invalid Data_Type.simulated_data_name type")
    else:
        raise Exception("Invalid Session_Indicator type")
    sg_gridsearch = gridsearch.Stationary_Gaussian(structure_data, params)
    read_write.save_gridsearch_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        Stationary_Gaussian(),
        sg_gridsearch,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )


def submit_all_ripple_gridsearches(session_indicator: Session_Indicator, time_window_ms: int, data_type: Data_Type, likelihood_function: Likelihood_Function, spikemat_indices: List[int], bin_size_cm: int = 4, o2: bool = True, filename_ext: str = "", continue_on_error: bool = False) -> Tuple[int, List[str]]:
    """Run diffusion, stationary_gaussian, and per-spikemat momentum gridsearches with one structure-data load.

    Replaces three separate `submit_*_gridsearch` calls (each of which previously reloaded the
    structure data from disk) with a single load and an in-process iteration. Saved artifact
    paths/contents are unchanged and remain compatible with `replay_structure.read_write` callers.

    Returns (n_momentum_successes, momentum_error_messages). Diffusion and stationary_gaussian
    failures still propagate as exceptions to match prior behavior.
    """
    print(f"Loading structure data once for batched ripple gridsearches: session={session_indicator}, bin_size_cm={bin_size_cm}, time_window_ms={time_window_ms}")
    structure_data = read_write.load_structure_data(session_indicator, time_window_ms, data_type.name, likelihood_function, bin_size_cm=bin_size_cm, o2=o2, ext=filename_ext)

    print("Running diffusion gridsearch...")
    diffusion_params = _select_diffusion_params(session_indicator, data_type)
    diffusion_gridsearch = gridsearch.Diffusion(structure_data, diffusion_params)
    read_write.save_gridsearch_results(session_indicator, time_window_ms, data_type.name, likelihood_function, Diffusion(), diffusion_gridsearch, bin_size_cm=bin_size_cm, o2=o2, ext=filename_ext)

    print("Running stationary_gaussian gridsearch...")
    sg_params = _select_stationary_gaussian_params(session_indicator, data_type)
    sg_gridsearch = gridsearch.Stationary_Gaussian(structure_data, sg_params)
    read_write.save_gridsearch_results(session_indicator, time_window_ms, data_type.name, likelihood_function, Stationary_Gaussian(), sg_gridsearch, bin_size_cm=bin_size_cm, o2=o2, ext=filename_ext)

    n_successes = 0
    momentum_errors: List[str] = []
    for spikemat_ind in spikemat_indices:
        print(f"Running momentum gridsearch for spikemat {spikemat_ind}...")
        try:
            momentum_params, adjust_params = _select_momentum_params(session_indicator, data_type)
            momentum_gridsearch = gridsearch.Momentum(structure_data, momentum_params, spikemat_ind, adjust_params=adjust_params)
            read_write.save_gridsearch_results(session_indicator, time_window_ms, data_type.name, likelihood_function, Momentum(), momentum_gridsearch, spikemat_ind=spikemat_ind, bin_size_cm=bin_size_cm, o2=o2, ext=filename_ext)
            n_successes += 1
        except Exception as exc:
            error_message = f"spikemat {spikemat_ind}: {exc}"
            momentum_errors.append(error_message)
            print(f"[WARN] momentum gridsearch failed for {error_message}")
            if not continue_on_error:
                raise RuntimeError(error_message) from exc
    return n_successes, momentum_errors


def submit_viterbi(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    o2: bool = True,
    filename_ext: str = "",
):
    structure_data = read_write.load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )
    if isinstance(session_indicator, Session_Name) or isinstance(
        session_indicator, External_Session_Name
    ):
        if (
            isinstance(data_type.name, Ripples)
            or isinstance(data_type.name, PlaceFieldID_Shuffle)
            or isinstance(data_type.name, PlaceField_Rotation)
            or isinstance(data_type.name, HighSynchronyEvents)
        ):
            params = MAX_LIKELIHOOD_SD_METERS_RIPPLES
        elif isinstance(data_type.name, Run_Snippets):
            params = MAX_LIKELIHOOD_SD_METERS_RUN_SNIPPETS
        else:
            raise Exception("Invalid Data_Type.name type")
    elif isinstance(session_indicator, Simulated_Session_Name):
        if isinstance(data_type.simulated_data_name, Ripples):
            params = MAX_LIKELIHOOD_SD_METERS_RIPPLES
        elif isinstance(data_type.simulated_data_name, Run_Snippets):
            params = MAX_LIKELIHOOD_SD_METERS_RUN_SNIPPETS
        else:
            raise Exception("Invalid Data_Type.simulated_data_name type")
    elif isinstance(session_indicator, SessionSpikemat_Name):
        params = MAX_LIKELIHOOD_SD_METERS_RIPPLES
    else:
        raise Exception("Invalid Session_Indicator type")
    trajectories = Most_Likely_Trajectories(structure_data, params)
    read_write.save_trajectory_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        trajectories,
        bin_size_cm=bin_size_cm,
        o2=o2,
        ext=filename_ext,
    )
