import os
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sp

import replay_structure.predictive_analysis as pred
import replay_structure.structure_models as models
from replay_structure.descriptive_stats import get_descriptive_stats as build_descriptive_stats
from replay_structure.descriptive_stats import get_descriptive_stats_hse
from replay_structure.deviance_models import Deviance_Explained
from replay_structure.diffusion_constant import Diffusion_Constant
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.marginals import All_Models_Marginals
from replay_structure.metadata import (
    MODELS,
    Data_Type,
    Diffusion,
    External_Session_Name,
    HighSynchronyEvents,
    HighSynchronyEvents_PF_Data,
    Likelihood_Function,
    MODELS_AS_STR,
    Momentum,
    Momentum_Model,
    Neg_Binomial,
    Poisson,
    Ripples,
    Ripples_PF_Data,
    Run_Snippets,
    Session_Indicator,
    Session_List,
    Session_Name,
    Simulated_Session_Name,
)
from replay_structure.model_comparison import (
    Factorial_Model_Comparison,
    Gridsearch_Marginalization,
    Model_Comparison,
)
from replay_structure.model_recovery import (
    Diffusion_Model_Parameter_Prior,
    Gaussian_Model_Parameter_Prior,
    Model_Parameter_Distribution_Prior,
    Model_Recovery_Trajectory_Set,
    Model_Recovery_Trajectory_Set_Parameters,
    Momentum_Model_Parameter_Prior,
    Random_Model_Parameter_Prior,
    Stationary_Model_Parameter_Prior,
)
from replay_structure.pf_analysis import PF_Analysis
from replay_structure.read_write import (
    aggregate_momentum_gridsearch,
    load_gridsearch_results,
    load_marginalized_gridsearch_results,
    load_model_comparison_results,
    load_model_recovery_simulated_trajectory_set,
    load_pf_analysis,
    load_ratday_data,
    load_spikemat_data,
    load_structure_data,
    load_structure_model_results,
    load_trajectory_results,
    save_deviance_explained_results,
    save_descriptive_stats,
    save_diffusion_constant_results,
    save_diffusion_marginals,
    save_factorial_model_comparison_results,
    save_marginalized_gridsearch_results,
    save_marginals,
    save_model_comparison_results,
    save_model_recovery_simulated_trajectory_set,
    save_pf_analysis,
    save_predictive_analysis,
    save_spikemat_data,
    save_structure_model_results,
    save_trajectory_results,
)
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing
from replay_structure.simulated_neural_data import Simulated_Data_Preprocessing, Simulated_Spikes_Parameters
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.structure_models import Diffusion as Diffusion_Model_For_Marginals
from replay_structure.structure_models_gridsearch import Structure_Gridsearch
from replay_structure.structure_trajectory import Most_Likely_Trajectories
from replay_structure.utils import LogNorm_Distribution

SG_PARAMS = {"ripples": {"sd_meters": 0.06}, "run_snippets": {"sd_meters": 0.1}}
DIFFUSION_PARAMS = {"ripples": {"sd_meters": 0.98}, "run_snippets": {"sd_meters": 0.14}}
MOMENTUM_PARAMS = {
    "ripples": {"sd_meters": 130, "decay": 100, "sd_0_meters": 0.03},
    "run_snippets": {"sd_meters": 2.4, "decay": 20, "sd_0_meters": 0.03},
}


def _run_model_for_session(
    model, data_type: Data_Type, session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int,
    likelihood_function: Likelihood_Function, filename_ext: str
) -> None:
    print(
        f"running {model.name} model on {data_type.name} data, "
        f"with {bin_size_cm}cm bins and {time_window_ms}ms time window"
    )
    structure_data = load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )
    if isinstance(model.name, Diffusion):
        raise NotImplementedError("Diffusion gridsearch is handled separately.")
    if isinstance(model.name, Momentum):
        raise NotImplementedError("Momentum gridsearch is handled separately.")
    if str(model.name) == "stationary":
        model_results = models.Stationary(structure_data).get_model_evidences()
    elif str(model.name) == "stationary_gaussian":
        raise NotImplementedError("Stationary Gaussian gridsearch is handled separately.")
    elif str(model.name) == "random":
        model_results = models.Random(structure_data).get_model_evidences()
    else:
        raise ValueError(f"Unsupported model: {model.name}")
    save_structure_model_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        model.name,
        model_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def run_model(
    model,
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    likelihood_function: Optional[Likelihood_Function] = None,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if likelihood_function is None:
        likelihood_function = data_type.default_likelihood_function
    if session is not None:
        _run_model_for_session(model, data_type, session, bin_size_cm, time_window_ms, likelihood_function, filename_ext)
        return
    for session_indicator in data_type.session_list:
        _run_model_for_session(
            model, data_type, session_indicator, bin_size_cm, time_window_ms, likelihood_function, filename_ext
        )


def _run_gridsearch_marginalization(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    session_indicator: Session_Indicator,
    filename_ext: str,
) -> None:
    for model in MODELS:
        if model.n_params is not None:
            gridsearch_results = load_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )
            assert isinstance(gridsearch_results, Structure_Gridsearch)
            if isinstance(session_indicator, Session_Name):
                marginalized_gridsearch = Gridsearch_Marginalization(gridsearch_results)
            elif isinstance(session_indicator, Simulated_Session_Name):
                assert data_type.simulated_data_name is not None
                marginalization_info = load_marginalized_gridsearch_results(
                    Session_List[0],
                    time_window_ms,
                    data_type.simulated_data_name,
                    Poisson(),
                    model.name,
                    bin_size_cm=bin_size_cm,
                ).marginalization_info
                marginalized_gridsearch = Gridsearch_Marginalization(
                    gridsearch_results, marginalization_info=marginalization_info
                )
            else:
                marginalized_gridsearch = Gridsearch_Marginalization(gridsearch_results)
            save_marginalized_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                marginalized_gridsearch,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )


def _load_model_evidences_for_model_comparison(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    filename_ext: str,
):
    model_evidences: Dict[str, np.ndarray] = dict()
    for model in MODELS:
        if model.n_params is not None:
            model_evidences[str(model.name)] = load_marginalized_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            ).marginalized_model_evidences
        else:
            model_evidences[str(model.name)] = load_structure_model_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )
    return model_evidences


def _run_model_comparison_for_session(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    session_indicator: Session_Indicator,
    filename_ext: str,
) -> None:
    if Momentum_Model in MODELS:
        aggregate_momentum_gridsearch(
            session_indicator,
            time_window_ms,
            data_type.name,
            likelihood_function,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        )
    _run_gridsearch_marginalization(
        data_type, bin_size_cm, time_window_ms, likelihood_function, session_indicator, filename_ext
    )
    model_evidences = _load_model_evidences_for_model_comparison(
        data_type, session_indicator, bin_size_cm, time_window_ms, likelihood_function, filename_ext
    )
    if isinstance(session_indicator, Session_Name) or isinstance(session_indicator, External_Session_Name):
        random_effects_prior = 10
    elif isinstance(session_indicator, Simulated_Session_Name):
        random_effects_prior = 2
    else:
        raise Exception("Invalid session_indicator type.")
    mc_results = Model_Comparison(model_evidences, random_effects_prior=random_effects_prior)
    save_model_comparison_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        mc_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def run_model_comparison(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    likelihood_function: Optional[Likelihood_Function] = None,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if likelihood_function is None:
        likelihood_function = data_type.default_likelihood_function
    if session is not None:
        _run_model_comparison_for_session(data_type, bin_size_cm, time_window_ms, likelihood_function, session, filename_ext)
        return
    for session_indicator in data_type.session_list:
        _run_model_comparison_for_session(
            data_type, bin_size_cm, time_window_ms, likelihood_function, session_indicator, filename_ext
        )


def _load_model_evidences_for_factorial(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    filename_ext: str,
):
    model_evidences: Dict[str, np.ndarray] = dict()
    for model in MODELS:
        if model.n_params is not None:
            model_evidences[str(model.name)] = load_marginalized_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            ).marginalized_model_evidences
        else:
            model_evidences[str(model.name)] = load_structure_model_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )
    return model_evidences


def _run_factorial_model_comparison_for_session(
    data_type: Data_Type, bin_size_cm: int, time_window_ms: int, session_indicator: Session_Indicator, filename_ext: str
) -> None:
    model_evidences: Dict[str, dict] = dict()
    for likelihood in [Poisson(), Neg_Binomial()]:
        model_evidences[str(likelihood)] = _load_model_evidences_for_factorial(
            data_type, session_indicator, bin_size_cm, time_window_ms, likelihood, filename_ext
        )
    if isinstance(session_indicator, Session_Name):
        random_effects_prior = 8
    elif isinstance(session_indicator, Simulated_Session_Name):
        random_effects_prior = 3
    else:
        raise Exception("Invalid session_indicator type.")
    mc_results = Factorial_Model_Comparison(model_evidences, random_effects_prior=random_effects_prior)
    save_factorial_model_comparison_results(
        session_indicator, time_window_ms, data_type.name, mc_results, bin_size_cm=bin_size_cm, ext=filename_ext
    )


def run_factorial_model_comparison(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: int = 3,
    filename_ext: str = "",
) -> None:
    if session is not None:
        _run_factorial_model_comparison_for_session(data_type, bin_size_cm, time_window_ms, session, filename_ext)
        return
    for session_indicator in data_type.session_list:
        _run_factorial_model_comparison_for_session(data_type, bin_size_cm, time_window_ms, session_indicator, filename_ext)


def _run_deviance_explained_for_session(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    session: Session_Indicator,
    filename_ext: str,
) -> None:
    structure_data = load_structure_data(
        session, time_window_ms, data_type.name, likelihood_function, bin_size_cm=bin_size_cm, ext=filename_ext
    )
    structure_data_for_null: Optional[Structure_Analysis_Input]
    if data_type.simulated_data_name is not None:
        structure_data_for_null = load_structure_data(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            likelihood_function,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        )
    else:
        structure_data_for_null = None
    model_comparison_results = load_model_comparison_results(
        session, time_window_ms, data_type.name, likelihood_function, bin_size_cm=bin_size_cm, ext=filename_ext
    )
    deviance_explained_results = Deviance_Explained(
        structure_data, model_comparison_results, structure_data_for_null=structure_data_for_null
    )
    save_deviance_explained_results(
        session,
        time_window_ms,
        data_type.name,
        likelihood_function,
        deviance_explained_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def run_deviance_explained(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    likelihood_function: Optional[Likelihood_Function] = None,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if likelihood_function is None:
        likelihood_function = data_type.default_likelihood_function
    if session is not None:
        _run_deviance_explained_for_session(
            data_type, bin_size_cm, time_window_ms, likelihood_function, session, filename_ext
        )
        return
    for session_indicator in data_type.session_list:
        _run_deviance_explained_for_session(
            data_type, bin_size_cm, time_window_ms, likelihood_function, session_indicator, filename_ext
        )


def _get_marginals_for_spikemat(
    session: Session_Indicator,
    structure_data: Structure_Analysis_Input,
    spikemat_ind: int,
    bin_size_cm: int,
    time_window_ms: int,
    data_type: Data_Type,
    filename_ext: str,
) -> None:
    print(f"running spikemat {spikemat_ind}")
    plotting_folder = os.path.join("plots", f"{session}spikemat{spikemat_ind}")
    if not os.path.exists(plotting_folder):
        os.mkdir(plotting_folder)
    marginals = All_Models_Marginals(
        structure_data,
        spikemat_ind,
        stationary_gaussian_params=SG_PARAMS[str(data_type.name)],
        diffusion_params=DIFFUSION_PARAMS[str(data_type.name)],
        momentum_params=MOMENTUM_PARAMS[str(data_type.name)],
        plotting_folder=plotting_folder,
    )
    print(marginals.marginals["momentum"].sum())
    shutil.rmtree(plotting_folder)
    save_marginals(
        session,
        spikemat_ind,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        marginals,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def _get_diffusion_marginals_for_session(
    session: Session_Indicator,
    structure_data: Structure_Analysis_Input,
    bin_size_cm: int,
    time_window_ms: int,
    data_type: Data_Type,
    filename_ext: str,
) -> None:
    sd_meters = DIFFUSION_PARAMS[str(data_type.name)]["sd_meters"]
    print(sd_meters)
    marginals = Diffusion_Model_For_Marginals(structure_data, sd_meters).get_marginals()
    save_diffusion_marginals(
        session,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        marginals,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def run_marginals(
    data_type: Data_Type,
    session: Session_Indicator,
    spikemat_ind: Optional[int] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    diffusion_only: bool = False,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    structure_data = load_structure_data(
        session,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )
    if diffusion_only:
        _get_diffusion_marginals_for_session(session, structure_data, bin_size_cm, time_window_ms, data_type, filename_ext)
        return
    if spikemat_ind is not None:
        _get_marginals_for_spikemat(
            session, structure_data, spikemat_ind, bin_size_cm, time_window_ms, data_type, filename_ext
        )
        return
    for current_spikemat_ind in range(len(structure_data.spikemats)):
        _get_marginals_for_spikemat(
            session, structure_data, current_spikemat_ind, bin_size_cm, time_window_ms, data_type, filename_ext
        )


def _run_trajectories_for_session(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    sd_meters: Optional[float],
    filename_ext: str = "",
) -> None:
    print(
        f"running viterbi algorithm on {data_type.name} data, "
        f"with {bin_size_cm}cm bins and {time_window_ms}ms time window"
    )
    structure_data = load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )
    if sd_meters is None:
        raise AttributeError("Enter sd_meters")
    print(sd_meters)
    if structure_data is not None:
        trajectory_results = Most_Likely_Trajectories(structure_data, sd_meters)
    else:
        trajectory_results = None
    save_trajectory_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        trajectory_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def run_trajectories(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    likelihood_function: Optional[Likelihood_Function] = None,
    sd_meters: Optional[float] = None,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if likelihood_function is None:
        likelihood_function = data_type.default_likelihood_function
    if session is not None:
        _run_trajectories_for_session(
            data_type, session, bin_size_cm, time_window_ms, likelihood_function, sd_meters, filename_ext
        )
        return
    for session_indicator in data_type.session_list[2:]:
        _run_trajectories_for_session(
            data_type, session_indicator, bin_size_cm, time_window_ms, likelihood_function, sd_meters, filename_ext
        )


def _run_pf_analysis_for_session(
    session: Session_Indicator, bin_size_cm: int, time_window_ms: int, data_type: Data_Type, decoding_type: str,
    filename_ext: str
) -> None:
    print(f"running session {session} with {bin_size_cm}cm bins")
    pf_data = load_structure_data(
        session,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )
    if pf_data is None:
        print(f"no data for: {session}")
        map_results = None
    else:
        map_results = PF_Analysis(pf_data, decoding_type=decoding_type, save_only_trajectories=False)
    save_pf_analysis(
        session, time_window_ms, data_type.name, map_results, decoding_type, bin_size_cm=bin_size_cm, ext=filename_ext
    )


def run_pf_analysis(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    decoding_type: str = "map",
    filename_ext: str = "",
) -> None:
    time_window_ms = data_type.default_time_window_ms
    if session is not None:
        _run_pf_analysis_for_session(session, bin_size_cm, time_window_ms, data_type, decoding_type, filename_ext)
        return
    for session_indicator in data_type.session_list:
        _run_pf_analysis_for_session(
            session_indicator, bin_size_cm, time_window_ms, data_type, decoding_type, filename_ext
        )


def _run_diffusion_constant_for_session(
    bin_size_cm: int,
    time_window_ms: int,
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    trajectory_type: str,
    bin_space: bool,
    filename_ext: str,
) -> None:
    if trajectory_type == "inferred":
        trajectories = load_trajectory_results(
            session_indicator,
            time_window_ms,
            data_type.name,
            data_type.default_likelihood_function,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).most_likely_trajectories
    elif trajectory_type == "true":
        if isinstance(session_indicator, Session_Name):
            spikemat_data = load_spikemat_data(
                session_indicator, time_window_ms, data_type.name, bin_size_cm=bin_size_cm, ext=filename_ext
            )
            assert isinstance(spikemat_data, Run_Snippet_Preprocessing)
            trajectories = spikemat_data.run_info["true_trajectories_cm"]
        elif isinstance(session_indicator, Simulated_Session_Name):
            mc_recovery_trajectories = load_model_recovery_simulated_trajectory_set(
                data_type.name, session_indicator, ext=filename_ext
            )
            trajectories = {
                i: mc_recovery_trajectories.trajectory_set[i].trajectory_cm
                for i in range(len(mc_recovery_trajectories.trajectory_set))
            }
        else:
            raise Exception("Invalid session indicator for true trajectories")
    else:
        raise Exception("Invalid trajectory_type")
    diffusion_constant_results = Diffusion_Constant(trajectories, bin_size_cm=bin_size_cm)
    save_diffusion_constant_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        diffusion_constant_results,
        trajectory_type,
        bin_space=bin_space,
        bin_size_cm=bin_size_cm,
    )


def run_diffusion_constant_analysis(
    data_type: Data_Type,
    session: Optional[Session_Indicator] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    trajectory_type: str = "inferred",
    bin_space: bool = False,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if session is not None:
        _run_diffusion_constant_for_session(
            bin_size_cm, time_window_ms, data_type, session, trajectory_type, bin_space, filename_ext
        )
        return
    for session_indicator in data_type.session_list:
        _run_diffusion_constant_for_session(
            bin_size_cm, time_window_ms, data_type, session_indicator, trajectory_type, bin_space, filename_ext
        )


def _load_ratday_data_all_sessions(bin_size_cm: int = 4) -> Dict[Tuple[int, int], RatDay_Preprocessing]:
    ratday_data: Dict[Tuple[int, int], RatDay_Preprocessing] = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        ratday_data[session.rat, session.day] = load_ratday_data(session, bin_size_cm=bin_size_cm)
    return ratday_data


def _load_ripple_data_all_sessions(data_type: Data_Type, bin_size_cm: int = 4) -> dict:
    ripple_data = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        rd_ripple_data = load_spikemat_data(
            session, data_type.default_time_window_ms, data_type.name, bin_size_cm=bin_size_cm
        )
        ripple_data[session.rat, session.day] = rd_ripple_data
    return ripple_data


def _load_model_comparison_results_all_sessions(data_type: Data_Type, bin_size_cm: int = 4) -> Dict[Tuple[int, int], Model_Comparison]:
    model_comparison_results: Dict[Tuple[int, int], Model_Comparison] = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        model_comparison_results[session.rat, session.day] = load_model_comparison_results(
            session,
            data_type.default_time_window_ms,
            data_type.name,
            data_type.default_likelihood_function,
            bin_size_cm=bin_size_cm,
        )
    return model_comparison_results


def _load_trajectory_results_all_sessions(
    data_type: Data_Type, bin_size_cm: int = 4
) -> Dict[Tuple[int, int], Most_Likely_Trajectories]:
    trajectory_results: Dict[Tuple[int, int], Most_Likely_Trajectories] = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        trajectory_results[session.rat, session.day] = load_trajectory_results(
            session,
            data_type.default_time_window_ms,
            data_type.name,
            data_type.default_likelihood_function,
            bin_size_cm=bin_size_cm,
        )
    return trajectory_results


def run_descriptive_stats(data_type: Data_Type, bin_size_cm: int = 4) -> None:
    ratday_data = _load_ratday_data_all_sessions(bin_size_cm)
    ripple_data = _load_ripple_data_all_sessions(data_type, bin_size_cm)
    model_comparison_results = _load_model_comparison_results_all_sessions(data_type, bin_size_cm)
    trajectory_results = _load_trajectory_results_all_sessions(data_type, bin_size_cm)
    if isinstance(data_type.name, Ripples):
        descriptive_stats: pd.DataFrame = build_descriptive_stats(
            ratday_data, ripple_data, model_comparison_results, trajectory_results
        )
    elif isinstance(data_type.name, HighSynchronyEvents):
        descriptive_stats = get_descriptive_stats_hse(ratday_data, ripple_data, model_comparison_results)
    else:
        raise Exception("Invalid data type.")
    save_descriptive_stats(
        data_type.default_time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        descriptive_stats,
        bin_size_cm=bin_size_cm,
    )


def _load_spikemat_times_all_sessions(data_type: Data_Type, bin_size_cm: int = 4) -> Dict[Tuple[int, int], np.ndarray]:
    spikemat_times_s = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        rd_ripple_data = load_spikemat_data(
            session, data_type.default_time_window_ms, data_type.name, bin_size_cm=bin_size_cm
        )
        if isinstance(rd_ripple_data, Ripple_Preprocessing):
            spikemat_times_s[session.rat, session.day] = rd_ripple_data.data["ripple_times_s"]
        elif isinstance(rd_ripple_data, HighSynchronyEvents_Preprocessing):
            spikemat_times_s[session.rat, session.day] = rd_ripple_data.spikemat_info["popburst_times_s"]
        else:
            raise Exception("Invalid data_type.")
    return spikemat_times_s


def _load_predictive_trajectory_results_all_sessions(
    data_type: Data_Type, trajectory_type: str, bin_size_cm: int = 4
) -> Dict[Tuple[int, int], dict]:
    trajectory_results = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        if trajectory_type == "viterbi":
            viterbi_trajectories = load_trajectory_results(
                session,
                data_type.default_time_window_ms,
                data_type.name,
                data_type.default_likelihood_function,
                bin_size_cm=bin_size_cm,
            )
            trajectory_results[session.rat, session.day] = viterbi_trajectories.most_likely_trajectories
        elif (trajectory_type == "map") or (trajectory_type == "mean"):
            if isinstance(data_type.name, Ripples):
                pf_data_type = Ripples_PF_Data
            elif isinstance(data_type.name, HighSynchronyEvents):
                pf_data_type = HighSynchronyEvents_PF_Data
            else:
                raise Exception("Invalid data type.")
            pf_trajectories = load_pf_analysis(
                session, pf_data_type.default_time_window_ms, pf_data_type.name, decoding_type=trajectory_type
            )
            trajectory_results[session.rat, session.day] = pf_trajectories.results["trajectory_map_positions"]
        else:
            raise Exception("Invalid trajectory type")
    return trajectory_results


def _get_behavior_paths(
    ratday_data: Dict[Tuple[int, int], RatDay_Preprocessing],
    spikemat_times_s: Dict[Tuple[int, int], np.ndarray],
    distance_threshold_cm: float = 75,
) -> dict:
    past_path: dict = dict()
    future_path: dict = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        rat = session.rat
        day = session.day
        past_path[(rat, day)] = dict()
        future_path[(rat, day)] = dict()
        n_ripples = spikemat_times_s[rat, day].shape[0]
        for ripple_num in range(n_ripples):
            ripple_start = spikemat_times_s[rat, day][ripple_num, 0]
            ripple_end = spikemat_times_s[rat, day][ripple_num, 1]
            past_path[(rat, day)][ripple_num] = pred.get_behavior_path(
                ripple_start - 10,
                ripple_start,
                ratday_data[(rat, day)].data["pos_times_s"],
                ratday_data[(rat, day)].data["pos_xy_cm"],
                dist_thresh=distance_threshold_cm,
                path_type="past",
            )
            future_path[(rat, day)][ripple_num] = pred.get_behavior_path(
                ripple_end,
                ripple_end + 10,
                ratday_data[(rat, day)].data["pos_times_s"],
                ratday_data[(rat, day)].data["pos_xy_cm"],
                dist_thresh=distance_threshold_cm,
                path_type="future",
            )
    return {"past": past_path, "future": future_path}


def _get_angular_distances(
    data_type: Data_Type, behavior_paths: dict, trajectory_results: dict, radius_array: np.ndarray = np.arange(15, 75, 3)
) -> dict:
    angular_distances: dict = dict()
    if isinstance(data_type.name, Ripples):
        n_ripples_total = 2980
    elif isinstance(data_type.name, HighSynchronyEvents):
        n_ripples_total = 4469
    else:
        raise Exception("Invalid data type.")
    angular_distances["past"] = np.full((n_ripples_total, len(radius_array)), np.nan)
    angular_distances["future"] = np.full((n_ripples_total, len(radius_array)), np.nan)
    angular_distances["behavior"] = np.full((n_ripples_total, len(radius_array)), np.nan)
    angular_distances["control_past"] = np.full((n_ripples_total, len(radius_array)), np.nan)
    angular_distances["control_future"] = np.full((n_ripples_total, len(radius_array)), np.nan)
    ripple_id = 0
    for rat, day in trajectory_results:
        for ripple in trajectory_results[rat, day]:
            if trajectory_results[rat, day][ripple] is not None:
                if len(behavior_paths["future"][rat, day][ripple]) > 0:
                    ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                        behavior_paths["past"][rat, day][ripple], trajectory_results[rat, day][ripple], radius_array=radius_array
                    )
                    angular_distances["past"][ripple_id] = ripple_angular_dist
                    ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                        behavior_paths["future"][rat, day][ripple], trajectory_results[rat, day][ripple], radius_array=radius_array
                    )
                    angular_distances["future"][ripple_id] = ripple_angular_dist
                    behavior_angular_dist, _, _ = pred.get_angular_dist_array(
                        behavior_paths["past"][rat, day][ripple],
                        behavior_paths["future"][rat, day][ripple],
                        radius_array=radius_array,
                    )
                    angular_distances["behavior"][ripple_id] = behavior_angular_dist
                    random_rat = np.random.choice([1, 3])
                    random_day = np.random.randint(1, 3)
                    random_ripple = np.random.randint(len(trajectory_results[random_rat, random_day]))
                    while (trajectory_results[random_rat, random_day][random_ripple] is None) or (
                        len(trajectory_results[random_rat, random_day][random_ripple]) == 0
                    ):
                        random_ripple = np.random.randint(len(trajectory_results[random_rat, random_day]))
                    if len(behavior_paths["past"][rat, day][ripple]) > 0:
                        dist = trajectory_results[random_rat, random_day][random_ripple][0] - behavior_paths["past"][rat, day][ripple][0]
                        random_trajectory_shifted = trajectory_results[random_rat, random_day][random_ripple] - dist
                        if len(behavior_paths["future"][rat, day][ripple]) > 0:
                            ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                                behavior_paths["past"][rat, day][ripple], random_trajectory_shifted, radius_array=radius_array
                            )
                            angular_distances["control_past"][ripple_id] = ripple_angular_dist
                            ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                                behavior_paths["future"][rat, day][ripple], random_trajectory_shifted, radius_array=radius_array
                            )
                            angular_distances["control_future"][ripple_id] = ripple_angular_dist
            ripple_id += 1
    return angular_distances


def run_predictive_analysis(data_type: Data_Type, trajectory_type: str = "viterbi", bin_size_cm: int = 4) -> None:
    ratday_data = _load_ratday_data_all_sessions(bin_size_cm)
    spikemat_times_s = _load_spikemat_times_all_sessions(data_type, bin_size_cm)
    trajectory_results = _load_predictive_trajectory_results_all_sessions(data_type, trajectory_type, bin_size_cm)
    behavior_paths = _get_behavior_paths(ratday_data, spikemat_times_s)
    angular_distances = _get_angular_distances(data_type, behavior_paths, trajectory_results)
    save_predictive_analysis(
        data_type.default_time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        trajectory_type,
        (behavior_paths, angular_distances),
        bin_size_cm=bin_size_cm,
    )


def _get_duration_distribution(spikemats: dict, time_window_s: float) -> LogNorm_Distribution:
    spikemat_durations_s = np.array([spikemats[i].shape[0] for i in range(len(spikemats)) if spikemats[i] is not None]) * time_window_s
    s, loc, scale = sp.lognorm.fit(spikemat_durations_s, floc=0)
    return LogNorm_Distribution(s=s, loc=loc, scale=scale)


def _get_model_param_dist(
    session_indicator: Simulated_Session_Name, data_type: Data_Type, time_window_ms: int, bin_size_cm: int, filename_ext: str = ""
) -> Model_Parameter_Distribution_Prior:
    print(session_indicator.model.name)
    if isinstance(session_indicator.model.name, Diffusion):
        gridsearch_best_fit_distibution = load_marginalized_gridsearch_results(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            Poisson(),
            session_indicator.model.name,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).marginalization_info["fit_prior_params"]
        return Diffusion_Model_Parameter_Prior(gridsearch_best_fit_distibution["sd_meters"])
    if isinstance(session_indicator.model.name, Momentum):
        gridsearch_best_fit_distibution = load_marginalized_gridsearch_results(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            Poisson(),
            session_indicator.model.name,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).marginalization_info["fit_prior_params"]
        return Momentum_Model_Parameter_Prior(gridsearch_best_fit_distibution["2d_normal"])
    if str(session_indicator.model.name) == "stationary":
        return Stationary_Model_Parameter_Prior()
    if str(session_indicator.model.name) == "stationary_gaussian":
        gridsearch_best_fit_distibution = load_marginalized_gridsearch_results(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            Poisson(),
            session_indicator.model.name,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).marginalization_info["fit_prior_params"]
        return Gaussian_Model_Parameter_Prior(gridsearch_best_fit_distibution["sd_meters"])
    if str(session_indicator.model.name) == "random":
        return Random_Model_Parameter_Prior()
    raise Exception("Invalid model.")


def _generate_data_for_session(
    bin_size_cm: int, time_window_ms: int, data_type: Data_Type, session_indicator: Simulated_Session_Name, filename_ext: str
) -> None:
    print("generating simulated data under {} model dynamics with {}cm bins".format(session_indicator.model.name, bin_size_cm))
    assert data_type.simulated_data_name is not None
    structure_data = load_structure_data(
        Session_List[0],
        time_window_ms,
        data_type.simulated_data_name,
        data_type.default_likelihood_function,
        bin_size_cm=bin_size_cm,
    )
    duration_s_dist = _get_duration_distribution(structure_data.spikemats, structure_data.params.time_window_s)
    model_param_dist = _get_model_param_dist(session_indicator, data_type, time_window_ms, bin_size_cm, filename_ext)
    trajectory_set_params = Model_Recovery_Trajectory_Set_Parameters(model_param_dist, duration_s_dist)
    trajectory_set = Model_Recovery_Trajectory_Set(trajectory_set_params)
    print("DONE")
    save_model_recovery_simulated_trajectory_set(trajectory_set, session_indicator, data_type.name, ext=filename_ext)
    simulated_spikes_params = Simulated_Spikes_Parameters(
        structure_data.pf_matrix,
        structure_data.params.time_window_ms,
        structure_data.params.likelihood_function_params,
    )
    simulated_data = Simulated_Data_Preprocessing(trajectory_set.trajectory_set, simulated_spikes_params)
    save_spikemat_data(
        simulated_data, session_indicator, time_window_ms, data_type.name, bin_size_cm=bin_size_cm, ext=filename_ext
    )


def generate_model_recovery_data(
    data_type: Data_Type,
    model: Optional[Simulated_Session_Name] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if model is not None:
        _generate_data_for_session(bin_size_cm, time_window_ms, data_type, model, filename_ext)
        return
    for session_indicator in data_type.session_list:
        assert isinstance(session_indicator, Simulated_Session_Name)
        _generate_data_for_session(bin_size_cm, time_window_ms, data_type, session_indicator, filename_ext)
