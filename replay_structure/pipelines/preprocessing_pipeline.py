import os
from typing import Optional, Union

import scipy.io as spio

from replay_structure.config import (
    HighSynchronyEvents_Preprocessing_Parameters,
    RatDay_Preprocessing_Parameters,
    Ripple_Preprocessing_Parameters,
    Run_Snippet_Preprocessing_Parameters,
)
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.metadata import (
    DATA_PATH,
    Data_Type,
    HighSynchronyEvents,
    HighSynchronyEvents_PF,
    PlaceField_Rotation,
    PlaceFieldID_Shuffle,
    Ripples,
    Ripples_PF,
    Run_Snippets,
    Session_List,
    Session_Name,
)
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.read_write import (
    load_ratday_data,
    load_spikemat_data,
    save_ratday_data,
    save_spikemat_data,
)
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing


def load_matlab_struct(file_path: str):
    matlab_struct = spio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return matlab_struct


def get_session_data(matlab_struct, session_indicator: Session_Name) -> dict:
    matlab_struct_dict = {
        (1, 1): matlab_struct["Data"].Rat1.Day1,
        (1, 2): matlab_struct["Data"].Rat1.Day2,
        (2, 1): matlab_struct["Data"].Rat2.Day1,
        (2, 2): matlab_struct["Data"].Rat2.Day2,
        (3, 1): matlab_struct["Data"].Rat3.Day1,
        (3, 2): matlab_struct["Data"].Rat3.Day2,
        (4, 1): matlab_struct["Data"].Rat4.Day1,
        (4, 2): matlab_struct["Data"].Rat4.Day2,
    }
    return matlab_struct_dict[(session_indicator.rat, session_indicator.day)]


def _run_ratday_preprocessing_for_session(
    matlab_struct, session_indicator: Session_Name, bin_size_cm: int, rotate_placefields: bool, filename_ext: str
) -> None:
    print(f"Running session {session_indicator} with {bin_size_cm}cm bins")
    session_data = get_session_data(matlab_struct, session_indicator)
    params = RatDay_Preprocessing_Parameters(bin_size_cm=bin_size_cm, rotate_placefields=rotate_placefields)
    ratday = RatDay_Preprocessing(session_data, params)
    save_ratday_data(ratday, session_indicator, bin_size_cm, placefields_rotated=rotate_placefields, ext=filename_ext)


def run_preprocess_ratday(
    session: Optional[Session_Name] = None, bin_size_cm: int = 4, filename_ext: str = "", rotate_placefields: bool = False
) -> None:
    print("loading data")
    file_path = os.path.join(DATA_PATH, "OpenFieldData.mat")
    matlab_struct = load_matlab_struct(file_path)
    if session is not None:
        _run_ratday_preprocessing_for_session(matlab_struct, session, bin_size_cm, rotate_placefields, filename_ext)
        return
    for session_indicator in Session_List:
        assert isinstance(session_indicator, Session_Name)
        _run_ratday_preprocessing_for_session(matlab_struct, session_indicator, bin_size_cm, rotate_placefields, filename_ext)


def _run_spikemat_preprocessing_for_session(
    data_type: Data_Type, session_indicator: Session_Name, bin_size_cm: int, time_window_ms: int, filename_ext: str
) -> None:
    print(f"running session {session_indicator} with {bin_size_cm}cm binsand {time_window_ms}ms time window.")
    ratday_data = load_ratday_data(session_indicator, bin_size_cm)
    spikemat_data: Union[Ripple_Preprocessing, Run_Snippet_Preprocessing, HighSynchronyEvents_Preprocessing]
    if isinstance(data_type.name, Ripples):
        ripple_params = Ripple_Preprocessing_Parameters(ratday_data.params, time_window_ms=time_window_ms)
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params)
    elif isinstance(data_type.name, Run_Snippets):
        run_snippet_params = Run_Snippet_Preprocessing_Parameters(ratday_data.params, time_window_ms=time_window_ms)
        ripple_data = load_spikemat_data(session_indicator, 3, Ripples(), bin_size_cm=bin_size_cm, ext="")
        assert isinstance(ripple_data, Ripple_Preprocessing)
        spikemat_data = Run_Snippet_Preprocessing(ratday_data, ripple_data, run_snippet_params)
    elif isinstance(data_type.name, Ripples_PF):
        ripple_data = load_spikemat_data(session_indicator, 3, Ripples(), bin_size_cm=bin_size_cm, ext=filename_ext)
        assert isinstance(ripple_data, Ripple_Preprocessing)
        popburst_times_s = ripple_data.ripple_info["popburst_times_s"]
        ripple_params = Ripple_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms, time_window_advance_ms=5
        )
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params, popburst_times_s=popburst_times_s)
    elif isinstance(data_type.name, HighSynchronyEvents):
        params = HighSynchronyEvents_Preprocessing_Parameters(ratday_data.params, time_window_ms=time_window_ms)
        spikemat_data = HighSynchronyEvents_Preprocessing(ratday_data, params)
    elif isinstance(data_type.name, HighSynchronyEvents_PF):
        params = HighSynchronyEvents_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms, time_window_advance_ms=5
        )
        spikemat_data = HighSynchronyEvents_Preprocessing(ratday_data, params)
    elif isinstance(data_type.name, PlaceFieldID_Shuffle):
        ripple_params = Ripple_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms, shuffle_placefieldIDs=True
        )
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params)
    elif isinstance(data_type.name, PlaceField_Rotation):
        ratday_data = load_ratday_data(session_indicator, bin_size_cm, placefields_rotated=True, ext="")
        ripple_params = Ripple_Preprocessing_Parameters(ratday_data.params, time_window_ms=time_window_ms)
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params)
    else:
        raise AttributeError("Invalid data_type.")
    save_spikemat_data(spikemat_data, session_indicator, time_window_ms, data_type.name, bin_size_cm, ext=filename_ext)


def run_preprocess_spikemat(
    data_type: Data_Type,
    session: Optional[Session_Name] = None,
    bin_size_cm: int = 4,
    time_window_ms: Optional[int] = None,
    filename_ext: str = "",
) -> None:
    if time_window_ms is None:
        time_window_ms = data_type.default_time_window_ms
    if session is not None:
        _run_spikemat_preprocessing_for_session(data_type, session, bin_size_cm, time_window_ms, filename_ext)
        return
    for session_indicator in data_type.session_list:
        assert isinstance(session_indicator, Session_Name)
        _run_spikemat_preprocessing_for_session(data_type, session_indicator, bin_size_cm, time_window_ms, filename_ext)
