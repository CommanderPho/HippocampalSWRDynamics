import numpy as np

from replay_structure.config import RatDay_Preprocessing_Parameters
from replay_structure.ratday_preprocessing import RatDay_Preprocessing


def _build_preprocessor(n_cells: int = 2, rotate_placefields: bool = False) -> RatDay_Preprocessing:
    params = RatDay_Preprocessing_Parameters(bin_size_cm=100, place_field_gaussian_sd_cm=4, rotate_placefields=rotate_placefields)
    preprocessor = RatDay_Preprocessing.__new__(RatDay_Preprocessing)
    preprocessor.params = params
    preprocessor.data = {"n_cells": n_cells}
    return preprocessor


def _reference_find_position_during_spike(pos_xy: np.ndarray, pos_times: np.ndarray, spike_time: float) -> np.ndarray:
    abs_diff = np.abs(pos_times - spike_time)
    min_diff = np.min(abs_diff)
    nearest_pos_xy = pos_xy[abs_diff == min_diff][0]
    if nearest_pos_xy.shape != (2,):
        nearest_pos_xy = nearest_pos_xy[0]
    return nearest_pos_xy


def _reference_calc_spike_histograms(
    preprocessor: RatDay_Preprocessing,
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    pos_xy: np.ndarray,
    pos_times: np.ndarray,
    spatial_grid: dict,
) -> np.ndarray:
    spike_histograms = np.zeros(
        (preprocessor.data["n_cells"], preprocessor.params.n_bins_x, preprocessor.params.n_bins_y)
    )
    for cell_id in range(preprocessor.data["n_cells"]):
        cell_spike_times = spike_times[spike_ids == cell_id]
        cell_spike_pos_xy = np.array(
            [
                _reference_find_position_during_spike(pos_xy, pos_times, time)
                for time in cell_spike_times
            ]
        )
        if len(cell_spike_times) > 0:
            spike_hist, _, _ = np.histogram2d(
                cell_spike_pos_xy[:, 0],
                cell_spike_pos_xy[:, 1],
                bins=(spatial_grid["x"], spatial_grid["y"]),
            )
            spike_histograms[cell_id] = spike_hist.T
    return spike_histograms


def _reference_calc_place_fields(
    preprocessor: RatDay_Preprocessing,
    position_histogram: np.ndarray,
    spike_histograms: np.ndarray,
    posterior: bool = True,
) -> np.ndarray:
    place_fields = np.zeros(
        (preprocessor.data["n_cells"], preprocessor.params.n_bins_x, preprocessor.params.n_bins_y)
    )
    for cell_id in range(preprocessor.data["n_cells"]):
        place_fields[cell_id] = preprocessor.calc_one_place_field(
            position_histogram, spike_histograms[cell_id], posterior=posterior
        )
    return place_fields


def test_get_spike_positions_matches_reference_nearest_frame_selection():
    preprocessor = _build_preprocessor()
    pos_times = np.array([0.0, 1.0, 2.0, 3.0])
    pos_xy = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    spike_times = np.array([0.25, 0.5, 1.75, 3.0])

    actual = preprocessor.get_spike_positions(spike_times, pos_xy, pos_times)
    expected = np.array(
        [
            _reference_find_position_during_spike(pos_xy, pos_times, spike_time)
            for spike_time in spike_times
        ]
    )

    assert np.array_equal(actual, expected)


def test_calc_spike_histograms_and_place_fields_match_reference_loop():
    preprocessor = _build_preprocessor()
    spatial_grid = preprocessor.get_spatial_grid()
    pos_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    pos_xy = np.array(
        [
            [25.0, 25.0],
            [75.0, 75.0],
            [np.nan, np.nan],
            [125.0, 125.0],
            [175.0, 175.0],
        ]
    )
    spike_times = np.array([0.1, 0.5, 1.0, 1.49, 1.9])
    spike_ids = np.array([0, 1, 0, 1, 1])
    run_pos_xy = np.array([[20.0, 20.0], [80.0, 80.0], [120.0, 120.0], [180.0, 180.0]])

    actual_spike_histograms = preprocessor.calc_spike_histograms(
        spike_times, spike_ids, pos_xy, pos_times, spatial_grid
    )
    expected_spike_histograms = _reference_calc_spike_histograms(
        preprocessor, spike_times, spike_ids, pos_xy, pos_times, spatial_grid
    )
    assert np.array_equal(actual_spike_histograms, expected_spike_histograms)

    position_histogram = preprocessor.calc_position_histogram(run_pos_xy, spatial_grid)
    actual_place_fields = preprocessor.calc_place_fields(
        position_histogram, actual_spike_histograms, posterior=True
    )
    expected_place_fields = _reference_calc_place_fields(
        preprocessor, position_histogram, expected_spike_histograms, posterior=True
    )
    assert np.allclose(actual_place_fields, expected_place_fields)

    actual_place_fields_likelihood = preprocessor.calc_place_fields(
        position_histogram, actual_spike_histograms, posterior=False
    )
    expected_place_fields_likelihood = _reference_calc_place_fields(
        preprocessor, position_histogram, expected_spike_histograms, posterior=False
    )
    assert np.allclose(actual_place_fields_likelihood, expected_place_fields_likelihood)
