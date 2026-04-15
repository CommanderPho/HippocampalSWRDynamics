import numpy as np

import replay_structure.utils as utils


def _reference_get_spikemat(
    spike_ids: np.ndarray,
    spike_times: np.ndarray,
    place_cell_ids: np.ndarray,
    start_time: float,
    end_time: float,
    time_window_s: float,
    time_window_advance_s: float,
) -> np.ndarray:
    spikemat = np.empty(shape=(0, len(place_cell_ids)), dtype=int)
    timebin_start_time = start_time
    timebin_end_time = start_time + time_window_s
    while timebin_end_time < end_time:
        spikes_after_start = spike_times >= timebin_start_time
        spikes_before_end = spike_times < timebin_end_time
        timebin_bool = spikes_after_start == spikes_before_end
        spike_ids_in_window = spike_ids[timebin_bool]
        spikevector = np.array([[sum(spike_ids_in_window == cell_id) for cell_id in place_cell_ids]])
        spikemat = np.append(spikemat, spikevector, axis=0)
        timebin_start_time = timebin_start_time + time_window_advance_s
        timebin_end_time = timebin_end_time + time_window_advance_s
    return np.array(spikemat).astype(int)


def test_get_spikemat_empty_interval_matches_reference():
    spike_ids = np.array([0, 1, 0], dtype=int)
    spike_times = np.array([0.05, 0.10, 0.15], dtype=float)
    place_cell_ids = np.array([0, 1], dtype=int)

    actual = utils.get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.2, 0.2, 0.1)
    expected = _reference_get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.2, 0.2, 0.1)

    assert np.array_equal(actual, expected)
    assert actual.shape == (0, 2)
    assert np.issubdtype(actual.dtype, np.integer)


def test_get_spikemat_single_bin_and_boundary_behavior_matches_reference():
    spike_ids = np.array([0, 1, 1, 0], dtype=int)
    spike_times = np.array([0.0, 0.099, 0.1, 0.199], dtype=float)
    place_cell_ids = np.array([0, 1], dtype=int)

    actual = utils.get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.21, 0.1, 0.2)
    expected = _reference_get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.21, 0.1, 0.2)

    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, np.array([[1, 1]]))


def test_get_spikemat_overlapping_windows_match_reference():
    spike_ids = np.array([0, 1, 0, 1, 0], dtype=int)
    spike_times = np.array([0.05, 0.12, 0.18, 0.24, 0.31], dtype=float)
    place_cell_ids = np.array([0, 1], dtype=int)

    actual = utils.get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.5, 0.2, 0.1)
    expected = _reference_get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.5, 0.2, 0.1)

    assert np.array_equal(actual, expected)


def test_get_spikemat_ignores_non_place_cells_and_unsorted_spike_times():
    spike_ids = np.array([5, 1, 3, 5, 1, 3], dtype=int)
    spike_times = np.array([0.41, 0.09, 0.24, 0.02, 0.31, 0.19], dtype=float)
    place_cell_ids = np.array([1, 3], dtype=int)

    actual = utils.get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.5, 0.2, 0.1)
    expected = _reference_get_spikemat(spike_ids, spike_times, place_cell_ids, 0.0, 0.5, 0.2, 0.1)

    assert np.array_equal(actual, expected)
