import numpy as np
import pandas as pd

from replay_structure.external_event_adapters import (
    build_canonical_payload,
    build_payload_from_pypho_decoded_epochs_result,
)
from replay_structure.metadata import Poisson
from replay_structure.model_comparison import Model_Comparison
from replay_structure.structure_analysis_input import Structure_Analysis_Input
import replay_structure.structure_models as models


class DummyDecodedEpochsResult:
    def __init__(self, filter_epochs, spkcount, decoding_time_bin_size, decoding_slideby=None):
        self.filter_epochs = filter_epochs
        self.spkcount = spkcount
        self.decoding_time_bin_size = decoding_time_bin_size
        self.decoding_slideby = decoding_slideby


def _build_square_place_fields() -> np.ndarray:
    return np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 3.0], [2.0, 1.0]],
        ]
    )


def _build_square_spikemats_cells_by_time():
    return [
        np.array([[1, 0, 1], [0, 1, 1]]),
        np.array([[0, 1, 1], [1, 0, 1]]),
    ]


def test_build_canonical_payload_normalizes_orientation_and_time_units():
    payload = build_canonical_payload(
        place_fields=_build_square_place_fields(),
        spikemats=_build_square_spikemats_cells_by_time(),
        time_window_s=0.01,
        time_window_advance_s=0.005,
        bin_size_cm=5,
        place_field_layout="cells_by_x_by_y",
        spikemat_layout="cells_by_time",
    )

    assert payload["pf_matrix"].shape == (2, 4)
    assert payload["spikemats"][0].shape == (3, 2)
    assert payload["time_window_ms"] == 10
    assert payload["time_window_advance_ms"] == 5
    assert payload["diagnostics"]["has_overlapping_time_bins"] is True


def test_pypho_mapper_uses_decoded_result_metadata():
    decoded_result = DummyDecodedEpochsResult(
        filter_epochs=pd.DataFrame({"start": [0.0, 1.0], "stop": [0.3, 1.3]}),
        spkcount=_build_square_spikemats_cells_by_time(),
        decoding_time_bin_size=0.02,
        decoding_slideby=0.01,
    )

    payload = build_payload_from_pypho_decoded_epochs_result(
        decoded_result,
        place_fields=_build_square_place_fields(),
        bin_size_cm=5,
        place_field_layout="cells_by_x_by_y",
    )

    assert payload["time_window_ms"] == 20
    assert payload["time_window_advance_ms"] == 10
    assert payload["event_intervals_s"].shape == (2, 2)


def test_external_structure_input_runs_full_model_comparison_on_square_grid():
    payload = build_canonical_payload(
        place_fields=_build_square_place_fields(),
        spikemats=_build_square_spikemats_cells_by_time(),
        time_window_ms=10,
        bin_size_cm=5,
        place_field_layout="cells_by_x_by_y",
        spikemat_layout="cells_by_time",
    )
    structure_data = Structure_Analysis_Input.reformat_external_data(
        payload, likelihood_function=Poisson()
    )

    diffusion = models.Diffusion(structure_data, sd_meters=0.2).get_model_evidences()
    momentum = np.array(
        [
            models.Momentum(
                structure_data, sd_0_meters=0.03, sd_meters=0.2, decay=10
            ).get_spikemat_model_evidence(i)
            for i in range(len(structure_data.spikemats))
        ]
    )
    stationary = models.Stationary(structure_data).get_model_evidences()
    stationary_gaussian = models.Stationary_Gaussian(
        structure_data, sd_meters=0.2
    ).get_model_evidences()
    random = models.Random(structure_data).get_model_evidences()

    assert diffusion.shape == (2,)
    assert momentum.shape == (2,)

    comparison = Model_Comparison(
        {
            "diffusion": diffusion,
            "momentum": momentum,
            "stationary": stationary,
            "stationary_gaussian": stationary_gaussian,
            "random": random,
        }
    )
    assert comparison.results_dataframe.shape[0] == 2
    assert set(comparison.results_dataframe.columns).issuperset(
        {"diffusion", "momentum", "stationary", "stationary_gaussian", "random"}
    )


def test_external_structure_input_rejects_nonsquare_grid():
    payload = build_canonical_payload(
        place_fields=np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]]),
        spikemats=[np.array([[1, 0], [0, 1], [1, 1]])],
        time_window_ms=10,
        bin_size_cm=5,
        n_bins_x=3,
        n_bins_y=1,
        place_field_layout="grid_by_cells",
        spikemat_layout="time_by_cells",
    )

    try:
        Structure_Analysis_Input.reformat_external_data(payload, likelihood_function=Poisson())
        raised = False
    except ValueError as exc:
        raised = True
        assert "square spatial grid" in str(exc)
    assert raised is True
