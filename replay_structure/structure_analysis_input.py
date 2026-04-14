from replay_structure.metadata import (
    Likelihood_Function,
    Likelihood_Function_Params,
    Neg_Binomial,
    Neg_Binomial_Params,
    Poisson,
    Poisson_Params,
)
import numpy as np
from typing import Optional

from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.simulated_neural_data import Simulated_Data_Preprocessing
from replay_structure.config import (
    Structure_Analysis_Input_Parameters,
    PF_SCALING_FACTOR,
)
from replay_structure.external_event_adapters import validate_canonical_payload


class Structure_Analysis_Input:
    """Reformats preprocessed SWR, HSE, run snippet, and simulated data into a
    consistent format. Instances of this class are the input to the Structure_Models.
    """

    def __init__(
        self,
        pf_matrix: np.ndarray,
        spikemats: dict,
        likelihood_function_params: Likelihood_Function_Params,
        time_window_ms: int,
        bin_size_cm: int = 4,
        n_bins_x: int = 50,
        n_bins_y: int = 50,
        time_window_advance_ms: Optional[int] = None,
        source_metadata: Optional[dict] = None,
    ):
        self.pf_matrix = pf_matrix
        self.spikemats = spikemats
        self.source_metadata = dict(source_metadata or {})
        self.params = Structure_Analysis_Input_Parameters(
            likelihood_function_params,
            time_window_ms,
            bin_size_cm=bin_size_cm,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            time_window_advance_ms=time_window_advance_ms,
        )

    @classmethod
    def reformat_ripple_data(
        cls,
        ripple_data: Ripple_Preprocessing,
        likelihood_function: Likelihood_Function,
        select_population_burst=True,
    ):
        if isinstance(likelihood_function, Poisson):
            likelihood_function_params: Likelihood_Function_Params = Poisson_Params(
                rate_scaling=PF_SCALING_FACTOR
            )
        elif isinstance(likelihood_function, Neg_Binomial):
            likelihood_function_params = Neg_Binomial_Params(
                alpha=ripple_data.ripple_info["firing_rate_scaling"]["alpha"],
                beta=ripple_data.ripple_info["firing_rate_scaling"]["beta"],
            )
        else:
            raise Exception("Invalid likelihood function")

        if select_population_burst:
            spikemats = ripple_data.ripple_info["spikemats_popburst"]
        else:
            spikemats = ripple_data.ripple_info["spikemats_fullripple"]
        return cls(
            ripple_data.pf_matrix,
            spikemats,
            likelihood_function_params,
            ripple_data.params.time_window_ms,
            bin_size_cm=ripple_data.params.bin_size_cm,
            n_bins_x=ripple_data.params.n_bins_x,
            n_bins_y=ripple_data.params.n_bins_y,
        )

    @classmethod
    def reformat_highsynchrony_data(
        cls,
        highsynchrony_data: HighSynchronyEvents_Preprocessing,
        likelihood_function: Likelihood_Function,
        select_population_burst=True,
    ):
        if isinstance(likelihood_function, Poisson):
            likelihood_function_params: Likelihood_Function_Params = Poisson_Params(
                rate_scaling=PF_SCALING_FACTOR
            )
        elif isinstance(likelihood_function, Neg_Binomial):
            likelihood_function_params = Neg_Binomial_Params(
                alpha=highsynchrony_data.spikemat_info["firing_rate_scaling"]["alpha"],
                beta=highsynchrony_data.spikemat_info["firing_rate_scaling"]["beta"],
            )
        else:
            raise Exception("Invalid likelihood function")

        if select_population_burst:
            spikemats = highsynchrony_data.spikemat_info["spikemats_popburst"]
        else:
            spikemats = highsynchrony_data.spikemat_info["spikemats_fullripple"]
        return cls(
            highsynchrony_data.pf_matrix,
            spikemats,
            likelihood_function_params,
            highsynchrony_data.params.time_window_ms,
            bin_size_cm=highsynchrony_data.params.bin_size_cm,
            n_bins_x=highsynchrony_data.params.n_bins_x,
            n_bins_y=highsynchrony_data.params.n_bins_y,
        )

    @classmethod
    def reformat_run_snippet_data(
        cls,
        run_snippet_data: Run_Snippet_Preprocessing,
        likelihood_function: Likelihood_Function,
    ):
        assert isinstance(likelihood_function, Poisson)
        likelihood_function_params: Likelihood_Function_Params = Poisson_Params(
            rate_scaling=None
        )
        return cls(
            run_snippet_data.pf_matrix,
            run_snippet_data.run_info["spikemats"],
            likelihood_function_params,
            run_snippet_data.params.time_window_ms,
            bin_size_cm=run_snippet_data.params.bin_size_cm,
            n_bins_x=run_snippet_data.params.n_bins_x,
            n_bins_y=run_snippet_data.params.n_bins_y,
        )

    @classmethod
    def reformat_simulated_data(
        cls,
        simulated_data: Simulated_Data_Preprocessing,
        likelihood_function_params: Likelihood_Function_Params,
    ):
        return cls(
            simulated_data.params.pf_matrix,
            simulated_data.spikemats,
            likelihood_function_params,
            simulated_data.params.time_window_ms,
            bin_size_cm=simulated_data.params.bin_size_cm,
            n_bins_x=simulated_data.params.n_bins_x,
            n_bins_y=simulated_data.params.n_bins_y,
        )

    @classmethod
    def reformat_pfanalysis_data(
        cls,
        ripple_data: Ripple_Preprocessing,
        likelihood_function: Likelihood_Function,
        select_population_burst=False,
    ):
        assert isinstance(likelihood_function, Poisson)
        likelihood_function_params: Likelihood_Function_Params = Poisson_Params(
            rate_scaling=PF_SCALING_FACTOR
        )
        if select_population_burst:
            spikemats = ripple_data.ripple_info["spikemats_popburst"]
        else:
            spikemats = ripple_data.ripple_info["spikemats_fullripple"]
        return cls(
            ripple_data.pf_matrix,
            spikemats,
            likelihood_function_params,
            ripple_data.params.time_window_ms,
            bin_size_cm=ripple_data.params.bin_size_cm,
            n_bins_x=ripple_data.params.n_bins_x,
            n_bins_y=ripple_data.params.n_bins_y,
            time_window_advance_ms=ripple_data.params.time_window_advance_ms,
        )

    @classmethod
    def reformat_highsynchronypf_data(
        cls,
        highsynchrony_data: HighSynchronyEvents_Preprocessing,
        likelihood_function: Likelihood_Function,
    ):
        assert isinstance(likelihood_function, Poisson)
        likelihood_function_params: Likelihood_Function_Params = Poisson_Params(
            rate_scaling=PF_SCALING_FACTOR
        )
        spikemats = highsynchrony_data.spikemat_info["spikemats_full"]
        return cls(
            highsynchrony_data.pf_matrix,
            spikemats,
            likelihood_function_params,
            highsynchrony_data.params.time_window_ms,
            bin_size_cm=highsynchrony_data.params.bin_size_cm,
            n_bins_x=highsynchrony_data.params.n_bins_x,
            n_bins_y=highsynchrony_data.params.n_bins_y,
        )

    @classmethod
    def reformat_external_data(
        cls,
        external_payload: dict,
        likelihood_function: Optional[Likelihood_Function] = None,
        likelihood_function_params: Optional[Likelihood_Function_Params] = None,
        require_square_grid: bool = True,
    ):
        validate_canonical_payload(
            external_payload, require_square_grid=require_square_grid
        )
        if likelihood_function_params is None:
            if likelihood_function is None:
                likelihood_function = Poisson()
            if isinstance(likelihood_function, Poisson):
                rate_scaling = external_payload.get(
                    "poisson_rate_scaling", PF_SCALING_FACTOR
                )
                likelihood_function_params = Poisson_Params(
                    rate_scaling=rate_scaling
                )
            elif isinstance(likelihood_function, Neg_Binomial):
                alpha = external_payload.get("alpha")
                beta = external_payload.get("beta")
                if (alpha is None) or (beta is None):
                    raise ValueError(
                        "external negbinomial payloads must provide alpha and beta"
                    )
                likelihood_function_params = Neg_Binomial_Params(
                    alpha=alpha, beta=beta
                )
            else:
                raise Exception("Invalid likelihood function")
        source_metadata = dict(external_payload.get("source_metadata", {}))
        source_metadata["diagnostics"] = external_payload.get("diagnostics", {})
        source_metadata["source_name"] = external_payload.get("source_name")
        source_metadata["event_intervals_s"] = external_payload.get("event_intervals_s")
        return cls(
            external_payload["pf_matrix"],
            external_payload["spikemats"],
            likelihood_function_params,
            external_payload["time_window_ms"],
            bin_size_cm=external_payload["bin_size_cm"],
            n_bins_x=external_payload["n_bins_x"],
            n_bins_y=external_payload["n_bins_y"],
            time_window_advance_ms=external_payload["time_window_advance_ms"],
            source_metadata=source_metadata,
        )
