import math
import time
from typing import Optional

import numpy as np

import replay_structure.utils as utils
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.config import Structure_Analysis_Input_Parameters
from replay_structure.metadata import Poisson_Params, Neg_Binomial_Params
from replay_structure.viterbi import Viterbi


class Most_Likely_Trajectories:
    """Finds the most likely trajectory given a sequence of neural activity using the
    Viterbi algorithm."""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        sd_meters: float,
        run_all: bool = True,
        verbose: bool = True,
        progress_every: Optional[int] = None,
    ):
        self.params: Structure_Analysis_Input_Parameters = structure_data.params
        self.sd_meters = sd_meters
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.verbose = verbose
        self.progress_every = progress_every
        self._transition_matrix: Optional[np.ndarray] = None
        if isinstance(self.params.likelihood_function_params, Poisson_Params):
            if self.params.likelihood_function_params.rate_scaling is not None:
                self.emission_prob_time_window = (
                    self.params.time_window_s
                    * self.params.likelihood_function_params.rate_scaling
                )
            else:
                self.emission_prob_time_window = self.params.time_window_s
        self.viterbi_input = self._initialize_viterbi_input()
        if run_all:
            if self.verbose:
                print("Getting most likely trajectories", flush=True)
            self.most_likely_trajectories = self.run_all(structure_data)


    def _initialize_viterbi_input(self):
        viterbi_input = dict()
        viterbi_input["initial_state_prior"] = (
            np.ones(self.params.n_grid) / self.params.n_grid
        )
        return viterbi_input


    def _get_transition_matrix(self) -> np.ndarray:
        if self._transition_matrix is None:
            self._transition_matrix = self._calc_transition_matrix(self.sd_bins)
        return self._transition_matrix


    def run_all(self, structure_data: Structure_Analysis_Input):
        most_likely_trajectories = dict()
        n = len(structure_data.spikemats)
        n_nonempty = sum(
            1 for k in range(n) if structure_data.spikemats[k] is not None
        )
        if self.verbose:
            print(
                f"Most_Likely_Trajectories: {n} spikemats ({n_nonempty} non-empty)",
                flush=True,
            )
        interval = self.progress_every
        if interval is None:
            interval = max(1, n // 50) if n > 0 else 1
        t0 = time.perf_counter()
        for i in range(n):
            if self.verbose and (i == 0 or i == n - 1 or (i + 1) % interval == 0):
                elapsed = time.perf_counter() - t0
                pct = 100.0 * (i + 1) / n if n else 100.0
                rate = (i + 1) / elapsed if elapsed > 0 else float("nan")
                remaining = n - (i + 1)
                eta = remaining / rate if math.isfinite(rate) and rate > 0 else float("nan")
                eta_s = f"{eta:.1f}s" if math.isfinite(eta) else "n/a"
                rate_s = f"{rate:.1f}" if math.isfinite(rate) else "n/a"
                print(
                    f"  progress {i + 1}/{n} ({pct:.1f}%) | elapsed {elapsed:.1f}s | "
                    f"ETA {eta_s} | {rate_s} spikemats/s",
                    flush=True,
                )
            most_likely_trajectories[i] = self.get_most_likely_trajectory(
                structure_data, i
            )
        if self.verbose and n > 0:
            total = time.perf_counter() - t0
            print(
                f"Most_Likely_Trajectories: finished {n} in {total:.1f}s "
                f"({n / total:.1f} spikemats/s)",
                flush=True,
            )
        return most_likely_trajectories


    def get_most_likely_trajectory(
        self, structure_data: Structure_Analysis_Input, spikemat_ind: int
    ) -> Optional[np.ndarray]:
        if structure_data.spikemats[spikemat_ind] is not None:
            most_likely_trajectory_flattened = self._get_most_likely_trajectory(
                structure_data, spikemat_ind
            )
            nx = self.params.n_bins_x
            b = self.params.bin_size_cm
            x_cm = (most_likely_trajectory_flattened % nx) * b
            y_cm = (most_likely_trajectory_flattened // nx) * b
            most_likely_trajectory = np.column_stack((x_cm, y_cm))
        else:
            most_likely_trajectory = None
        return most_likely_trajectory


    def _get_most_likely_trajectory(
        self, structure_data: Structure_Analysis_Input, spikemat_ind: int
    ) -> np.ndarray:
        if structure_data.spikemats[spikemat_ind] is not None:
            self.viterbi_input[
                "emission_probabilities"
            ] = self._calc_emission_probabilities(structure_data, spikemat_ind)
            self.viterbi_input["transition_matrix"] = self._get_transition_matrix()
            viterbi_outputs = Viterbi(self.viterbi_input).run_viterbi_algorithm()
            most_likely_trajectory = viterbi_outputs["z_max"]
        else:
            most_likely_trajectory = np.array([np.nan, np.nan])
        return most_likely_trajectory


    def _calc_transition_matrix(self, sd_bins: float) -> np.ndarray:
        """(NxN)x(NxN) matrix"""
        nx = self.params.n_bins_x
        ny = self.params.n_bins_y
        K = nx * ny
        transition_mat = np.zeros((K, K))
        m = np.arange(nx)
        n = np.arange(ny)
        mm, nn = np.meshgrid(m, n)  # t x,y
        I = np.arange(nx)[:, None, None, None]
        J = np.arange(ny)[None, :, None, None]
        NN = nn[None, None, :, :]
        MM = mm[None, None, :, :]
        dist2 = (NN - I) ** 2 + (MM - J) ** 2
        kernel = np.exp(
            -dist2 / (2 * sd_bins ** 2 * self.params.time_window_s)
        )
        I_rep = np.repeat(np.arange(nx), ny)
        J_rep = np.tile(np.arange(ny), nx)
        block = kernel[I_rep, J_rep]
        sums = block.sum(axis=(1, 2), keepdims=True)
        block_norm = block / sums
        block_flat = block_norm.reshape(I_rep.shape[0], K)
        for idx in range(I_rep.shape[0]):
            col = int(I_rep[idx]) * nx + int(J_rep[idx])
            transition_mat[:, col] = block_flat[idx]
        return transition_mat


    def _calc_emission_probabilities(
        self, structure_data: Structure_Analysis_Input, spikemat_ind: int
    ):
        if isinstance(self.params.likelihood_function_params, Neg_Binomial_Params):
            emission_probabilities = utils.calc_neg_binomial_emission_probabilities(
                structure_data.spikemats[spikemat_ind],
                structure_data.pf_matrix,
                self.params.time_window_s,
                self.params.likelihood_function_params.alpha,
                self.params.likelihood_function_params.beta,
            )
        elif isinstance(self.params.likelihood_function_params, Poisson_Params):
            emission_probabilities = utils.calc_poisson_emission_probabilities(
                structure_data.spikemats[spikemat_ind],
                structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities
