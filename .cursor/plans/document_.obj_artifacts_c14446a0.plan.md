---
name: Document .obj artifacts
overview: Infer every canonical `.obj` artifact type from [replay_structure/read_write.py](replay_structure/read_write.py) and the preprocessing/model classes it pickles, then append a concise reference section (Markdown heading + bullets) to the end of [notebooks/Inspect_Data.ipynb](notebooks/Inspect_Data.ipynb).
todos:
  - id: draft-reference-md
    content: Draft Markdown sections (format note, data/*.obj map, results/*.obj map) from read_write + model classes
    status: completed
  - id: append-inspect-data
    content: Append new heading + Markdown cell(s) to end of notebooks/Inspect_Data.ipynb via EditNotebook
    status: completed
isProject: false
---

# Document `.obj` file contents in Inspect_Data.ipynb

## Findings (from source)

- **On-disk format**: `.obj` artifacts under `replay_structure/data` and `replay_structure/results` are **plain Python pickles** written by [`save_data`](replay_structure/read_write.py) (`pickle.dumps`), not Wavefront `.obj` meshes. Loading uses [`load_data`](replay_structure/read_write.py) (`pickle.loads`).
- **Naming**: Paths follow helpers in [`read_write.py`](replay_structure/read_write.py); folder names match [`Data_Type`](replay_structure/metadata.py) string forms (`ripples`, `run_snippets`, `high_synchrony_events`, `placefieldID_shuffle`, `placefield_rotation`, `poisson_simulated_ripples`, `negbinomial_simulated_ripples`, `ripples_pf`, `high_synchrony_events_pf`). Published notebooks often show `data_final` / `results_final`; types are the same as this codebase’s `data/` and `results/` layout.

## Content map (what each pickle contains)

### Under `replay_structure/data`

| Location / pattern | Python type | Semantics (high level) |
|--------------------|-------------|-------------------------|
| `ratday/{rat}{day}_{bin}cm.obj` (optional `_placefields_rotated`) | [`RatDay_Preprocessing`](replay_structure/ratday_preprocessing.py) | Session-level preprocessing from MATLAB export: `params`; cleaned behavior/spikes in `data`; `velocity_info`; `place_field_data` (maps, cell IDs, firing rates). |
| `{data_type}/{session}_{bin}cm_{ms}ms.obj` | Usually [`Ripple_Preprocessing`](replay_structure/ripple_preprocessing.py), [`Run_Snippet_Preprocessing`](replay_structure/run_snippet_preprocessing.py), [`HighSynchronyEvents_Preprocessing`](replay_structure/highsynchronyevents.py), or [`Simulated_Data_Preprocessing`](replay_structure/simulated_neural_data.py) depending on `data_type` | **Ripples / PF variants / shuffles / rotation**: `params`, `pf_matrix`, `ripple_info` dict (spikemats full vs population burst, popburst times, firing-rate scaling for NegBinomial). **Run snippets**: `run_info` with matched run windows, true trajectories, spikemats. **HSE**: `highsynchronyevent_info`, `spikemat_info` (parallel structure to ripples). **Simulated**: synthetic spikes binned into `spikemats`. |
| `structure_analysis_input/{session}_{data_type}_{bin}cm_{ms}ms_{likelihood}.obj` | [`Structure_Analysis_Input`](replay_structure/structure_analysis_input.py) | Unified input to structure models: `pf_matrix`, `spikemats` (indexed dict), `params` (likelihood params, grid geometry, time bins), optional `source_metadata`. |
| `spikemat_structure_analysis_input/...` | Same | Same class when `session_indicator` is a per-spikemat name ([`SessionSpikemat_Name`](replay_structure/metadata.py)). |
| `{data_type}/{simulated_session}_simulated_trajectories.obj` | [`Model_Recovery_Trajectory_Set`](replay_structure/model_recovery.py) | List of [`Simulated_Trajectory`](replay_structure/simulated_trajectories.py) objects plus parameters for model-recovery simulations. |

### Under `replay_structure/results/{data_type}`

| Filename suffix pattern | Python type | Semantics |
|-------------------------|-------------|-----------|
| `_{model}.obj` (e.g. `_diffusion`, `_momentum`) via `save_structure_model_results` | `np.ndarray` | Per-spikemat **log model evidence** for a fixed-parameter model (not gridsearch). |
| `_{model}_gridsearch.obj`; optional `_spikemat{k}_...` for momentum | [`Structure_Gridsearch`](replay_structure/structure_models_gridsearch.py) subclass | `gridsearch_params` + `gridsearch_results` (evidence over parameter grid); momentum uses per-spikemat files when `spikemat_ind` is set. |
| `_{model}_gridsearch_marginalization.obj` | [`Gridsearch_Marginalization`](replay_structure/model_comparison.py) | Marginalized evidences + fitted priors over gridsearch parameters. |
| `_model_comparison.obj` | [`Model_Comparison`](replay_structure/model_comparison.py) | `results_dataframe` (per-model evidences + winning model column), `max_ll_counts`, `random_effects_results` (Gibbs samples, `p_models`, etc.). |
| `_factorial_model_comparison.obj` | [`Factorial_Model_Comparison`](replay_structure/model_comparison.py) | Dynamics × emission (Poisson vs NegBinomial) factorial comparison (`results_dataframe`, `random_effects_results`). |
| `_deviance_explained.obj` | [`Deviance_Explained`](replay_structure/deviance_models.py) | Per-spikemat deviance-explained per dynamics model (`results` DataFrame vs saturated/null). |
| `_{likelihood}_trajectories.obj` | [`Most_Likely_Trajectories`](replay_structure/structure_trajectory.py) | Viterbi-decoded trajectories (`most_likely_trajectories` dict), diffusion transition setup, `sd_meters`. Legacy notebook names like `_poisson_trajectories_74` are the same logical artifact with an extra filename suffix. |
| `_spikemat{k}_{likelihood}_marginals.obj` | [`All_Models_Marginals`](replay_structure/marginals.py) | Per-model latent marginals for one spikemat (`marginals` dict). |
| `_{likelihood}_diffusion_marginals.obj` | `dict` | Diffusion-specific marginal outputs (see [`save_diffusion_marginals`](replay_structure/read_write.py)). |
| `_{likelihood}_{trajectory_type}_trajectories_diffusion_constant.obj` (+ optional `_binned_`) | [`Diffusion_Constant`](replay_structure/diffusion_constant.py) | Stella-style diffusion constant from decoded paths (`diffusion_constant_info`, bootstrap distribution). |
| `{session}_{bin}cm_{ms}ms_pf_analysis_{decoding_type}.obj` | [`PF_Analysis`](replay_structure/pf_analysis.py) | Place-field–based decoding summary (`results`, `decoding_type` e.g. `map` / `mean`). |
| `predictive_analysis_{bin}cm_{ms}ms_{likelihood}_{trajectory_type}trajectories.obj` | `tuple` | `(behavior_paths, angular_distances)` from [`run_predictive_analysis`](replay_structure/pipelines/modeling_pipeline.py). |

## Notebook change (after plan approval)

- Append **one or two Markdown cells** at the end of [`notebooks/Inspect_Data.ipynb`](notebooks/Inspect_Data.ipynb) (user-requested context includes this notebook).
- Use a clear top-level heading, e.g. **`# Reference: contents of `.obj` artifact files`**, followed by:
  - One sentence on pickle serialization and pointer to [`read_write.py`](replay_structure/read_write.py).
  - Subsections **Data artifacts** and **Results artifacts** mirroring the tables above (bullets, not overly long).
  - Short cross-links to class docstrings where helpful (`RatDay_Preprocessing`, `Structure_Analysis_Input`, `Model_Comparison`, etc.).
- **No code execution required** for the documentation itself; keep it static reference material aligned with the existing discovery/summary cells above it.

## Scope note

- This documents **types and semantics** derived from the Python codebase. It does not introspect binary pickles at rest (no `.obj` files are in the repo workspace from the search).
- If you later want **auto-generated** docs from `inspect`/`typing`, that could be a follow-up code cell; the current ask is human-readable documentation at the notebook end.
