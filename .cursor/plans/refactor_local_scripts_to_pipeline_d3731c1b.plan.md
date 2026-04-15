---
name: Refactor Local Scripts To Pipeline
overview: Refactor `scripts/local` run scripts into importable, notebook-friendly APIs under a new `replay_structure/pipelines` package while preserving existing CLI behavior through thin wrapper scripts.
todos:
  - id: create-pipeline-package
    content: Create `replay_structure/pipelines` package and define public API exports for notebook imports.
    status: in_progress
  - id: extract-local-script-logic
    content: Move execution logic from `scripts/local` into pipeline modules with minimal behavioral changes.
    status: pending
  - id: convert-scripts-to-wrappers
    content: Update `scripts/local` files to thin CLI wrappers that call new pipeline functions.
    status: pending
  - id: refactor-session-orchestrator
    content: Replace subprocess orchestration in `run_all_comps_for_sess.py` with direct pipeline function calls.
    status: pending
  - id: validate-imports-and-cli
    content: Run import smoke checks and representative CLI parity checks.
    status: pending
isProject: false
---

# Refactor Local Run Scripts Into `replay_structure/pipelines`

## Goal
Move execution logic out of `scripts/local` into importable pipeline modules so top-level notebooks can call the workflow programmatically, while keeping existing command-line script behavior intact via wrappers.

## Scope Confirmed
- Include: `scripts/local` main run scripts only.
- Exclude for now: `scripts/o2` wrappers.
- Compatibility: keep existing script file paths as thin wrappers.

## Proposed Package Structure
Create a new package:
- `replay_structure/pipelines/__init__.py`
- `replay_structure/pipelines/preprocessing_pipeline.py`
- `replay_structure/pipelines/structure_analysis_pipeline.py`
- `replay_structure/pipelines/modeling_pipeline.py`
- `replay_structure/pipelines/session_pipeline.py`

Map existing scripts to pipeline modules (minimal movement, no behavior changes):
- Preprocessing scripts (`preprocess_ratday_data.py`, `preprocess_spikemat_data.py`) -> `preprocessing_pipeline.py`
- Structure input script (`reformat_data_for_structure_analysis.py`) -> `structure_analysis_pipeline.py`
- Model scripts (`run_model.py`, `run_model_comparison.py`, `run_factorial_model_comparison.py`, `run_deviance_explained.py`, `get_marginals.py`, `get_trajectories.py`, `run_pf_analysis.py`, `run_diffusion_constant.py`) -> `modeling_pipeline.py`
- Orchestration script (`run_all_comps_for_sess.py`) -> `session_pipeline.py`
- Dataset-level analytics scripts (`get_descriptive_stats.py`, `run_predictive_analysis.py`, `generate_model_recovery_data.py`) -> `modeling_pipeline.py` or a small `analysis_pipeline.py` if separation becomes clearer during extraction.

## API Design (Notebook-first)
Expose explicit callable functions (no CLI decorators) with typed parameters and defaults matching current behavior, for example:
- `run_preprocess_ratday(...)`
- `run_preprocess_spikemat(...)`
- `run_structure_analysis_reformat(...)`
- `run_model(...)`, `run_model_comparison(...)`, `run_factorial_model_comparison(...)`
- `run_session_pipeline(...)` (replacement for subprocess-based orchestration)

In `replay_structure/pipelines/__init__.py`, re-export stable public functions so notebook imports are short:
- `from replay_structure.pipelines import run_session_pipeline, run_model_comparison, ...`

## Wrapper Strategy (Preserve CLI)
For each `scripts/local/*.py` script:
- Keep file path and `click` options unchanged.
- Replace internal logic with thin call-through to corresponding pipeline function.
- Keep `if __name__ == "__main__": ...` behavior unchanged.

This preserves existing shell workflows while enabling notebook imports.

## Refactor Sequence (Low-risk to high-risk)
1. Extract preprocessing and reformat scripts first (cleanest boundaries).
2. Extract model run/comparison scripts.
3. Extract trajectory/marginal/PF/diffusion scripts.
4. Refactor `run_all_comps_for_sess.py` last: remove subprocess invocation and call pipeline APIs directly.

## Validation
- Smoke test imports from a top-level notebook context:
  - `from replay_structure.pipelines import ...`
- Run existing CLI wrappers for one representative session/data type and verify output parity.
- Ensure no import-time side effects (scripts remain entrypoints; pipeline modules import cleanly).

## Key Files to Update
- [H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/HippocampalSWRDynamics/replay_structure/__init__.py](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/HippocampalSWRDynamics/replay_structure/__init__.py)
- [H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/HippocampalSWRDynamics/replay_structure/pipelines](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/HippocampalSWRDynamics/replay_structure/pipelines)
- [H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/HippocampalSWRDynamics/scripts/local](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/HippocampalSWRDynamics/scripts/local)