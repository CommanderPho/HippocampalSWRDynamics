from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Union

from replay_structure.metadata import (
    DATA_PATH,
    Diffusion,
    Momentum,
    RESULTS_PATH,
    SESSION_RATDAY,
    Session_Indicator,
    Session_Name,
    Stationary_Gaussian,
    string_to_data_type,
    string_to_likelihood_function,
    string_to_model,
    string_to_session_indicator,
)
from replay_structure.pipelines.modeling_pipeline import (
    run_deviance_explained,
    run_diffusion_constant_analysis,
    run_marginals,
    run_model,
    run_model_comparison,
    run_trajectories,
)
from replay_structure.pipelines.preprocessing_pipeline import run_preprocess_ratday, run_preprocess_spikemat
from replay_structure.pipelines.structure_analysis_pipeline import run_structure_analysis_reformat

try:
    from scripts.o2.o2_lib import (
        submit_diffusion_gridsearch,
        submit_momentum_gridsearch,
        submit_stationary_gaussian_gridsearch,
    )
except ModuleNotFoundError as err:
        submit_diffusion_gridsearch = None
        submit_momentum_gridsearch = None
        submit_stationary_gaussian_gridsearch = None


RIPPLES_DATA_TYPE = string_to_data_type("ripples")
DEFAULT_LIKELIHOOD = str(RIPPLES_DATA_TYPE.default_likelihood_function)
DEFAULT_TIME_WINDOW_MS = RIPPLES_DATA_TYPE.default_time_window_ms
DEFAULT_TRAJECTORY_SD_METERS = 0.98


@dataclass
class StepResult:
    name: str
    critical: bool
    status: str
    detail: str = ""


def _ratday_obj_path(session: int, bin_size_cm: int, filename_ext: str, rotate_placefields: bool = False) -> Path:
    session_meta = SESSION_RATDAY[session]
    session_indicator = Session_Name(rat=session_meta["rat"], day=session_meta["day"])
    if rotate_placefields:
        return DATA_PATH.joinpath("ratday", f"{session_indicator}_{bin_size_cm}cm_placefields_rotated{filename_ext}.obj")
    return DATA_PATH.joinpath("ratday", f"{session_indicator}_{bin_size_cm}cm{filename_ext}.obj")


def _spikemat_ripple_artifact_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_spikemat_data paths."""
    return DATA_PATH.joinpath(str(RIPPLES_DATA_TYPE.name), f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms{filename_ext}.obj")


def _structure_analysis_input_ripple_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_structure_data (structure_analysis_input folder)."""
    return DATA_PATH.joinpath(
        "structure_analysis_input",
        f"{session_indicator}_{RIPPLES_DATA_TYPE.name}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}{filename_ext}.obj",
    )


def _structure_model_results_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, model_name: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_structure_model_results paths."""
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}_{model_name}{filename_ext}.obj",
    )


def _model_comparison_artifact_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_model_comparison_results paths."""
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}_model_comparison{filename_ext}.obj",
    )


def _deviance_explained_artifact_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_deviance_explained_results paths."""
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}_deviance_explained{filename_ext}.obj",
    )


def _trajectories_artifact_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_trajectory_results paths."""
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}_trajectories{filename_ext}.obj",
    )


def _marginals_artifact_path(session_indicator: Session_Indicator, spikemat_ind: int, bin_size_cm: int, time_window_ms: int, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_marginals; run_marginals uses data_type.default_likelihood_function."""
    lf = RIPPLES_DATA_TYPE.default_likelihood_function
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_spikemat{spikemat_ind}_{bin_size_cm}cm_{time_window_ms}ms_{lf}_marginals{filename_ext}.obj",
    )


def _diffusion_constant_inferred_artifact_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_diffusion_constant_results (bin_space=False, trajectory_type=inferred)."""
    lf = RIPPLES_DATA_TYPE.default_likelihood_function
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_{lf}_inferred_trajectories_diffusion_constant{filename_ext}.obj",
    )


def _gridsearch_session_level_artifact_path(session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, model_name: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_gridsearch_results when spikemat_ind is None."""
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}_{model_name}_gridsearch{filename_ext}.obj",
    )


def _gridsearch_momentum_spikemat_artifact_path(session_indicator: Session_Indicator, spikemat_ind: int, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, filename_ext: str) -> Path:
    """Must match replay_structure.read_write.save_gridsearch_results when spikemat_ind is set."""
    return RESULTS_PATH.joinpath(
        str(RIPPLES_DATA_TYPE.name),
        f"{session_indicator}_spikemat{spikemat_ind}_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function_}_{Momentum()}_gridsearch{filename_ext}.obj",
    )


def _ripple_gridsearch_artifacts_complete(session: int, session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, likelihood_function_: object, filename_ext: str) -> bool:
    for model in (Diffusion(), Stationary_Gaussian()):
        p = _gridsearch_session_level_artifact_path(session_indicator, bin_size_cm, time_window_ms, likelihood_function_, model, filename_ext)
        if not p.is_file():
            return False
    n_spikemats = SESSION_RATDAY[session]["n_SWRs"]
    for spikemat_ind in range(n_spikemats):
        p = _gridsearch_momentum_spikemat_artifact_path(session_indicator, spikemat_ind, bin_size_cm, time_window_ms, likelihood_function_, filename_ext)
        if not p.is_file():
            return False
    return True


def _marginals_artifacts_complete(session: int, session_indicator: Session_Indicator, bin_size_cm: int, time_window_ms: int, filename_ext: str) -> bool:
    n_spikemats = SESSION_RATDAY[session]["n_SWRs"]
    for spikemat_ind in range(n_spikemats):
        if not _marginals_artifact_path(session_indicator, spikemat_ind, bin_size_cm, time_window_ms, filename_ext).is_file():
            return False
    return True


def _record_skip(results: Dict[str, StepResult], name: str, critical: bool, reason: str) -> None:
    results[name] = StepResult(name=name, critical=critical, status="skipped", detail=reason)
    print(f"[SKIP] {name}: {reason}")


def _run_step(
    results: Dict[str, StepResult], name: str, critical: bool, dependencies: List[str], strict: bool, action: Callable[[], None]
) -> None:
    unmet_dependencies = [dep for dep in dependencies if results.get(dep) is None or results[dep].status != "success"]
    if unmet_dependencies:
        _record_skip(results, name, critical, f"dependencies not satisfied: {', '.join(unmet_dependencies)}")
        return
    print(f"\n=== {name} ===")
    try:
        action()
    except Exception as exc:
        traceback.print_exc()
        results[name] = StepResult(name=name, critical=critical, status="failed", detail=str(exc))
        print(f"[FAIL] {name}: {exc}")
        if strict:
            raise
    else:
        results[name] = StepResult(name=name, critical=critical, status="success")
        print(f"[OK] {name}")


def _execute_or_print(dry_run: bool, description: str, action: Callable[[], None]) -> None:
    print(description)
    if dry_run:
        return
    action()


def _run_gridsearch_phase(
    session: int,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: str,
    filename_ext: str,
    dry_run: bool,
    continue_on_gridsearch_error: bool,
) -> None:
    session_indicator = string_to_session_indicator(session)
    likelihood_function_ = string_to_likelihood_function(likelihood_function)
    if dry_run:
        print(
            "Would run in-process gridsearches: diffusion, stationary_gaussian, "
            f"momentum for {SESSION_RATDAY[session]['n_SWRs']} spikemats"
        )
        return
    submit_diffusion_gridsearch(
        session_indicator,
        time_window_ms,
        RIPPLES_DATA_TYPE,
        likelihood_function_,
        bin_size_cm=bin_size_cm,
        o2=False,
        filename_ext=filename_ext,
    )
    submit_stationary_gaussian_gridsearch(
        session_indicator,
        time_window_ms,
        RIPPLES_DATA_TYPE,
        likelihood_function_,
        bin_size_cm=bin_size_cm,
        o2=False,
        filename_ext=filename_ext,
    )
    n_spikemats = SESSION_RATDAY[session]["n_SWRs"]
    n_successes = 0
    momentum_errors: List[str] = []
    for spikemat_ind in range(n_spikemats):
        try:
            submit_momentum_gridsearch(
                session_indicator,
                spikemat_ind,
                time_window_ms,
                RIPPLES_DATA_TYPE,
                likelihood_function_,
                bin_size_cm=bin_size_cm,
                o2=False,
                filename_ext=filename_ext,
            )
            n_successes += 1
        except Exception as exc:
            error_message = f"spikemat {spikemat_ind}: {exc}"
            momentum_errors.append(error_message)
            print(f"[WARN] momentum gridsearch failed for {error_message}")
            if not continue_on_gridsearch_error:
                raise RuntimeError(error_message) from exc
    if n_successes == 0:
        raise RuntimeError("momentum gridsearch failed for all spikemats")
    if momentum_errors:
        print(f"Momentum gridsearch completed with {len(momentum_errors)} failures and {n_successes} successes.")


def _print_summary(results: Dict[str, StepResult]) -> None:
    print("\n=== Summary ===")
    for result in results.values():
        detail = f" ({result.detail})" if result.detail else ""
        critical_label = "critical" if result.critical else "optional"
        print(f"{result.status.upper():8s} [{critical_label}] {result.name}{detail}")


def run_session_pipeline(
    session: int,
    bin_size_cm: int = 4,
    time_window_ms: int = DEFAULT_TIME_WINDOW_MS,
    likelihood_function: str = DEFAULT_LIKELIHOOD,
    filename_ext: str = "",
    trajectory_sd_meters: float = DEFAULT_TRAJECTORY_SD_METERS,
    skip_gridsearch: bool = False,
    force_recompute: bool = False,
    run_marginals_phase: bool = False,
    run_diffusion_constant_phase: bool = False,
    continue_on_gridsearch_error: bool = False,
    strict: bool = False,
    dry_run: bool = False,
) -> Dict[str, StepResult]:

    results: Dict[str, StepResult] = {}
    session_indicator = string_to_session_indicator(session)
    likelihood_function_ = string_to_likelihood_function(likelihood_function)
    ratday_obj_path = _ratday_obj_path(session, bin_size_cm, filename_ext)

    if ratday_obj_path.exists() and (not force_recompute):
        results["preprocess_ratday_data"] = StepResult(
            name="preprocess_ratday_data",
            critical=True,
            status="success",
            detail=f"using existing ratday file: {ratday_obj_path}",
        )
        print(f"[OK] preprocess_ratday_data: using existing ratday file: {ratday_obj_path}")
    else:
        _run_step(
            results,
            "preprocess_ratday_data",
            critical=True,
            dependencies=[],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                f"run_preprocess_ratday(session={session}, bin_size_cm={bin_size_cm}, filename_ext={filename_ext!r})",
                lambda: run_preprocess_ratday(session=session_indicator, bin_size_cm=bin_size_cm, filename_ext=filename_ext),
            ),
        )

    spikemat_path = _spikemat_ripple_artifact_path(session_indicator, bin_size_cm, time_window_ms, filename_ext)
    if spikemat_path.exists() and (not force_recompute):
        results["preprocess_spikemat_data_ripples"] = StepResult(
            name="preprocess_spikemat_data_ripples",
            critical=True,
            status="success",
            detail=f"using existing file: {spikemat_path}",
        )
        print(f"[OK] preprocess_spikemat_data_ripples: using existing file: {spikemat_path}")
    else:
        _run_step(
            results,
            "preprocess_spikemat_data_ripples",
            critical=True,
            dependencies=["preprocess_ratday_data"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                f"run_preprocess_spikemat(data_type=ripples, session={session}, bin_size_cm={bin_size_cm}, time_window_ms={time_window_ms}, filename_ext={filename_ext!r})",
                lambda: run_preprocess_spikemat(
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    filename_ext=filename_ext,
                ),
            ),
        )

    reformat_path = _structure_analysis_input_ripple_path(session_indicator, bin_size_cm, time_window_ms, likelihood_function_, filename_ext)
    if reformat_path.exists() and (not force_recompute):
        results["reformat_data_for_structure_analysis_ripples"] = StepResult(
            name="reformat_data_for_structure_analysis_ripples",
            critical=True,
            status="success",
            detail=f"using existing file: {reformat_path}",
        )
        print(f"[OK] reformat_data_for_structure_analysis_ripples: using existing file: {reformat_path}")
    else:
        _run_step(
            results,
            "reformat_data_for_structure_analysis_ripples",
            critical=True,
            dependencies=["preprocess_spikemat_data_ripples"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                f"run_structure_analysis_reformat(data_type=ripples, session={session}, bin_size_cm={bin_size_cm}, time_window_ms={time_window_ms}, likelihood_function={likelihood_function!r}, filename_ext={filename_ext!r})",
                lambda: run_structure_analysis_reformat(
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    likelihood_function=likelihood_function_,
                    filename_ext=filename_ext,
                ),
            ),
        )

    stationary_path = _structure_model_results_path(
        session_indicator, bin_size_cm, time_window_ms, likelihood_function_, string_to_model("stationary").name, filename_ext
    )
    if stationary_path.exists() and (not force_recompute):
        results["run_model_stationary"] = StepResult(
            name="run_model_stationary",
            critical=True,
            status="success",
            detail=f"using existing file: {stationary_path}",
        )
        print(f"[OK] run_model_stationary: using existing file: {stationary_path}")
    else:
        _run_step(
            results,
            "run_model_stationary",
            critical=True,
            dependencies=["reformat_data_for_structure_analysis_ripples"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                "run_model(model=stationary, data_type=ripples, ...)",
                lambda: run_model(
                    model=string_to_model("stationary"),
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    likelihood_function=likelihood_function_,
                    filename_ext=filename_ext,
                ),
            ),
        )

    random_path = _structure_model_results_path(
        session_indicator, bin_size_cm, time_window_ms, likelihood_function_, string_to_model("random").name, filename_ext
    )
    if random_path.exists() and (not force_recompute):
        results["run_model_random"] = StepResult(
            name="run_model_random",
            critical=True,
            status="success",
            detail=f"using existing file: {random_path}",
        )
        print(f"[OK] run_model_random: using existing file: {random_path}")
    else:
        _run_step(
            results,
            "run_model_random",
            critical=True,
            dependencies=["reformat_data_for_structure_analysis_ripples"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                "run_model(model=random, data_type=ripples, ...)",
                lambda: run_model(
                    model=string_to_model("random"),
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    likelihood_function=likelihood_function_,
                    filename_ext=filename_ext,
                ),
            ),
        )

    gridsearch_use_cache = (not skip_gridsearch) and _ripple_gridsearch_artifacts_complete(
        session, session_indicator, bin_size_cm, time_window_ms, likelihood_function_, filename_ext
    )
    if gridsearch_use_cache and (not force_recompute):
        results["gridsearch_ripples"] = StepResult(
            name="gridsearch_ripples",
            critical=True,
            status="success",
            detail="using existing gridsearch artifacts",
        )
        print("[OK] gridsearch_ripples: using existing gridsearch artifacts")
    else:
        _run_step(
            results,
            "gridsearch_ripples",
            critical=True,
            dependencies=["reformat_data_for_structure_analysis_ripples"],
            strict=strict,
            action=lambda: print("Skipping gridsearch execution and assuming existing results.")
            if skip_gridsearch
            else _run_gridsearch_phase(
                session,
                bin_size_cm,
                time_window_ms,
                likelihood_function,
                filename_ext,
                dry_run,
                continue_on_gridsearch_error,
            ),
        )

    mc_path = _model_comparison_artifact_path(session_indicator, bin_size_cm, time_window_ms, likelihood_function_, filename_ext)
    if mc_path.exists() and (not force_recompute):
        results["run_model_comparison_ripples"] = StepResult(
            name="run_model_comparison_ripples",
            critical=True,
            status="success",
            detail=f"using existing file: {mc_path}",
        )
        print(f"[OK] run_model_comparison_ripples: using existing file: {mc_path}")
    else:
        _run_step(
            results,
            "run_model_comparison_ripples",
            critical=True,
            dependencies=["run_model_stationary", "run_model_random", "gridsearch_ripples"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                "run_model_comparison(data_type=ripples, ...)",
                lambda: run_model_comparison(
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    likelihood_function=likelihood_function_,
                    filename_ext=filename_ext,
                ),
            ),
        )

    deviance_path = _deviance_explained_artifact_path(session_indicator, bin_size_cm, time_window_ms, likelihood_function_, filename_ext)
    if deviance_path.exists() and (not force_recompute):
        results["run_deviance_explained_ripples"] = StepResult(
            name="run_deviance_explained_ripples",
            critical=True,
            status="success",
            detail=f"using existing file: {deviance_path}",
        )
        print(f"[OK] run_deviance_explained_ripples: using existing file: {deviance_path}")
    else:
        _run_step(
            results,
            "run_deviance_explained_ripples",
            critical=True,
            dependencies=["run_model_comparison_ripples"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                "run_deviance_explained(data_type=ripples, ...)",
                lambda: run_deviance_explained(
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    likelihood_function=likelihood_function_,
                    filename_ext=filename_ext,
                ),
            ),
        )

    if run_marginals_phase:
        if _marginals_artifacts_complete(session, session_indicator, bin_size_cm, time_window_ms, filename_ext) and (not force_recompute):
            results["get_marginals_ripples"] = StepResult(
                name="get_marginals_ripples",
                critical=False,
                status="success",
                detail="using existing marginals files",
            )
            print("[OK] get_marginals_ripples: using existing marginals files")
        else:
            _run_step(
                results,
                "get_marginals_ripples",
                critical=False,
                dependencies=["reformat_data_for_structure_analysis_ripples"],
                strict=strict,
                action=lambda: _execute_or_print(
                    dry_run,
                    "run_marginals(data_type=ripples, ...)",
                    lambda: run_marginals(
                        data_type=RIPPLES_DATA_TYPE,
                        session=session_indicator,
                        bin_size_cm=bin_size_cm,
                        time_window_ms=time_window_ms,
                        filename_ext=filename_ext,
                    ),
                ),
            )
    else:
        _record_skip(results, "get_marginals_ripples", critical=False, reason="disabled; pass --run-marginals to enable")

    traj_path = _trajectories_artifact_path(session_indicator, bin_size_cm, time_window_ms, likelihood_function_, filename_ext)
    if traj_path.exists() and (not force_recompute):
        results["get_trajectories_ripples"] = StepResult(
            name="get_trajectories_ripples",
            critical=True,
            status="success",
            detail=f"using existing file: {traj_path}",
        )
        print(f"[OK] get_trajectories_ripples: using existing file: {traj_path}")
    else:
        _run_step(
            results,
            "get_trajectories_ripples",
            critical=True,
            dependencies=["reformat_data_for_structure_analysis_ripples"],
            strict=strict,
            action=lambda: _execute_or_print(
                dry_run,
                "run_trajectories(data_type=ripples, ...)",
                lambda: run_trajectories(
                    data_type=RIPPLES_DATA_TYPE,
                    session=session_indicator,
                    bin_size_cm=bin_size_cm,
                    time_window_ms=time_window_ms,
                    likelihood_function=likelihood_function_,
                    sd_meters=trajectory_sd_meters,
                    filename_ext=filename_ext,
                ),
            ),
        )

    if run_diffusion_constant_phase:
        dc_path = _diffusion_constant_inferred_artifact_path(session_indicator, bin_size_cm, time_window_ms, filename_ext)
        if dc_path.exists() and (not force_recompute):
            results["run_diffusion_constant_inferred"] = StepResult(
                name="run_diffusion_constant_inferred",
                critical=False,
                status="success",
                detail=f"using existing file: {dc_path}",
            )
            print(f"[OK] run_diffusion_constant_inferred: using existing file: {dc_path}")
        else:
            _run_step(
                results,
                "run_diffusion_constant_inferred",
                critical=False,
                dependencies=["get_trajectories_ripples"],
                strict=strict,
                action=lambda: _execute_or_print(
                    dry_run,
                    "run_diffusion_constant_analysis(data_type=ripples, trajectory_type=inferred, ...)",
                    lambda: run_diffusion_constant_analysis(
                        data_type=RIPPLES_DATA_TYPE,
                        session=session_indicator,
                        bin_size_cm=bin_size_cm,
                        time_window_ms=time_window_ms,
                        trajectory_type="inferred",
                        filename_ext=filename_ext,
                    ),
                ),
            )
    else:
        _record_skip(
            results,
            "run_diffusion_constant_inferred",
            critical=False,
            reason="disabled; pass --run-diffusion-constant to enable",
        )

    _print_summary(results)
    critical_failures = [result for result in results.values() if result.critical and result.status == "failed"]
    if critical_failures:
        raise SystemExit(1)
    return results
