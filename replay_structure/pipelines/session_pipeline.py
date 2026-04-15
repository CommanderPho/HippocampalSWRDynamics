from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from replay_structure.metadata import (
    DATA_PATH,
    SESSION_RATDAY,
    Session_Name,
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
    force_ratday_preprocess: bool = False,
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

    if ratday_obj_path.exists() and (not force_ratday_preprocess):
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
        


    ripple_spikemat_path = Path(r"H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\data\ripples\rat2day1_4cm_3ms.obj").resolve()
    if ripple_spikemat_path.exists() and (not force_ratday_preprocess):
        ## TODO: just get the loaded result and don't waste time recomputing
        # results["preprocess_ratday_data"] = StepResult(
        #     name="preprocess_ratday_data",
        #     critical=True,
        #     status="success",
        #     detail=f"using existing ratday file: {ratday_obj_path}",
        # )
        # print(f"[OK] preprocess_ratday_data: using existing ratday file: {ratday_obj_path}")

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
