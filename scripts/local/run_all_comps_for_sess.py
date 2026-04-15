"""Run the per-session ripples dynamics pipeline in dependency order.

This standalone orchestrator targets one real recording session index (0-7) and runs the
single-session ripples computations in the order required by the existing CLI scripts:

1. preprocess_ratday_data
2. preprocess_spikemat_data --data_type ripples
3. reformat_data_for_structure_analysis --data_type ripples
4. run_model --model_name stationary
5. run_model --model_name random
6. diffusion gridsearch
7. stationary_gaussian gridsearch
8. momentum gridsearch for each ripple/spikemat
9. run_model_comparison --data_type ripples
10. run_deviance_explained --data_type ripples
11. get_marginals --data_type ripples (optional)
12. get_trajectories --data_type ripples
13. run_diffusion_constant --trajectory_type inferred (optional)

The simulated model-recovery path in generate_model_recovery_data.py is intentionally not
included here. That script hardcodes Session_List[0] / rat1day1 to estimate priors for
simulated sessions, so it is not part of a generic "single real session" pipeline.

This script also intentionally omits get_descriptive_stats.py and
run_predictive_analysis.py because those scripts aggregate across Session_List rather than
operating on a single session.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import click

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_LOCAL_DIR = SCRIPT_PATH.parent
SCRIPTS_DIR = SCRIPTS_LOCAL_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from replay_structure.metadata import DATA_PATH, SESSION_RATDAY, Session_Name, string_to_data_type, string_to_likelihood_function, string_to_session_indicator
from scripts.o2.o2_lib import submit_diffusion_gridsearch, submit_momentum_gridsearch, submit_stationary_gaussian_gridsearch

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


def _format_command(command: List[str]) -> str:
    return subprocess.list2cmdline(command)


def _build_common_args(session: int, bin_size_cm: int, time_window_ms: int, filename_ext: str) -> List[str]:
    return [
        "--session",
        str(session),
        "--bin_size_cm",
        str(bin_size_cm),
        "--time_window_ms",
        str(time_window_ms),
        "--filename_ext",
        filename_ext,
    ]


def _script_path(script_name: str) -> str:
    return str(SCRIPTS_LOCAL_DIR.joinpath(script_name))


def _ratday_obj_path(session: int, bin_size_cm: int, filename_ext: str, rotate_placefields: bool = False) -> Path:
    session_meta = SESSION_RATDAY[session]
    session_indicator = Session_Name(rat=session_meta["rat"], day=session_meta["day"])
    if rotate_placefields:
        return DATA_PATH.joinpath("ratday", f"{session_indicator}_{bin_size_cm}cm_placefields_rotated{filename_ext}.obj")
    return DATA_PATH.joinpath("ratday", f"{session_indicator}_{bin_size_cm}cm{filename_ext}.obj")


def _run_subprocess(script_name: str, args: List[str], dry_run: bool) -> None:
    command = [sys.executable, _script_path(script_name), *args]
    print(_format_command(command))
    if dry_run:
        return
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(REPO_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


def _record_skip(results: Dict[str, StepResult], name: str, critical: bool, reason: str) -> None:
    results[name] = StepResult(name=name, critical=critical, status="skipped", detail=reason)
    print(f"[SKIP] {name}: {reason}")


def _run_step(
    results: Dict[str, StepResult],
    name: str,
    critical: bool,
    dependencies: List[str],
    strict: bool,
    action: Callable[[], None],
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


def _run_gridsearch_phase(
    session: int,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: str,
    filename_ext: str,
    dry_run: bool,
    continue_on_gridsearch_error: bool,
) -> None:
    session_indicator = string_to_session_indicator(str(session))
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
        print(
            f"Momentum gridsearch completed with {len(momentum_errors)} failures and "
            f"{n_successes} successes."
        )


def _print_summary(results: Dict[str, StepResult]) -> None:
    print("\n=== Summary ===")
    for result in results.values():
        detail = f" ({result.detail})" if result.detail else ""
        critical_label = "critical" if result.critical else "optional"
        print(f"{result.status.upper():8s} [{critical_label}] {result.name}{detail}")


@click.command()
@click.option("--session", type=click.IntRange(0, len(SESSION_RATDAY) - 1), required=True)
@click.option("--bin_size_cm", type=click.INT, default=4, show_default=True)
@click.option("--time_window_ms", type=click.INT, default=DEFAULT_TIME_WINDOW_MS, show_default=True)
@click.option("--likelihood_function", type=click.Choice(["poisson", "negbinomial"]), default=DEFAULT_LIKELIHOOD, show_default=True)
@click.option("--filename_ext", type=click.STRING, default="", show_default=True)
@click.option("--trajectory_sd_meters", type=click.FLOAT, default=DEFAULT_TRAJECTORY_SD_METERS, show_default=True)
@click.option("--skip-gridsearch", is_flag=True, help="Assume gridsearch result inputs already exist.")
@click.option("--force-ratday-preprocess", is_flag=True, help="Regenerate ratday data from OpenFieldData.mat even if a saved ratday .obj already exists.")
@click.option("--run-marginals", is_flag=True, help="Run get_marginals.py after the core model comparison pipeline.")
@click.option("--run-diffusion-constant", is_flag=True, help="Run run_diffusion_constant.py after trajectories are computed.")
@click.option("--continue-on-gridsearch-error", is_flag=True, help="Continue momentum gridsearch after per-spikemat failures when possible.")
@click.option("--strict", is_flag=True, help="Stop immediately on the first step failure.")
@click.option("--dry-run", is_flag=True, help="Print the planned commands and phases without executing them.")
def main(
    session: int,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: str,
    filename_ext: str,
    trajectory_sd_meters: float,
    skip_gridsearch: bool,
    force_ratday_preprocess: bool,
    run_marginals: bool,
    run_diffusion_constant: bool,
    continue_on_gridsearch_error: bool,
    strict: bool,
    dry_run: bool,
) -> None:
    results: Dict[str, StepResult] = {}
    common_args = _build_common_args(session, bin_size_cm, time_window_ms, filename_ext)
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
            action=lambda: _run_subprocess(
                "preprocess_ratday_data.py",
                ["--session", str(session), "--bin_size_cm", str(bin_size_cm), "--filename_ext", filename_ext],
                dry_run,
            ),
        )
    _run_step(
        results,
        "preprocess_spikemat_data_ripples",
        critical=True,
        dependencies=["preprocess_ratday_data"],
        strict=strict,
        action=lambda: _run_subprocess("preprocess_spikemat_data.py", ["--data_type", "ripples", *common_args], dry_run),
    )
    _run_step(
        results,
        "reformat_data_for_structure_analysis_ripples",
        critical=True,
        dependencies=["preprocess_spikemat_data_ripples"],
        strict=strict,
        action=lambda: _run_subprocess(
            "reformat_data_for_structure_analysis.py",
            ["--data_type", "ripples", *common_args, "--likelihood_function", likelihood_function],
            dry_run,
        ),
    )
    _run_step(
        results,
        "run_model_stationary",
        critical=True,
        dependencies=["reformat_data_for_structure_analysis_ripples"],
        strict=strict,
        action=lambda: _run_subprocess(
            "run_model.py",
            ["--model_name", "stationary", "--data_type", "ripples", *common_args, "--likelihood_function", likelihood_function],
            dry_run,
        ),
    )
    _run_step(
        results,
        "run_model_random",
        critical=True,
        dependencies=["reformat_data_for_structure_analysis_ripples"],
        strict=strict,
        action=lambda: _run_subprocess(
            "run_model.py",
            ["--model_name", "random", "--data_type", "ripples", *common_args, "--likelihood_function", likelihood_function],
            dry_run,
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
        action=lambda: _run_subprocess(
            "run_model_comparison.py",
            ["--data_type", "ripples", *common_args, "--likelihood_function", likelihood_function],
            dry_run,
        ),
    )
    _run_step(
        results,
        "run_deviance_explained_ripples",
        critical=True,
        dependencies=["run_model_comparison_ripples"],
        strict=strict,
        action=lambda: _run_subprocess(
            "run_deviance_explained.py",
            ["--data_type", "ripples", *common_args, "--likelihood_function", likelihood_function],
            dry_run,
        ),
    )

    if run_marginals:
        _run_step(
            results,
            "get_marginals_ripples",
            critical=False,
            dependencies=["reformat_data_for_structure_analysis_ripples"],
            strict=strict,
            action=lambda: _run_subprocess("get_marginals.py", ["--data_type", "ripples", *common_args], dry_run),
        )
    else:
        _record_skip(results, "get_marginals_ripples", critical=False, reason="disabled; pass --run-marginals to enable")

    _run_step(
        results,
        "get_trajectories_ripples",
        critical=True,
        dependencies=["reformat_data_for_structure_analysis_ripples"],
        strict=strict,
        action=lambda: _run_subprocess(
            "get_trajectories.py",
            ["--data_type", "ripples", *common_args, "--likelihood_function", likelihood_function, "--sd_meters", str(trajectory_sd_meters)],
            dry_run,
        ),
    )

    if run_diffusion_constant:
        _run_step(
            results,
            "run_diffusion_constant_inferred",
            critical=False,
            dependencies=["get_trajectories_ripples"],
            strict=strict,
            action=lambda: _run_subprocess(
                "run_diffusion_constant.py",
                ["--data_type", "ripples", *common_args, "--trajectory_type", "inferred"],
                dry_run,
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
    raise SystemExit(1 if critical_failures else 0)


if __name__ == "__main__":
    main()
