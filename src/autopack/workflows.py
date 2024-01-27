import warnings
from typing import Optional

import xarray as xr

from . import __version__, logger
from .cost_fields import cost_field_vis, create_ergo_cost_fields, create_ips_cost_field
from .data_model import (
    DatasetVariableAttrs,
    ErgoSettings,
    HarnessSetup,
    ProblemSetup,
    StudySettings,
)
from .harness_optimization import build_optimization_problem
from .io import load_scene
from .ips import IPSError, IPSInstance
from .optimization import minimize


def build_problem(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    ergo_settings: Optional[ErgoSettings] = None,
):
    load_scene(ips, harness_setup.scene_path, clear=True)

    ips_cost_field = create_ips_cost_field(ips, harness_setup)

    if ergo_settings:
        ergo_cost_fields = create_ergo_cost_fields(
            ips=ips,
            harness_setup=harness_setup,
            ergo_settings=ergo_settings,
            ref_cost_field=ips_cost_field,
        )
    else:
        ergo_cost_fields = []

    problem_setup = ProblemSetup(
        harness_setup=harness_setup,
        ergo_settings=ergo_settings,
        cost_fields=[ips_cost_field, *ergo_cost_fields],
    )
    for cost_field in problem_setup.cost_fields:
        cost_field_vis(ips=ips, cost_field=cost_field, visible=False)

    return problem_setup


def run_study(
    ips: IPSInstance,
    problem_setup: ProblemSetup,
    study_settings: StudySettings,
):
    if (
        problem_setup.harness_setup.allow_infeasible_topology
        and study_settings.opt_batches > 1
    ):
        raise ValueError(
            "Cannot optimize harnesses with infeasible topology. Set allow_infeasible_topology to False in the harness setup or run with batches=0 to disable optimization."
        )

    opt_problem = build_optimization_problem(ips=ips, problem_setup=problem_setup)

    try:
        with warnings.catch_warnings():
            if study_settings.silence_warnings:
                # Silence warnings from within botorch and linear_operator
                warnings.filterwarnings(
                    category=RuntimeWarning, module="botorch", action="ignore"
                )
                warnings.filterwarnings(
                    category=RuntimeWarning, module="linear_operator", action="ignore"
                )

            minimize(
                problem=opt_problem,
                batches=study_settings.opt_batches,
                batch_size=study_settings.opt_batch_size,
                init_samples=study_settings.doe_samples,
                seed=0,
            )
    except IPSError as exc:
        if study_settings.return_partial_results:
            logger.exception(
                "Optimization failed due to IPS error. Trying to continue with the data gathered until this happened.",
                exc_info=exc,
            )
        else:
            raise exc

    dataset = xr.concat(opt_problem.state["batch_datasets"], dim="solution")
    dataset.attrs["autopack_version"] = __version__
    dataset.attrs["ips_version"] = ips.version
    dataset.attrs["problem_setup"] = problem_setup
    dataset.attrs["study_settings"] = study_settings

    DatasetVariableAttrs.apply(dataset)

    return dataset


def build_problem_and_run_study(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    study_settings: StudySettings,
    ergo_settings: Optional[ErgoSettings] = None,
):
    problem_setup = build_problem(
        ips=ips,
        harness_setup=harness_setup,
        ergo_settings=ergo_settings,
    )

    return run_study(
        ips=ips,
        problem_setup=problem_setup,
        study_settings=study_settings,
    )
