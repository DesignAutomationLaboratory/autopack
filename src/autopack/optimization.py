import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from pydantic import BaseModel

from . import data_model
from .harness_optimization import route_evaluate_harness

CUDA_AVAILABLE = torch.cuda.is_available()

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if CUDA_AVAILABLE else "cpu"),
}


class OptimizationResult(BaseModel, arbitrary_types_allowed=True):
    x: np.ndarray
    obj: np.ndarray
    con: np.ndarray


class OptimizationProblem(BaseModel, arbitrary_types_allowed=True):
    func: callable
    bounds: np.ndarray | torch.Tensor
    num_objectives: int
    num_constraints: int = 0
    ref_point: Optional[np.ndarray | torch.Tensor] = None
    state: Optional[Any] = None


class OptimizationMeta(BaseModel, arbitrary_types_allowed=True):
    category: str
    batch: int


def batch_dss(
    ips_instance,
    problem_setup: data_model.ProblemSetup,
    xs: np.ndarray,
    meta: OptimizationMeta,
):
    cost_field_ids = [cf.name for cf in problem_setup.cost_fields]

    for i, x in enumerate(xs):
        case_id = f"{meta.category}.{meta.batch}.{i}"

        bundle_costs, total_costs, num_clips = route_evaluate_harness(
            ips_instance,
            problem_setup,
            cost_field_weights=x[:-1],
            bundling_factor=x[-1],
            harness_id=case_id,
        )

        num_ips_solutions = 1
        ds = xr.Dataset(
            {
                "timestamp": pd.Timestamp.utcnow(),
                "cost_field_weight": xr.DataArray(
                    x[:-1], coords={"cost_field": cost_field_ids}
                ),
                "bundling_factor": xr.DataArray(x[-1]),
                "bundling_cost": xr.DataArray(
                    bundle_costs,
                    # dims=["cost_field", "ips_solution"],
                    coords={
                        "ips_solution": range(num_ips_solutions),
                        "cost_field": cost_field_ids,
                    },
                ),
                "total_cost": xr.DataArray(
                    total_costs,
                    # dims=["cost_field", "ips_solution"],
                    coords={
                        "ips_solution": range(num_ips_solutions),
                        "cost_field": cost_field_ids,
                    },
                ),
                "num_estimated_clips": xr.DataArray(
                    num_clips,
                    coords=[range(num_ips_solutions)],
                    dims=["ips_solution"],
                ),
            }
        )
        ds = ds.expand_dims({"case": [case_id]}, axis=0)
        yield ds


def batch_voi(
    batch_ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the dataset for a full batch and returns the variables of
    interest (xs, objs, cons)
    """
    objs = (
        batch_ds[["bundling_cost", "total_cost", "num_estimated_clips"]]
        .stack(combined=["case", "ips_solution"])
        .to_stacked_array(new_dim="obj", sample_dims=["combined"], name="objectives")
        .dropna("combined")
    )
    cons = np.empty((objs.shape[0], 0), dtype=float)

    # Index the xs by the case from obj, so we get the corresponding x
    # for each combined case and ips_solution
    xs = (
        batch_ds[["cost_field_weight", "bundling_factor"]]
        .to_stacked_array(
            new_dim="desvar", sample_dims=["case"], name="design_variables"
        )
        .sel(case=objs.case)
    )

    return (
        xs.values,
        objs.values,
        cons,
    )


def problem_from_setup(problem_setup, ips_instance) -> OptimizationProblem:
    num_cost_fields = len(problem_setup.cost_fields)
    num_dims = num_cost_fields + 1
    num_objectives = 2 * num_cost_fields + 1
    weights_bounds = np.array([[0.001, 1.0]] * num_cost_fields)
    bundling_factor_bounds = np.array([[0.05, 0.9]])
    bounds = np.array([*weights_bounds, *bundling_factor_bounds])

    batch_datasets = []

    def batch_analysis_func(xs: np.ndarray, meta: OptimizationMeta):
        this_batch_dataset = xr.concat(
            batch_dss(
                ips_instance=ips_instance, problem_setup=problem_setup, xs=xs, meta=meta
            ),
            dim="case",
        )
        batch_datasets.append(this_batch_dataset)

        return batch_voi(this_batch_dataset)

    return OptimizationProblem(
        func=batch_analysis_func,
        bounds=bounds,
        num_objectives=num_objectives,
        num_constraints=0,
        state={"batch_datasets": batch_datasets},
    )


def global_optimize_harness(
    ips_instance: data_model.IPSInstance,
    problem_setup: data_model.ProblemSetup,
    init_samples: int = 8,
    batches: int = 4,
    batch_size: int = 4,
) -> xr.Dataset:
    problem = problem_from_setup(problem_setup, ips_instance)
    minimize(
        problem=problem,
        batches=batches,
        batch_size=batch_size,
        init_samples=init_samples,
    )

    dataset = xr.concat(problem.state["batch_datasets"], dim="case")
    dataset.attrs["problem_setup"] = problem_setup
    dataset.attrs["init_samples"] = init_samples
    dataset.attrs["batches"] = batches
    dataset.attrs["batch_size"] = batch_size
    dataset.attrs["ips_version"] = ips_instance.version

    return dataset


def initialize_model(problem, train_x, train_obj, train_con):
    # define models for objective and constraint
    train_x = normalize(train_x, bounds=problem.bounds.T)
    train_y = torch.cat([train_obj, train_con], dim=-1)
    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def evaluate_batch(problem, xs, meta):
    xs, objs, cons = problem.func(
        xs.cpu().numpy(),
        meta,
    )

    return (
        torch.tensor(xs, **tkwargs),
        # BoTorch assumes maximization
        torch.tensor(-objs, **tkwargs),
        # BoTorch constraints are of the form g(x) <= 0
        torch.tensor(cons, **tkwargs),
    )


def feasible_pareto_front(objs, cons):
    is_feas = (cons <= 0).all(dim=-1)
    feas_objs = objs[is_feas]
    if feas_objs.shape[0] > 0:
        pareto_mask = is_non_dominated(feas_objs)
        return feas_objs[pareto_mask]
    else:
        return torch.empty(0, objs.shape[-1], **tkwargs)


def auto_ref_point(objs, cons):
    pareto_objs = feasible_pareto_front(objs, cons)
    if pareto_objs.shape[0] > 0:
        used_objs = pareto_objs
    else:
        used_objs = objs

    objs_diff = used_objs.max(dim=0).values - used_objs.min(dim=0).values
    # If there is no difference between the max and min, use 1 as
    # placeholder to guarantee that the ref point will always be worse
    # in all objectives
    objs_diff[objs_diff == 0] = 1

    return used_objs.min(dim=0).values - objs_diff * 0.1


def optimize_qnehvi_and_get_candidates(
    problem,
    train_x,
    train_obj,
    train_con,
    model,
    sampler,
    standard_bounds,
    batch_size,
    restarts,
    raw_samples,
):
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, bounds=problem.bounds.T)

    # define an objective that specifies which outcomes are the objectives
    objective = IdentityMCMultiOutputObjective(
        outcomes=list(range(problem.num_objectives))
    )
    # specify that the constraints are on the last outcomes
    if problem.num_constraints > 0:
        constraints = [lambda Z: Z[..., i] for i in range(-problem.num_objectives, 0)]
    else:
        constraints = None

    if problem.ref_point is None:
        ref_point = auto_ref_point(train_obj, train_con)
        assert Hypervolume(ref_point).compute(train_obj) > 0, "Ref point is bad"
    else:
        # BoTorch assumes maximization, but the input ref point assumes
        # minimization
        ref_point = -problem.ref_point

    # Use a non-zero alpha for high-dimensional problems, to use an
    # approximated partitioning scheme for radically faster computation
    # FIXME: this needs tuning
    qnehvi_alpha = 0.1 if problem.num_objectives > 4 else 0.0

    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        alpha=qnehvi_alpha,
        objective=objective,
        constraints=constraints,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    new_x = unnormalize(candidates.detach(), bounds=problem.bounds.T)

    return new_x


def minimize(
    problem: OptimizationProblem,
    batches=8,
    batch_size=8,
    init_samples=8,
    init_seed=None,
    mc_samples=128,
    restarts=10,
    raw_samples=512,
):
    problem.bounds = torch.tensor(problem.bounds, **tkwargs)
    problem.ref_point = (
        torch.tensor(problem.ref_point, **tkwargs)
        if problem.ref_point is not None
        else None
    )

    num_dims = problem.bounds.shape[0]
    standard_bounds = torch.zeros(2, num_dims, **tkwargs)
    standard_bounds[1] = 1

    sampler_xs = draw_sobol_samples(
        bounds=problem.bounds.T,
        n=init_samples,
        q=1,
        seed=init_seed,
    ).squeeze(1)

    train_xs, train_objs, train_cons = evaluate_batch(
        problem,
        sampler_xs,
        meta=OptimizationMeta(category="sobol", batch=0),
    )

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(batches):
        t0 = time.monotonic()

        # Note: we find improved performance from not warm
        # starting the model hyperparameters using the hyperparameters
        # from the previous iteration
        mll_qnehvi, model_qnehvi = initialize_model(
            problem=problem,
            train_x=train_xs,
            train_obj=train_objs,
            train_con=train_cons,
        )

        # fit the models
        fit_gpytorch_mll(mll_qnehvi)

        # define the qNEHVI acquisition module using a QMC sampler
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        # optimize acquisition functions and get new observations
        candidate_xs = optimize_qnehvi_and_get_candidates(
            problem=problem,
            train_x=train_xs,
            train_obj=train_objs,
            train_con=train_cons,
            model=model_qnehvi,
            sampler=qnehvi_sampler,
            standard_bounds=standard_bounds,
            batch_size=batch_size,
            restarts=restarts,
            raw_samples=raw_samples,
        )

        new_x, new_obj, new_con = evaluate_batch(
            problem,
            candidate_xs,
            meta=OptimizationMeta(
                category="qnehvi",
                batch=iteration,
            ),
        )

        # update training points
        train_xs = torch.cat([train_xs, new_x])
        train_objs = torch.cat([train_objs, new_obj])
        train_cons = torch.cat([train_cons, new_con])

        t1 = time.monotonic()

    return OptimizationResult(
        x=train_xs.cpu().numpy(),
        obj=-train_objs.cpu().numpy(),
        con=train_cons.cpu().numpy(),
    )
