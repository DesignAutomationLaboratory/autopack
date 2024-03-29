import os
import time
import warnings
from typing import Any, Callable, Optional

import numpy as np
import torch
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

from . import logger


def cuda_selftest():
    # Make a simple self-test to ensure that CUDA is working

    x = torch.tensor([[1.0, 2.0, 3.0]]).T
    y = torch.tensor([[1.0, 4.0, 9.0]]).T

    # Do we get tensors on the GPU?
    assert x.device.type == "cuda"
    assert y.device.type == "cuda"

    # Can PyTorch actually work?
    with warnings.catch_warnings(action="ignore"):
        model = ModelListGP(SingleTaskGP(x, y))
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)


def update_torch_settings(cuda_requested=None):
    # FIXME: stop changing global state

    torch.set_default_dtype(torch.double)

    if cuda_requested is None:
        cuda_requested = os.environ.get("AUTOPACK_USE_CUDA", "true").lower() == "true"
    cuda_available = torch.cuda.is_available()
    if cuda_requested and cuda_available:
        torch.set_default_device(torch.device("cuda"))
        try:
            cuda_selftest()
        except Exception as exc:
            logger.exception("CUDA requested, but not usable", exc_info=exc)
        else:
            logger.info(f"Using GPU for optimization: {torch.cuda.get_device_name()}")
            return
    elif cuda_requested and not cuda_available:
        logger.warning("CUDA requested, but not available")
    torch.set_default_device(torch.device("cpu"))
    logger.info("Using CPU for optimization")


class OptimizationResult(BaseModel, arbitrary_types_allowed=True):
    x: np.ndarray
    obj: np.ndarray
    con: np.ndarray


class OptimizationMeta(BaseModel, arbitrary_types_allowed=True):
    category: str
    batch: int


class OptimizationProblem(BaseModel, arbitrary_types_allowed=True):
    func: Callable[
        [np.ndarray, OptimizationMeta], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
    bounds: np.ndarray | torch.Tensor
    num_objectives: int
    num_constraints: int = 0
    ref_point: Optional[np.ndarray | torch.Tensor] = None
    state: Optional[Any] = None


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
    logger.debug(f"Evaluating {meta}: {xs}")
    xs, objs, cons = problem.func(
        xs.cpu().numpy(),
        meta,
    )

    assert xs.shape[-1] == problem.bounds.shape[0], "Design space mismatch"
    assert np.isfinite(xs).all(), "Non-finite values in design space"
    assert objs.shape[-1] == problem.num_objectives, "Objective space mismatch"
    assert np.isfinite(objs).all(), "Non-finite values in objective space"
    assert cons.shape[-1] == problem.num_constraints, "Constraint space mismatch"
    assert np.isfinite(cons).all(), "Non-finite values in constraint space"

    return (
        torch.tensor(xs),
        # BoTorch assumes maximization
        torch.tensor(-objs),
        # BoTorch constraints are of the form g(x) <= 0
        torch.tensor(cons),
    )


def feasible_pareto_front(objs, cons):
    is_feas = (cons <= 0).all(dim=-1)
    feas_objs = objs[is_feas]
    if feas_objs.shape[0] > 0:
        pareto_mask = is_non_dominated(feas_objs)
        return feas_objs[pareto_mask]
    else:
        return torch.empty(0, objs.shape[-1])


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
        logger.debug(f"Using auto reference point: {ref_point}")
        assert Hypervolume(ref_point).compute(train_obj) > 0, "Ref point is bad"
    else:
        # BoTorch assumes maximization, but the input ref point assumes
        # minimization
        ref_point = -problem.ref_point

    # Use a non-zero alpha for high-dimensional problems, to use an
    # approximated partitioning scheme for radically faster computation
    # FIXME: this needs tuning
    qnehvi_alpha = 0.1 * np.fmax(0, problem.num_objectives - 3)

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
    logger.debug("Acquisition function built. Optimizing...")
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
    mc_samples=128,
    restarts=10,
    raw_samples=512,
    seed=None,
    use_cuda=None,
):
    logger.notice(
        f"Optimizing in {batches} batches of {batch_size} samples each, with {init_samples} initial samples"
    )
    update_torch_settings(cuda_requested=use_cuda)
    problem.bounds = torch.tensor(problem.bounds)
    problem.ref_point = (
        torch.tensor(problem.ref_point) if problem.ref_point is not None else None
    )

    num_dims = problem.bounds.shape[0]
    standard_bounds = torch.zeros(2, num_dims)
    standard_bounds[1] = 1

    sampler_xs = draw_sobol_samples(
        bounds=problem.bounds.T,
        n=init_samples,
        q=1,
        seed=seed,
    ).squeeze(1)

    train_xs, train_objs, train_cons = evaluate_batch(
        problem,
        sampler_xs,
        meta=OptimizationMeta(category="sobol", batch=0),
    )

    if seed is not None:
        torch.manual_seed(seed)
    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(batches):
        logger.info(f"Starting optimization batch {iteration + 1} of {batches}...")
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
                batch=iteration + 1,
            ),
        )

        # update training points
        train_xs = torch.cat([train_xs, new_x])
        train_objs = torch.cat([train_objs, new_obj])
        train_cons = torch.cat([train_cons, new_con])

        t1 = time.monotonic()
        logger.notice(f"Batch {iteration + 1} finished after {t1 - t0:.2f}s")

    return OptimizationResult(
        x=train_xs.cpu().numpy(),
        obj=-train_objs.cpu().numpy(),
        con=train_cons.cpu().numpy(),
    )
