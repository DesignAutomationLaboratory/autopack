import time
from typing import Optional

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

CUDA_AVAILABLE = torch.cuda.is_available()

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if CUDA_AVAILABLE else "cpu"),
}


class OptimizationResult(BaseModel, arbitrary_types_allowed=True):
    x: np.ndarray
    obj: np.ndarray
    con: np.ndarray


class OptimizationProblem(BaseModel, arbitrary_types_allowed=True):
    obj_func: callable
    con_func: Optional[callable] = None
    bounds: np.ndarray | torch.Tensor
    num_objectives: int
    num_constraints: int = 0
    ref_point: np.ndarray | torch.Tensor


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


def evaluate_problem(problem, x):
    # BoTorch assumes maximization
    obj = -problem.obj_func(x.cpu().numpy())
    if problem.num_constraints > 0:
        # BoTorch constraints are of the form g(x) <= 0
        con = problem.con_func(x.cpu().numpy())
    else:
        con = []

    return torch.tensor(obj, **tkwargs), torch.tensor(con, **tkwargs)


def evaluate_batch(problem, xs):
    objs = []
    cons = []

    for x in xs:
        obj, con = evaluate_problem(problem, x)
        objs.append(obj)
        cons.append(con)

    return torch.stack(objs), torch.stack(cons)


def generate_initial_data(problem, n):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds.T, n=n, q=1).squeeze(1)

    train_obj, train_con = evaluate_batch(problem, train_x)

    return train_x, train_obj, train_con


def optimize_qnehvi_and_get_observation(
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

    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=(-problem.ref_point).tolist(),  # use known reference point
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
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

    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds.T)
    new_obj, new_con = evaluate_batch(problem, new_x)

    return new_x, new_obj, new_con


def minimize(
    problem: OptimizationProblem,
    batches=8,
    batch_size=8,
    mc_samples=128,
    restarts=10,
    raw_samples=512,
    progress=True,
):
    problem.bounds = torch.tensor(problem.bounds, **tkwargs)
    problem.ref_point = torch.tensor(problem.ref_point, **tkwargs)

    num_dims = problem.bounds.shape[0]
    standard_bounds = torch.zeros(2, num_dims, **tkwargs)
    standard_bounds[1] = 1

    # call helper functions to generate initial training data and initialize model
    train_x_qnehvi, train_obj_qnehvi, train_con_qnehvi = generate_initial_data(
        problem=problem, n=2 * (num_dims + 1)
    )
    mll_qnehvi, model_qnehvi = initialize_model(
        problem=problem,
        train_x=train_x_qnehvi,
        train_obj=train_obj_qnehvi,
        train_con=train_con_qnehvi,
    )

    if progress:
        hv = Hypervolume(ref_point=-problem.ref_point)
        hvs_qnehvi = []

        # compute pareto front
        is_feas = (train_con_qnehvi <= 0).all(dim=-1)
        feas_train_obj = train_obj_qnehvi[is_feas]
        if feas_train_obj.shape[0] > 0:
            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]
            # compute hypervolume
            volume = hv.compute(pareto_y)
        else:
            volume = 0.0

        hvs_qnehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(batches):
        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll_qnehvi)

        # define the qNEHVI acquisition module using a QMC sampler
        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        # optimize acquisition functions and get new observations
        (
            new_x_qnehvi,
            new_obj_qnehvi,
            new_con_qnehvi,
        ) = optimize_qnehvi_and_get_observation(
            problem=problem,
            train_x=train_x_qnehvi,
            train_obj=train_obj_qnehvi,
            train_con=train_con_qnehvi,
            model=model_qnehvi,
            sampler=qnehvi_sampler,
            standard_bounds=standard_bounds,
            batch_size=batch_size,
            restarts=restarts,
            raw_samples=raw_samples,
        )

        # update training points
        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
        train_con_qnehvi = torch.cat([train_con_qnehvi, new_con_qnehvi])

        if progress:
            # compute pareto front
            is_feas = (train_con_qnehvi <= 0).all(dim=-1)
            feas_train_obj = train_obj_qnehvi[is_feas]
            if feas_train_obj.shape[0] > 0:
                pareto_mask = is_non_dominated(feas_train_obj)
                pareto_y = feas_train_obj[pareto_mask]
                # compute feasible hypervolume
                volume = hv.compute(pareto_y)
            else:
                volume = 0.0
            hvs_qnehvi.append(volume)

        # reinitialize the models so they are ready for fitting on next
        # iteration
        # Note: we find improved performance from not warm
        # starting the model hyperparameters using the hyperparameters
        # from the previous iteration
        mll_qnehvi, model_qnehvi = initialize_model(
            problem=problem,
            train_x=train_x_qnehvi,
            train_obj=train_obj_qnehvi,
            train_con=train_con_qnehvi,
        )

        t1 = time.monotonic()

        if progress:
            print(
                f"\nBatch {iteration+1:>2}: Hypervolume (qNEHVI) = "
                f"({hvs_qnehvi[-1]:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )

    return OptimizationResult(
        x=train_x_qnehvi.cpu().numpy(),
        obj=-train_obj_qnehvi.cpu().numpy(),
        con=train_con_qnehvi.cpu().numpy(),
    )
