import random

import numpy as np
import pytest
import torch
import xarray as xr
from botorch.test_functions.multi_objective import DTLZ1

from autopack.optimization import OptimizationProblem, OptimizationResult, minimize


def create_binh_korn_problem(constrained):
    def binh_korn_obj(x):
        return np.array(
            [
                4 * x[0] ** 2 + 4 * x[1] ** 2,
                (x[0] - 5) ** 2 + (x[1] - 5) ** 2,
            ]
        )

    def binh_korn_con(x):
        return np.array(
            [
                (x[0] - 5) ** 2 + x[1] ** 2 - 25,
                -((x[0] - 8) ** 2) - (x[1] + 3) ** 2 + 7.7,
            ]
        )

    return OptimizationProblem(
        obj_func=binh_korn_obj,
        con_func=binh_korn_con if constrained else None,
        bounds=np.array([[0, 5], [0, 3]]),
        num_objectives=2,
        num_constraints=2 if constrained else 0,
        ref_point=np.array([140, 50]),
    )


def create_faux_autopack_analysis_problem(
    num_weights, cost_multiplier, clip_multiplier, datasets
):
    """
    Create a problem that mimics the Autopack problem, but is much
    faster to evaluate.

    The Autopack problem can have a variable number of design variables
    (weights) in addition to a bundling parameter, and a fixed number of
    objectives. The first objective is the accumulated cost according to
    the weighted cost field, and the second objective is the number of
    clips used.

    Here, a test problem is made to mimic this behavior by multiplying
    the underlying test function's objective values by the
    `cost_multiplier` and `clip_multiplier`, respectively.
    """
    num_dims = num_weights + 1

    test_problem = DTLZ1(dim=num_dims, num_objectives=2, noise_std=0.3)

    ref_point = np.array(
        [
            test_problem._ref_val * cost_multiplier * 1.1,
            test_problem._ref_val * clip_multiplier * 1.1,
        ]
    )

    def obj_func(x):
        raw_obj = test_problem.forward(torch.tensor(x)).cpu().numpy()
        return np.array(
            [
                raw_obj[0] * cost_multiplier + 100,
                np.round(raw_obj[1] * clip_multiplier) + 10,
            ]
        )

    def analysis_func(xs, source, batch):
        this_batch_datasets = []

        for i, x in enumerate(xs):
            costs = []
            clipss = []
            num_ips_solutions = random.randint(1, 5)
            for r in range(num_ips_solutions):
                cost, clips = obj_func(x)
                costs.append(cost)
                clipss.append(clips)

            ds = xr.Dataset(
                {
                    "cost_field_weight": xr.DataArray(
                        x, dims=["cost_field"], coords=[range(len(x))]
                    ),
                    "cost": xr.DataArray(
                        costs,
                        dims=["ips_solution"],
                        coords=[range(num_ips_solutions)],
                    ),
                    "clips": xr.DataArray(
                        clipss,
                        dims=["ips_solution"],
                        coords=[range(num_ips_solutions)],
                    ),
                }
            )
            ds = ds.expand_dims({"case": [f"{source}.{batch}.{i}"]}, axis=0)
            this_batch_datasets.append(ds)

        this_batch_dataset = xr.concat(this_batch_datasets, dim="case")
        datasets.append(this_batch_dataset)

        objs = (
            this_batch_dataset[["cost", "clips"]]
            .stack(combined=["case", "ips_solution"])
            .dropna("combined")
            .to_array()
            .T
        )
        cons = np.empty((objs.shape[0], 0), dtype=float)

        # Index the cost field weight by the case
        returned_xs = this_batch_dataset["cost_field_weight"].sel(case=objs.case)

        return (
            returned_xs.values,
            objs.values,
            cons,
        )

    return OptimizationProblem(
        func=analysis_func,
        bounds=np.array([[0, 1]] * num_dims),
        num_objectives=2,
        num_constraints=0,
        ref_point=ref_point,
    )


@pytest.mark.parametrize("constrained", [True, False])
def test_minimize_binh_korn_smoke(constrained):
    problem = create_binh_korn_problem(constrained)

    # Smoke test, just make sure it runs at all
    result = minimize(
        problem=problem,
        batches=2,
        batch_size=2,
    )

    assert isinstance(result, OptimizationResult)
    assert result.x.shape[0] >= 4
    assert result.obj.shape[0] == result.x.shape[0]
    assert result.con.shape[0] == result.x.shape[0]
    assert result.x.shape[1] == 2
    assert result.obj.shape[1] == 2
    assert result.con.shape[1] == (2 if constrained else 0)


@pytest.mark.parametrize("num_weights", [2, 3, 4])
@pytest.mark.parametrize("cost_multiplier", [10])
@pytest.mark.parametrize("clip_multiplier", [1])
def test_faux_autopack_smoke(num_weights, cost_multiplier, clip_multiplier):
    datasets = []

    problem = create_faux_autopack_analysis_problem(
        num_weights=num_weights,
        cost_multiplier=cost_multiplier,
        clip_multiplier=clip_multiplier,
        datasets=datasets,
    )

    result = minimize(
        problem=problem,
        batches=2,
        batch_size=2,
    )

    full_dataset = xr.concat(datasets, dim="case")

    # Check that the hypervolume is increasing, if ever so slightly
    assert False
