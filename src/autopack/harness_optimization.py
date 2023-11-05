import numpy as np
import pandas as pd
import xarray as xr

from . import data_model
from .data_model import CostField, IPSInstance, ProblemSetup
from .ips_communication.ips_commands import route_harness
from .optimization import OptimizationMeta, OptimizationProblem, minimize


def route_evaluate_harness(
    ips_instance,
    problem_setup,
    cost_field_weights,
    bundling_factor,
    harness_id=None,
):
    new_field = combine_cost_fields(
        problem_setup.cost_fields, cost_field_weights, normalize_fields=True
    )
    new_harness = route_harness(
        ips=ips_instance,
        harness_setup=problem_setup.harness_setup,
        cost_field=new_field,
        bundling_factor=bundling_factor,
        harness_id=harness_id,
    )
    bundle_costs = []
    total_costs = []

    for cost_field in problem_setup.cost_fields:
        bundle_cost, total_cost = evaluate_harness(new_harness, cost_field)
        bundle_costs.append(bundle_cost)
        total_costs.append(total_cost)

    # bundle_cost, total_cost = harness.evaluate_harness(new_harness, problem_setup.cost_fields[0])
    return (
        np.array([bundle_costs]),
        np.array([total_costs]),
        np.array([new_harness.numb_of_clips], dtype=int),
    )


def combine_cost_fields(cost_fields, weights, normalize_fields=True):
    coords = cost_fields[0].coordinates
    costs = np.zeros(np.shape(cost_fields[0].costs), dtype=float)
    name = "weighted"
    for i in range(len(cost_fields)):
        if normalize_fields:
            costs = costs + weights[i] * cost_fields[i].normalized_costs()
        else:
            costs = costs + weights[i] * cost_fields[i].costs
    return CostField(name=name, coordinates=coords, costs=costs)


def evaluate_harness(harness, cost_field):
    bundle_cost = 0
    total_cost = 0
    for segment in harness.harness_segments:
        for i in range(len(segment.points) - 1):
            start_node = segment.points[i]
            end_node = segment.points[i + 1]
            start_coord = cost_field.coordinates[
                start_node[0], start_node[1], start_node[2]
            ]
            end_coord = cost_field.coordinates[end_node[0], end_node[1], end_node[2]]
            distance = (
                (end_coord[0] - start_coord[0]) ** 2
                + (end_coord[1] - start_coord[1]) ** 2
                + (end_coord[2] - start_coord[2]) ** 2
            ) ** 0.5
            start_cost = cost_field.costs[start_node[0], start_node[1], start_node[2]]
            end_cost = cost_field.costs[end_node[0], end_node[1], end_node[2]]
            cost = (start_cost + end_cost) / 2 * distance
            bundle_cost = bundle_cost + cost
            total_cost = total_cost + cost * len(segment.cables)
    return bundle_cost, total_cost


def route_harness_from_dataset(
    ips: IPSInstance,
    ds: xr.Dataset,
    case_id: str,
    ips_solution_idx: int = 0,
    build_discrete_solution=False,
    build_presmooth_solution=False,
    build_smooth_solution=False,
):
    problem_setup: ProblemSetup = ds.attrs["problem_setup"]

    selected_ds = ds.sel(case=case_id, ips_solution=ips_solution_idx)
    cost_field_weights = selected_ds["cost_field_weight"].values
    bundling_factor = selected_ds["bundling_factor"].values

    combined_cf = combine_cost_fields(problem_setup.cost_fields, cost_field_weights)

    # This assumes that the harness router is deterministic and
    # will always return the same harness for the same inputs.
    # FIXME: investigate whether this is true.
    return route_harness(
        ips=ips,
        harness_setup=problem_setup.harness_setup,
        cost_field=combined_cf,
        bundling_factor=bundling_factor,
        harness_id=case_id,
        solutions_to_capture=[ips_solution_idx],
        build_discrete_solution=build_discrete_solution,
        build_presmooth_solution=build_presmooth_solution,
        build_smooth_solution=build_smooth_solution,
    )


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
