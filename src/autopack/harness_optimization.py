import numpy as np
import pandas as pd
import xarray as xr

from . import data_model
from .data_model import CostField, IPSInstance, ProblemSetup
from .ips_communication.ips_commands import route_harnesses
from .optimization import OptimizationMeta, OptimizationProblem, minimize
from .utils import path_length


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


def harness_volume(harness: data_model.Harness) -> float:
    # Use the first segments to see what kind of coordinates we have
    first_segment = harness.harness_segments[0]

    if first_segment.smooth_coords is not None:
        segment_lengths = np.array(
            [path_length(seg.smooth_coords) for seg in harness.harness_segments]
        )
    elif len(first_segment.presmooth_coords) > 0:
        segment_lengths = np.array(
            [path_length(seg.presmooth_coords) for seg in harness.harness_segments]
        )
    else:
        segment_lengths = np.array(
            [path_length(seg.discrete_coords) for seg in harness.harness_segments]
        )

    segment_radii = np.array([seg.radius for seg in harness.harness_segments])

    return np.sum(segment_lengths * segment_radii**2 * np.pi)


def route_harness_from_dataset(
    ips: IPSInstance,
    ds: xr.Dataset,
    case_id: str,
    ips_solution_idx: int = 0,
    smooth_solution=False,
    build_discrete_solution=False,
    build_presmooth_solution=False,
    build_smooth_solution=False,
    build_cable_simulation=False,
) -> data_model.Harness:
    problem_setup: ProblemSetup = ds.attrs["problem_setup"]

    selected_ds = ds.sel(case=case_id, ips_solution=ips_solution_idx)
    cost_field_weights = selected_ds["cost_field_weight"].values
    bundling_weight = selected_ds["bundling_weight"].values

    combined_cf = combine_cost_fields(problem_setup.cost_fields, cost_field_weights)

    # This assumes that the harness router is deterministic and
    # will always return the same harness for the same inputs.
    # FIXME: investigate whether this is true.
    return route_harnesses(
        ips=ips,
        harness_setup=problem_setup.harness_setup,
        cost_field=combined_cf,
        bundling_weight=bundling_weight,
        harness_id=case_id,
        solutions_to_capture=[ips_solution_idx],
        smooth_solutions=smooth_solution,
        build_discrete_solutions=build_discrete_solution,
        build_presmooth_solutions=build_presmooth_solution,
        build_smooth_solutions=build_smooth_solution,
        build_cable_simulations=build_cable_simulation,
    )[
        0
    ]  # We only capture one solution


def design_point_ds(
    ips: IPSInstance,
    problem_setup: ProblemSetup,
    meta: OptimizationMeta,
    x: np.ndarray,
    iter_in_batch: int,
) -> xr.Dataset:
    """
    Evaluates a design point and returns a dataset with the results.
    """
    case_id = f"{meta.batch}.{iter_in_batch}"
    cost_field_ids = [cf.name for cf in problem_setup.cost_fields]

    combined_cost_field = combine_cost_fields(
        cost_fields=problem_setup.cost_fields, weights=x[:-1], normalize_fields=True
    )

    harness_solutions = route_harnesses(
        ips=ips,
        harness_setup=problem_setup.harness_setup,
        cost_field=combined_cost_field,
        bundling_weight=x[-1],
        harness_id=case_id,
        solutions_to_capture=[],
        smooth_solutions=True,
        build_discrete_solutions=False,
        build_presmooth_solutions=False,
        build_smooth_solutions=False,
        build_cable_simulations=True,
    )

    num_ips_solutions = len(harness_solutions)

    bundle_total_costs = np.array(
        [
            [
                evaluate_harness(harness, cost_field)
                for cost_field in problem_setup.cost_fields
            ]
            for harness in harness_solutions
        ]
    )
    bundle_costs = bundle_total_costs[:, :, 0]
    total_costs = bundle_total_costs[:, :, 1]

    ds = xr.Dataset(
        {
            "meta.timestamp": xr.DataArray(
                np.tile(pd.Timestamp.utcnow(), num_ips_solutions),
                dims=["solution"],
            ),
            "meta.category": xr.DataArray(
                np.tile(meta.category, num_ips_solutions),
                dims=["solution"],
            ),
            "meta.batch_idx": xr.DataArray(
                np.tile(meta.batch, num_ips_solutions),
                dims=["solution"],
            ),
            "meta.iter_idx": xr.DataArray(
                np.tile(iter_in_batch, num_ips_solutions),
                dims=["solution"],
            ),
            "meta.ips_idx": xr.DataArray(
                range(num_ips_solutions),
                dims=["solution"],
            ),
            "cost_field_weight": xr.DataArray(
                np.tile(x[:-1], (num_ips_solutions, 1)),
                dims=["solution", "cost_field"],
            ),
            "bundling_weight": xr.DataArray(
                np.tile(x[-1], num_ips_solutions),
                dims=["solution"],
            ),
            "bundling_cost": xr.DataArray(
                bundle_costs,
                dims=["solution", "cost_field"],
            ),
            "total_cost": xr.DataArray(
                total_costs,
                dims=["solution", "cost_field"],
            ),
            "num_estimated_clips": xr.DataArray(
                [h.numb_of_clips for h in harness_solutions],
                dims=["solution"],
            ),
            "harness_volume": xr.DataArray(
                [harness_volume(h) for h in harness_solutions],
                dims=["solution"],
            ),
            # FIXME: this doesn't work with zarr
            "harness": xr.DataArray(
                np.array(harness_solutions, dtype=object),
                dims=["solution"],
            ),
        }
    )
    ds = ds.assign_coords(
        solution=[h.name for h in harness_solutions],
        cost_field=cost_field_ids,
    )
    return ds


def batch_voi(
    batch_ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the dataset for a full batch and returns the variables of
    interest (xs, objs, cons)
    """
    objs = batch_ds[
        ["bundling_cost", "total_cost", "num_estimated_clips"]
    ].to_stacked_array(new_dim="obj", sample_dims=["solution"], name="objectives")
    cons = np.empty((objs.shape[0], 0), dtype=float)

    # Index the xs by the solution from obj, so we get the corresponding
    # x for each solution
    xs = (
        batch_ds[["cost_field_weight", "bundling_weight"]]
        .to_stacked_array(
            new_dim="desvar", sample_dims=["solution"], name="design_variables"
        )
        .sel(solution=objs.solution)
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
    bundling_weight_bounds = np.array([[0.05, 0.9]])
    bounds = np.array([*weights_bounds, *bundling_weight_bounds])

    batch_datasets = []

    def batch_analysis_func(xs: np.ndarray, meta: OptimizationMeta):
        this_batch_dataset = xr.concat(
            (
                design_point_ds(
                    ips=ips_instance,
                    problem_setup=problem_setup,
                    meta=meta,
                    x=x,
                    iter_in_batch=i,
                )
                for i, x in enumerate(xs)
            ),
            dim="solution",
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

    dataset = xr.concat(problem.state["batch_datasets"], dim="solution")
    dataset.attrs["problem_setup"] = problem_setup
    dataset.attrs["init_samples"] = init_samples
    dataset.attrs["batches"] = batches
    dataset.attrs["batch_size"] = batch_size
    dataset.attrs["ips_version"] = ips_instance.version

    return dataset
