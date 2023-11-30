import numpy as np
from smt.surrogate_models import KRG

from autopack import logger
from autopack.data_model import Cable, CostField, Geometry, HarnessSetup, ProblemSetup
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import (
    check_distance_of_points,
    ergonomic_evaluation,
    load_scene,
)


def create_ergonomic_cost_field(
    ips: IPSInstance,
    problem_setup: ProblemSetup,
    max_geometry_dist=0.2,
    min_point_dist=0.1,
    max_grip_diff=0.1,
):
    logger.info("Creating ergonomy cost fields")
    ref_cost_field = problem_setup.cost_fields[0]
    ref_costs = ref_cost_field.costs
    ref_coords = ref_cost_field.coordinates
    ref_coords_flat = ref_coords.reshape(-1, 3)

    geometries_to_consider = [
        geo.name for geo in problem_setup.harness_setup.geometries if geo.assembly
    ]
    sparse_points = sparse_cost_field(ref_cost_field, min_point_dist)
    points_close_to_surface = check_distance_of_points(
        ips, problem_setup.harness_setup, sparse_points, max_geometry_dist
    )
    eval_coords = sparse_points[points_close_to_surface]
    logger.info(f"Evaluating {eval_coords.shape[0]} points for ergonomy")
    ergo_eval = ergonomic_evaluation(ips, geometries_to_consider, eval_coords)
    ergo_standards = ergo_eval["ergoStandards"]
    ergo_values = np.array(ergo_eval["ergoValues"])
    assert ergo_values.shape == (len(eval_coords), len(ergo_standards))
    grip_distances = np.array(ergo_eval["gripDiffs"])
    bad_grip_mask = grip_distances > max_grip_diff
    logger.info(
        f"{bad_grip_mask.sum()} out of {eval_coords.shape[0]} points are unreachable"
    )
    ergo_values[bad_grip_mask] = 10
    # predicted_grip_diffs = interpolation(eval_coords, grip_distances, ref_coords_flat)
    # infeasible_mask = predicted_grip_diffs > max_grip_diff

    cost_fields = []
    for ergo_std_idx, ergo_std in enumerate(ergo_standards):
        true_costs = ergo_values[:, ergo_std_idx]

        logger.info(f"Interpolating {ergo_std} cost field")
        predicted_costs = interpolation(eval_coords, true_costs, ref_coords_flat)
        # predicted_costs[infeasible_mask] = np.inf

        cost_fields.append(
            CostField(
                name=ergo_std,
                coordinates=ref_coords,
                costs=predicted_costs.reshape(ref_costs.shape),
            )
        )

    return cost_fields


def sparse_cost_field(cost_field, min_point_dist):
    p1 = cost_field.coordinates[0, 0, 0]
    p2 = cost_field.coordinates[0, 0, 1]
    current_dist = (
        (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
    ) ** 0.5
    sample_dist = max(round(min_point_dist / current_dist), 1)
    new_arr = cost_field.coordinates[::sample_dist, ::sample_dist, ::sample_dist]
    reshaped_array = new_arr.reshape(-1, 3)
    return reshaped_array


def interpolation(known_x, known_y, predict_x):
    # Defining Kriging model
    sm = KRG(theta0=[1e-2] * 3, print_global=False)
    sm.set_training_values(known_x, known_y)
    sm.train()
    predict_y = sm.predict_values(predict_x)

    return predict_y
