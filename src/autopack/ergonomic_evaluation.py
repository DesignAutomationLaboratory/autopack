import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist

from autopack import logger
from autopack.data_model import Cable, CostField, Geometry, HarnessSetup, ProblemSetup
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import check_distance_of_points
from autopack.utils import farthest_point_sampling


def create_ergonomic_cost_field(
    ips: IPSInstance,
    problem_setup: ProblemSetup,
    min_geometry_dist=0.001,
    max_geometry_dist=0.25,
    max_grip_diff=0.2,  # m
    min_point_dist=0.1,  # m
    max_samples=2000,
    use_rbpp=True,
    update_screen=False,
    keep_generated_objects=False,
):
    logger.info("Creating ergonomy cost fields")
    ref_cost_field = problem_setup.cost_fields[0]
    ref_costs = ref_cost_field.costs
    ref_coords = ref_cost_field.coordinates
    ref_coords_flat = ref_coords.reshape(-1, 3)
    # No point in evaluating infeasible points
    feasible_mask = np.invert(np.isposinf(ref_costs)).reshape(-1)

    families: list[dict[str, str]] = ips.call("autopack.getAllManikinFamilies")
    logger.info(f"Found {len(families)} manikin families")
    if not families:
        logger.warning(
            "No manikin families found in scene, ergonomy cost fields will be empty"
        )
        return []

    geometries_to_consider = [
        geo.name for geo in problem_setup.harness_setup.geometries if geo.assembly
    ]
    coords_to_distance_check = ref_coords_flat[feasible_mask]
    logger.info(
        f"Checking {coords_to_distance_check.shape[0]} points for distance to geometry"
    )
    distance_ok_mask = check_distance_of_points(
        ips,
        problem_setup.harness_setup,
        coords_to_distance_check,
        min_geometry_dist,
        max_geometry_dist,
    )
    coords_with_ok_distance = coords_to_distance_check[distance_ok_mask]
    logger.info(
        f"Picking at most {max_samples} points spaced {min_point_dist} meters apart, out of {coords_with_ok_distance.shape[0]}, using farthest point sampling"
    )
    eval_coords = farthest_point_sampling(
        points=coords_with_ok_distance,
        num_points=max_samples,
        min_farthest_distance=min_point_dist,
        seed=0,  # For deterministic behavior
    )

    _all_ergo_values = []
    for family in families:
        family_id = family["id"]
        family_name = family["name"]
        logger.info(f"Evaluating {eval_coords.shape[0]} points with {family_name}")
        ergo_eval = ips.call(
            "autopack.evalErgo",
            geometries_to_consider,
            family_id,
            eval_coords,
            use_rbpp,
            update_screen,
            keep_generated_objects,
        )
        ergo_standards = ergo_eval["ergoStandards"]
        ergo_values = np.array(ergo_eval["ergoValues"], dtype=float).max(axis=2)
        assert ergo_values.shape == (len(eval_coords), len(ergo_standards))
        grip_distances = np.array(ergo_eval["gripDiffs"])
        bad_grip_mask = grip_distances > max_grip_diff
        logger.notice(
            f"{bad_grip_mask.sum()} out of {eval_coords.shape[0]} points are unreachable for {family_name}"
        )
        ergo_values[bad_grip_mask] = np.inf
        _all_ergo_values.append(ergo_values)

    all_ergo_values = np.stack(_all_ergo_values)
    assert all_ergo_values.shape == (
        len(families),
        len(eval_coords),
        len(ergo_standards),
    )

    combined_ergo_values = all_ergo_values.min(axis=0)
    combined_feasible_mask = np.isfinite(combined_ergo_values)
    # Set the cost of infeasible points to a relatively high value
    combined_ergo_values[np.invert(combined_feasible_mask)] = 10

    cost_fields = []
    for ergo_std_idx, ergo_std in enumerate(ergo_standards):
        true_costs = combined_ergo_values[:, ergo_std_idx]

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
    rbf = RBFInterpolator(known_x, known_y)
    predict_y = rbf(predict_x)

    return predict_y
