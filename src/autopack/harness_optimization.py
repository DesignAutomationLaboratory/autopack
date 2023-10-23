import numpy as np

from autopack.data_model import CostField
from autopack.ips_communication.ips_commands import ips_optimize_harness


def optimize_harness(
    ips_instance,
    problem_setup,
    cost_field_weights,
    bundling_factor,
    harness_id=None,
):
    new_field = combine_cost_fields(
        problem_setup.cost_fields, cost_field_weights, normalize_fields=True
    )
    new_harness = ips_optimize_harness(
        ips_instance,
        problem_setup.harness_setup,
        new_field,
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
