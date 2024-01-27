from typing import Optional

import numpy as np

from autopack.data_model import (
    CostField,
    Harness,
    HarnessSegment,
    HarnessSetup,
    IPSInstance,
)

from ..utils import grid_idxs_to_coords


def route_harnesses(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    cost_field: CostField,
    bundling_weight: float,
    harness_id: str,
) -> list[Harness]:
    assert not np.isnan(cost_field.costs).any(), "Cost field contains NaNs"
    # IPS seems to go haywire when given negative costs (as in suddenly
    # chewing up all memory)
    assert not np.any(cost_field.costs < 0), "Cost field contains negative values"

    response = ips.call(
        "autopack.routeHarnesses",
        harness_setup,
        cost_field.costs,
        bundling_weight,
        harness_id,
    )

    def gen_harness_segments(segment_dict):
        for segment in segment_dict:
            discrete_nodes = np.array(segment["discreteNodes"])
            discrete_coords = grid_idxs_to_coords(
                grid_coords=cost_field.coordinates, grid_idxs=discrete_nodes
            )

            yield HarnessSegment(
                radius=segment["radius"],
                cables=segment["cables"],
                discrete_nodes=discrete_nodes,
                discrete_coords=discrete_coords,
                # Make sure that these are always arrays with the last dim=3, even if empty
                presmooth_coords=np.array(segment["presmoothCoords"]).reshape(-1, 3),
                smooth_coords=np.array(segment["smoothCoords"]).reshape(-1, 3),
                clip_coords=np.array(segment["clipPositions"]).reshape(-1, 3),
            )

    solutions = [
        Harness(
            name=solution["name"],
            topology_feasible=solution["topologyFeasible"],
            harness_segments=list(gen_harness_segments(solution["segments"])),
            cable_segment_order=solution["cableSegmentOrder"],
            num_branch_points=solution["numBranchPoints"],
            bundling_weight=solution["objectiveWeightBundling"],
            bundling_objective=solution["solutionObjectiveBundling"],
            length_objective=solution["solutionObjectiveLength"],
            length_total=solution["lengthTotal"],
            length_in_collision=solution["lengthInCollision"],
        )
        for solution in response
    ]
    return solutions
