import pathlib
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
from . import lua_commands


def create_costfield(ips_instance, harness_setup):
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.setup_export_cost_field()
    command = command1 + command2
    response = ips_instance.call_unpack(command)

    coords = np.array(response["coords"])
    costs = np.array(response["costs"])

    ips_field = CostField(name="ips", coordinates=coords, costs=costs)
    length_field = CostField(
        name="length",
        coordinates=coords,
        costs=np.ones_like(costs),
    )
    return ips_field, length_field


def load_scene(ips_instance, scene_file_path):
    escaped_string = scene_file_path.encode(
        "unicode_escape"
    ).decode()  # .encode('unicode_escape').decode()
    command = f"""
    local IPSFile = "{escaped_string}"
    Ips.loadScene(IPSFile)
    """

    ips_instance.call(command)


def route_harness_all_solutions(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    cost_field: CostField,
    bundling_factor: float,
    harness_id: str,
    solutions_to_capture: Optional[list[str]] = None,
    smooth_solutions: bool = False,
    build_discrete_solutions: bool = False,
    build_presmooth_solutions: bool = False,
    build_smooth_solutions: bool = False,
) -> list[Harness]:
    assert not np.isnan(cost_field.costs).any(), "Cost field contains NaNs"
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.route_harness(
        cost_field=cost_field,
        bundling_factor=bundling_factor,
        case_id=harness_id,
        solutions_to_capture=solutions_to_capture,
        smooth_solutions=smooth_solutions,
        build_discrete_solutions=build_discrete_solutions,
        build_presmooth_solutions=build_presmooth_solutions,
        build_smooth_solutions=build_smooth_solutions,
    )
    command = command1 + command2

    response = ips.call_unpack(command)

    def gen_harness_segments(segment_dict):
        for segment in segment_dict:
            discrete_nodes = np.array(segment["discreteNodes"])
            discrete_coords = grid_idxs_to_coords(
                grid_coords=cost_field.coordinates, grid_idxs=discrete_nodes
            )
            _smooth_coords = segment.get("smoothCoords", None)
            smooth_coords = (
                np.array(_smooth_coords) if _smooth_coords is not None else None
            )
            _clip_coords = segment.get("clipPositions", None)
            clip_coords = np.array(_clip_coords) if _clip_coords is not None else None

            yield HarnessSegment(
                radius=segment["radius"],
                cables=segment["cables"],
                discrete_nodes=discrete_nodes,
                discrete_coords=discrete_coords,
                presmooth_coords=np.array(segment["presmoothCoords"]),
                smooth_coords=smooth_coords,
                clip_coords=clip_coords,
            )

    solutions = [
        Harness(
            name=solution["name"],
            harness_segments=list(gen_harness_segments(solution["segments"])),
            numb_of_clips=solution["estimatedNumClips"],
            num_branch_points=solution["numBranchPoints"],
            bundling_factor=solution["objectiveWeightBundling"],
            bundling_objective=solution["solutionObjectiveBundling"],
            length_objective=solution["solutionObjectiveLength"],
        )
        for solution in response
    ]
    return solutions


def route_harness(
    *args,
    **kwargs,
) -> Harness:
    solutions = route_harness_all_solutions(*args, **kwargs, solutions_to_capture=[0])
    return solutions[0]


def check_distance_of_points(ips_instance, harness_setup, coords, max_geometry_dist):
    command = lua_commands.check_coord_distances(
        max_geometry_dist, harness_setup, coords
    )
    # with open(r"C:\Users\antwi87\Documents\IPS\test_environment\filename.txt", "w") as file:
    #    file.write(command)
    str_checked = ips_instance.call(command)
    numbers = [int(num) for num in str_checked.decode("utf-8").strip(' "\n').split()]
    return numbers


def ergonomic_evaluation(ips_instance, parts, coords):
    # ips_instance.start()
    ergo_path = pathlib.Path(__file__).parent / "ErgonomicEvaluation.ips"
    load_scene(ips_instance, str(ergo_path.resolve()))
    import time

    time.sleep(1)
    command = lua_commands.ergonomic_evaluation(parts, coords)
    results = ips_instance.call(command)
    result_array = results.decode("utf-8").strip().replace('"', "").split()
    output = []
    for i in range(0, len(result_array) - 1, 2):
        output.append([float(result_array[i]), float(result_array[i + 1])])
    return output


def cost_field_vis(ips_instance, cost_field):
    command = lua_commands.add_cost_field_vis(cost_field)
    with open(
        r"C:\Users\antwi87\Documents\IPS\test_environment\filename.lua", "w"
    ) as file:
        file.write(command)
    ips_instance.call(command)


def get_stl_meshes(ips_instance):
    command = lua_commands.get_stl_meshes()
    print(command)
    str_meshes = ips_instance.call(command)
    print(str_meshes)
