import pathlib
from itertools import product

import numpy as np

from autopack.data_model import CostField, Harness, HarnessSegment

from . import lua_commands


def create_costfield(ips_instance, harness_setup):
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.setup_export_cost_field()
    command = command1 + command2
    str_cost_field = ips_instance.call(command)
    str_cost_field = str_cost_field.decode("utf-8").strip('"')
    str_cost_field = str_cost_field[:-2]
    array_cost_field = str_cost_field.split()
    cost_field_size = (
        int(array_cost_field[0]),
        int(array_cost_field[1]),
        int(array_cost_field[2]),
    )
    array_cost_field = array_cost_field[3:]
    coordinates = [value for i, value in enumerate(array_cost_field) if (i % 4 != 3)]
    coordinates_grouped = [
        coordinates[i : i + 3] for i in range(0, len(coordinates), 3)
    ]
    costs = [value for i, value in enumerate(array_cost_field) if (i % 4 == 3)]

    np_costs = np.empty(cost_field_size, dtype=float)
    np_coords = np.empty(
        (cost_field_size[0], cost_field_size[1], cost_field_size[2], 3), dtype=float
    )
    n = 0
    for x, y, z in product(
        range(cost_field_size[0]), range(cost_field_size[1]), range(cost_field_size[2])
    ):
        cost_str = costs[n]
        np_costs[x, y, z] = float(cost_str)
        np_coords[x, y, z] = [
            float(coordinates_grouped[n][0]),
            float(coordinates_grouped[n][1]),
            float(coordinates_grouped[n][2]),
        ]
        n += 1
    ips_field = CostField(name="ips", coordinates=np_coords, costs=np_costs)
    length_field = CostField(
        name="length",
        coordinates=np_coords,
        costs=np.ones(cost_field_size, dtype=float),
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


def ips_optimize_harness(
    ips_instance, harness_setup, cost_field, bundling_factor=0.5, harness_id=None
):
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.setup_harness_optimization(
        cost_field, bundling_factor=bundling_factor, harness_id=harness_id
    )
    command = command1 + command2
    # print(command)
    str_harness = ips_instance.call(command)
    str_harness = str_harness.decode("utf-8").strip('"')
    array_harness = str_harness.split(",")
    array_harness[-1] = array_harness[-1].rstrip('"\n')
    nmb_of_clips = int(array_harness[0])
    array_harness = array_harness[1:]
    nmb_of_segments = array_harness.count("break")
    last_break = 0
    harness_segments = []
    for i in range(nmb_of_segments):
        cables = array_harness[
            last_break + 3 : last_break + 3 + int(array_harness[last_break + 2])
        ]
        cables_int = [int(x) for x in cables]
        start_loop = last_break + 3 + int(array_harness[last_break + 2])
        end_loop = start_loop + int(array_harness[last_break + 1]) * 3
        points = []
        for ii in range(start_loop, end_loop, 3):
            local_points = (
                int(array_harness[ii]),
                int(array_harness[ii + 1]),
                int(array_harness[ii + 2]),
            )
            points.append(local_points)
        last_break = end_loop
        harness_segments.append(HarnessSegment(cables=cables_int, points=points))

    return Harness(harness_segments=harness_segments, numb_of_clips=nmb_of_clips)


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
