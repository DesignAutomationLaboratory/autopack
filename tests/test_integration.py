import os
import pathlib

import numpy as np

from autopack.data_model import Cable, CostField, Geometry, HarnessSetup, ProblemSetup
from autopack.harness_optimization import optimize_harness
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import create_costfield, load_scene


def test_harness_optimization_setup():
    cable1 = Cable(
        start_node="Cable1_start",
        end_node="Cable1_end",
        cable_type="ID_Rubber_template",
    )
    cable2 = Cable(
        start_node="Cable2_start",
        end_node="Cable2_end",
        cable_type="ID_Rubber_template",
    )
    cable3 = Cable(
        start_node="Cable3_start",
        end_node="Cable3_end",
        cable_type="ID_Rubber_template",
    )
    part1 = Geometry(
        name="part1", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    part2 = Geometry(
        name="part2", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    part3 = Geometry(
        name="part3", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    setup1 = HarnessSetup(
        scene_path="", geometries=[part1, part2, part3], cables=[cable1, cable2, cable3]
    )

    cost_field = CostField(
        name="test",
        coordinates=np.ones((1, 1, 1, 3), dtype=float),
        costs=np.ones((1, 1, 1), dtype=float),
    )

    opt_setup = ProblemSetup(harness_setup=setup1, cost_fields=[cost_field])


def test_integration():
    # ips_path = os.environ.get("AUTOPACK_IPS_PATH")
    # ips = IPSInstance(ips_path)
    ips = IPSInstance(
        r"C:\Users\antwi87\Documents\IPS\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64"
    )
    ips.start()
    scene_path = str(
        (pathlib.Path(__file__).parent / "scenes" / "simple_plate.ips").resolve()
    )

    cable1 = Cable(
        start_node="Cable1_start",
        end_node="Cable1_end",
        cable_type="ID_Rubber_template",
    )
    cable2 = Cable(
        start_node="Cable2_start",
        end_node="Cable2_end",
        cable_type="ID_Rubber_template",
    )
    cable3 = Cable(
        start_node="Cable3_start",
        end_node="Cable3_end",
        cable_type="ID_Rubber_template",
    )
    part1 = Geometry(
        name="part1", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    part2 = Geometry(
        name="part2", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    part3 = Geometry(
        name="part3", clearance=5.0, preference="Near", clipable=True, assembly=True
    )
    setup1 = HarnessSetup(
        scene_path=scene_path,
        geometries=[part1, part2, part3],
        cables=[cable1, cable2, cable3],
    )

    load_scene(ips, setup1.scene_path)
    cost_field_ips, cost_field_length = create_costfield(ips, setup1)
    opt_setup = ProblemSetup(
        harness_setup=setup1, cost_fields=[cost_field_ips, cost_field_length]
    )

    costs, numb_of_clips = optimize_harness(
        ips, opt_setup, [0.5, 0.5], 0.5, save_harness=True, id=0
    )
    print("costs: ", costs)
    print("Number of clips: ", numb_of_clips)
