from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import load_scene
from autopack.data_model import Cable, Geometry, ProblemSetup, HarnessSetup
from autopack.ips_communication.ips_commands import create_costfield, optimize_harness

import os
import pathlib

def test_integration():
    ips_path = os.environ.get("AUTOPACK_IPS_PATH")
    ips = IPSInstance(ips_path)
    ips.start()
    scene_path = pathlib.Path(__file__).parent / "scenes" / "simple_plate.ips"
    load_scene(ips, str(scene_path.resolve()))


    cable1 = harness_setup.Cable(start_node="Cable1_start", end_node="Cable1_end", cable_type="ID_Rubber_template")
    cable2 = harness_setup.Cable(start_node="Cable2_start", end_node="Cable2_end", cable_type="ID_Rubber_template")
    cable3 = harness_setup.Cable(start_node="Cable3_start", end_node="Cable3_end", cable_type="ID_Rubber_template")
    part1 = harness_setup.Geometry(name = "part1", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part2 = harness_setup.Geometry(name = "part2", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part3 = harness_setup.Geometry(name = "part3", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    setup1 = harness_setup.HarnessSetup(geometries = [part1, part2, part2], cables = [cable1, cable2, cable3])

    cost_field_template, cost_field_ips, cost_field_constant = create_costfield(ips, setup1)

    opt_setup = problem_setup.ProblemSetup()
    opt_setup.harness_setup = setup1
    opt_setup.cost_fields = [cost_field_ips, cost_field_constant]

    bundle_cost,total_cost = problem_setup.harness_optimization(ips, opt_setup, [0.5, 0.5], 0.3)

    print("bundle_cost: ", bundle_cost)
    print("total_cost: ", total_cost)