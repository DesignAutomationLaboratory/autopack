from autopack.data_model import Cable, Geometry, ProblemSetup, HarnessSetup, CostField
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import load_scene, create_costfield
import numpy as np
import os
import pathlib


def test_harness_optimization_setup():
    cable1 = Cable(start_node="Cable1_start", end_node="Cable1_end", cable_type="ID_Rubber_template")
    cable2 = Cable(start_node="Cable2_start", end_node="Cable2_end", cable_type="ID_Rubber_template")
    cable3 = Cable(start_node="Cable3_start", end_node="Cable3_end", cable_type="ID_Rubber_template")
    part1 = Geometry(name = "part1", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part2 = Geometry(name = "part2", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part3 = Geometry(name = "part3", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    setup1 = HarnessSetup(geometries = [part1, part2, part3], cables = [cable1, cable2, cable3])
    
    cost_field = CostField(name="test", coordinates=np.ones((1,1,1,3),dtype=float), costs=np.ones((1,1,1),dtype=float))
    
    opt_setup = ProblemSetup(harness_setup=setup1, cost_fields=[cost_field])

def test_cost_field_creation():
    #ips_path = os.environ.get("AUTOPACK_IPS_PATH")
    #ips = IPSInstance(ips_path)
    ips = IPSInstance(r"C:\Users\antwi87\Documents\IPS\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64")
    ips.start()
    scene_path = pathlib.Path(__file__).parent / "scenes" / "simple_plate.ips"
    load_scene(ips, str(scene_path.resolve()))

    cable1 = Cable(start_node="Cable1_start", end_node="Cable1_end", cable_type="ID_Rubber_template")
    cable2 = Cable(start_node="Cable2_start", end_node="Cable2_end", cable_type="ID_Rubber_template")
    cable3 = Cable(start_node="Cable3_start", end_node="Cable3_end", cable_type="ID_Rubber_template")
    part1 = Geometry(name = "part1", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part2 = Geometry(name = "part2", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part3 = Geometry(name = "part3", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    setup1 = HarnessSetup(geometries = [part1, part2, part3], cables = [cable1, cable2, cable3])

    cost_field_ips, cost_field_length = create_costfield(ips, setup1)
    opt_setup = ProblemSetup(harness_setup=setup1, cost_fields=[cost_field_ips, cost_field_length])

    #bundle_cost,total_cost = problem_setup.harness_optimization(ips, opt_setup, [0.5, 0.5], 0.3)

