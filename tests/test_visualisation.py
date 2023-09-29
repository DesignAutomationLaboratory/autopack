from autopack.data_model import Cable, Geometry, ProblemSetup, HarnessSetup, CostField
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import load_scene, create_costfield, cost_field_vis
import numpy as np
import os
import pathlib

def test_cost_field_vis():
    ips = IPSInstance(r"C:\Users\antwi87\Documents\IPS\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64")
    ips.start()
    scene_path = str((pathlib.Path(__file__).parent / "scenes" / "simple_plate.ips").resolve())

    cable1 = Cable(start_node="Cable1_start", end_node="Cable1_end", cable_type="ID_Rubber_template")
    cable2 = Cable(start_node="Cable2_start", end_node="Cable2_end", cable_type="ID_Rubber_template")
    cable3 = Cable(start_node="Cable3_start", end_node="Cable3_end", cable_type="ID_Rubber_template")
    part1 = Geometry(name = "part1", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part2 = Geometry(name = "part2", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    part3 = Geometry(name = "part3", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
    setup1 = HarnessSetup(scene_path=scene_path, geometries = [part1, part2, part3], cables = [cable1, cable2, cable3])

    load_scene(ips, setup1.scene_path)
    cost_field_ips, cost_field_length = create_costfield(ips, setup1)
    cost_field_vis(ips, cost_field_ips)

