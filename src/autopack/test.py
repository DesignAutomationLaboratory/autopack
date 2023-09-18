from ips_communication.ips_class import IPSInstance
from ips_communication.ips_environment import load_scene
#from ips_communication.ips_costfield import create_ips_field
#from ips_communication.ips_optimization import optimize_harness

ips = IPSInstance("C:\\Users\\antwi87\\Documents\\IPS\\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64\\")
ips.start()
load_scene(ips,r"C:\Users\antwi87\Documents\IPS\test_environment\test_environment.ips")

import harness_setup
cable1 = harness_setup.Cable(start_node="Cable1_start", end_node="Cable1_end", cable_type="ID_Rubber_template")
cable2 = harness_setup.Cable(start_node="Cable2_start", end_node="Cable2_end", cable_type="ID_Rubber_template")
cable3 = harness_setup.Cable(start_node="Cable3_start", end_node="Cable3_end", cable_type="ID_Rubber_template")
part1 = harness_setup.Geometry(name = "part1", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
part2 = harness_setup.Geometry(name = "part2", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
part3 = harness_setup.Geometry(name = "part3", clearance = 5.0, preference = 'Near', clipable = True, assembly = True)
setup1 = harness_setup.HarnessSetup(geometries = [part1, part2, part2], cables = [cable1, cable2, cable3])


from ips_communication.ips_commands import create_costfield, optimize_harness

cost_field_template, cost_field_ips, cost_field_constant = create_costfield(ips, setup1)
new_harness = optimize_harness(ips, setup1, cost_field_ips)
print(new_harness)