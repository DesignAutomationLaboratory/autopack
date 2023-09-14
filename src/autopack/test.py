from ips_communication.ips_class import IPSInstance
from ips_communication.ips_environment import load_scene
from ips_communication.ips_costfield import create_ips_field
from ips_communication.ips_optimization import optimize_harness

ips = IPSInstance("C:\\Users\\antwi87\\Documents\\IPS\\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64\\")
ips.start()
load_scene(ips,r"C:\Users\antwi87\Documents\IPS\test_environment\test_environment.ips")
#nodes = ips_communication.get_nodes(ips)
#geo = ips_communication.get_geometries(ips)
#print(nodes)
#print(geo)
nodes = [("Cable1_start","Cable1_end",10),("Cable2_start","Cable2_end",10),("Cable3_start","Cable3_end",10)]
geo = [("part1",5,0,True),("part2",5,0,True),("part3",5,0,True)]
cost_field_template, cost_field_ips, cost_field_constant = create_ips_field(ips,nodes,geo)
optimize_harness(ips,nodes,geo,cost_field_ips)