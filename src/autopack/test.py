from ips_communication.ips_class import IPSInstance
from ips_communication.ips_environment import load_scene
from ips_communication.ips_costfield import create_ips_field

ips = IPSInstance("C:\\Users\\antwi87\\Documents\\IPS\\IPS_2023-R2-SP1-HarnessRouter_v3.0_x64\\")
ips.start()
load_scene(ips,r"C:\Users\antwi87\Documents\IPS\22w02 IPSsessionautopack20XC90.ips")
#nodes = ips_communication.get_nodes(ips)
#geo = ips_communication.get_geometries(ips)
#print(nodes)
#print(geo)
nodes = [("Start 1","End 1",10),("Start 2","End 2",10)]
geo = [("Panels",150,0,True)]
cost_field_template, cost_field_ips, cost_field_constant = create_ips_field(ips,nodes,geo)