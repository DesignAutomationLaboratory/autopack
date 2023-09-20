from . import lua_commands
import cost_field
import harness

def create_costfield(ips_instance, harness_setup):
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.setup_export_cost_field()
    command = command1 + command2
    str_cost_field = ips_instance.call(command)
    str_cost_field = str_cost_field.decode('utf-8').strip('"')
    array_cost_field = str_cost_field.split()
    cost_field_size = [int(array_cost_field[0]), int(array_cost_field[1]), int(array_cost_field[2])]
    array_cost_field = array_cost_field[3:]
    cost_field_template = cost_field.CostFieldTemplate(size=cost_field_size)
    cost_field_ips = cost_field.CostField(cost_field_template)
    cost_field_constant = cost_field.CostField(cost_field_template)
    coordinates = [value for i, value in enumerate(array_cost_field) if (i % 4 != 3)]
    costs = [value for i, value in enumerate(array_cost_field) if (i % 4 == 3)]
    cost_field_template.set_coords_from_str_array(coordinates)
    cost_field_ips.set_costs_from_str_array(costs)
    cost_field_constant.costs += 1

    return cost_field_template, cost_field_ips, cost_field_constant

def load_scene(ips_instance, scene_file_path):
    escaped_string = scene_file_path.encode('unicode_escape').decode() #.encode('unicode_escape').decode()
    command ="""
    local IPSFile = '""" + escaped_string + """'
    Ips.loadScene(IPSFile)
    """
    ips_instance.call(command)

def optimize_harness(ips_instance, harness_setup, cost_field, bundle_weight=0.5, save_harness=True, id=0):
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.setup_harness_optimization(cost_field, weight=bundle_weight, save_harness=save_harness, harness_id=id)
    command = command1 + command2

    str_harness = ips_instance.call(command)
    str_harness = str_harness.decode('utf-8').strip('"')
    array_harness = str_harness.split(",")
    array_harness[-1] = array_harness[-1].rstrip('"\n')
    nmb_of_clips = int(array_harness[0])
    array_harness = array_harness[1:]
    nmb_of_segments = array_harness.count('break')
    new_harness = harness.harness()
    new_harness.numb_of_clips = nmb_of_clips
    last_break = 0
    for i in range(nmb_of_segments):
        new_segment = harness.harness_segment()
        cables = array_harness[last_break+3:last_break+3+int(array_harness[last_break+2])]
        cables_int = [int(x) for x in cables]
        new_segment.cables = cables_int
        start_loop = last_break+3+int(array_harness[last_break+2])
        end_loop = start_loop+int(array_harness[last_break+1])*3
        for ii in range(start_loop,end_loop,3):
            points = [int(array_harness[ii]),int(array_harness[ii+1]),int(array_harness[ii+2])]
            new_segment.points.append(points)
        last_break = end_loop
        new_harness.harness_segments.append(new_segment)
    return new_harness

def check_distance_of_points(ips_instance, harness_setup, coords):
    command = lua_commands.check_coord_distances(0.1, harness_setup, coords)
    str_checked = ips_instance.call(command)

def get_stl_meshes(ips_instance):
    command = lua_commands.get_stl_meshes()
    print(command)
    str_meshes = ips_instance.call(command)
    print(str_meshes)
