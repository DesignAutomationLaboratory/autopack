from . import lua_commands
import cost_field

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

def optimize_harness(ips_instance, harness_setup, cost_field):
    command1 = lua_commands.setup_harness_routing(harness_setup)
    command2 = lua_commands.setup_harness_optimization(cost_field)
    command = command1 + command2

    str_harness = ips_instance.call(command)
    str_harness = str_harness.decode('utf-8').strip('"')
    array_harness = str_harness.split(",")
    array_harness[-1] = array_harness[-1].rstrip('"\n')
    nmb_of_clips = int(array_harness[0])
    nmb_of_paths = array_harness.count('break')
    print(array_harness)
