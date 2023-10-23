from autopack.data_model import Cable, Geometry, ProblemSetup, HarnessSetup, CostField
from autopack.ips_communication.ips_commands import load_scene, create_costfield
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ergonomic_evaluation import create_ergonomic_cost_field

def create_default_prob_setup(ips_instance, harness_setup, create_imma = False):
    load_scene(ips_instance, harness_setup.)
    cost_field_ips, cost_field_length = create_costfield(ips, setup1)
    opt_setup = ProblemSetup(harness_setup=setup1, cost_fields=[cost_field_ips, cost_field_length])

    rula_cost_field, reba_cost_field = create_ergonomic_cost_field(ips, opt_setup, max_geometry_dist=0.2, min_point_dist=0.3)
    opt_setup.cost_fields.append(rula_cost_field)
    opt_setup.cost_fields.append(reba_cost_field)
