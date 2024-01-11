from autopack.data_model import ProblemSetup
from autopack.ergonomic_evaluation import create_ergonomic_cost_field
from autopack.ips_communication.ips_commands import create_costfield, load_scene


def create_default_prob_setup(ips_instance, harness_setup, create_imma=False):
    load_scene(ips_instance, harness_setup.scene_path, clear=True)
    cost_field_ips, cost_field_length = create_costfield(ips_instance, harness_setup)
    opt_setup = ProblemSetup(
        harness_setup=harness_setup, cost_fields=[cost_field_ips, cost_field_length]
    )
    if create_imma:
        rula_cost_field, reba_cost_field = create_ergonomic_cost_field(
            ips=ips_instance,
            problem_setup=opt_setup,
            max_geometry_dist=0.2,
            min_point_dist=0.1,
            use_rbpp=False,
        )
        opt_setup.cost_fields.append(rula_cost_field)
        opt_setup.cost_fields.append(reba_cost_field)
    return opt_setup
