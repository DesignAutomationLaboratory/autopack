from autopack.data_model import ProblemSetup
from autopack.ergonomic_evaluation import create_ergonomic_cost_field
from autopack.ips_communication.ips_commands import (
    cost_field_vis,
    create_costfield,
    load_scene,
)


def create_default_prob_setup(ips_instance, harness_setup, create_imma=False):
    load_scene(ips_instance, harness_setup.scene_path, clear=True)
    cost_field_ips, cost_field_length = create_costfield(ips_instance, harness_setup)
    opt_setup = ProblemSetup(
        harness_setup=harness_setup, cost_fields=[cost_field_ips, cost_field_length]
    )
    if create_imma:
        rula_cost_field, reba_cost_field = create_ergonomic_cost_field(
            ips=ips_instance,
            harness_setup=harness_setup,
            ref_cost_field=cost_field_ips,
            use_rbpp=False,
            update_screen=False,
        )
        opt_setup.cost_fields.append(rula_cost_field)
        opt_setup.cost_fields.append(reba_cost_field)

    for cost_field in opt_setup.cost_fields:
        cost_field_vis(ips=ips_instance, cost_field=cost_field, visible=False)

    return opt_setup
