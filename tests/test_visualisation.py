from autopack.ips_communication.ips_commands import cost_field_vis
from autopack.workflows import build_problem


def test_cost_field_vis(ips, simple_plate_harness_setup):
    problem_setup = build_problem(
        ips=ips,
        harness_setup=simple_plate_harness_setup,
    )
    cost_field = problem_setup.cost_fields[0]
    # Smoke test, just check that it doesn't crash
    cost_field_vis(ips=ips, cost_field=cost_field, visible=True)
