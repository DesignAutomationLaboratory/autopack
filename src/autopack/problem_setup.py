from ips_communication.ips_commands import optimize_harness
import cost_field
import harness

class ProblemSetup():
    def __init__(self):
        self.harness_setup = None
        self.cost_fields = []

def harness_optimization(ips_instance, problem_setup, cost_field_weights, bundle_weight):
    new_field = cost_field.combine_cost_fields(problem_setup.cost_fields, cost_field_weights, normilise_fields = True)
    new_harness = optimize_harness(ips_instance, problem_setup.harness_setup, new_field, bundle_weight=bundle_weight)
    
    bundle_cost, total_cost = harness.evaluate_harness(new_harness, problem_setup.cost_fields[0])
    return bundle_cost,total_cost
    