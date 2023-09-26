from autopack.data_model import Cable, Geometry, ProblemSetup, HarnessSetup, CostField
from autopack.ips_communication.ips_commands import check_distance_of_points, load_scene, ergonomic_evaluation
from autopack.ips_communication.ips_class import IPSInstance
from smt.surrogate_models import KRG
import numpy as np

def create_ergonomic_cost_field(ips, problem_setup, stl_paths, max_geometry_dist=0.2, min_point_dist=0.1):
    sparse_points = sparse_cost_field(problem_setup.cost_fields[0], min_point_dist)
    points_close_to_surface = check_distance_of_points(ips, problem_setup.harness_setup, sparse_points, max_geometry_dist)
    selected_coords = [num for i, num in enumerate(sparse_points) if points_close_to_surface[i] == 1]
    ergo_evaluations = ergonomic_evaluation(ips, stl_paths, selected_coords)
    array1 = np.array(selected_coords)
    REBA_vals = np.array(ergo_evaluations)[:, 0].reshape(-1, 1)
    RULA_vals = np.array(ergo_evaluations)[:, 1].reshape(-1, 1)
    REBA_res = np.hstack([array1, REBA_vals])
    RULA_res = np.hstack([array1, RULA_vals])
    wanted_coords = problem_setup.cost_fields[0].coordinates.reshape(-1, 3)
    new_reba_values = interpolation(REBA_res, wanted_coords)[:,-1]
    new_rula_values = interpolation(RULA_res, wanted_coords)[:,-1]
    cost_field_shape = problem_setup.cost_fields[0].costs.shape
    rula_costs = new_rula_values.reshape(cost_field_shape)
    reba_costs = new_reba_values.reshape(cost_field_shape)
    rula_cost_field = CostField(name="RULA",coordinates=problem_setup.cost_fields[0].coordinates, costs=rula_costs)
    reba_cost_field = CostField(name="REBA",coordinates=problem_setup.cost_fields[0].coordinates, costs=reba_costs)
    return rula_cost_field, reba_cost_field

def sparse_cost_field(cost_field, min_point_dist):
    p1 = cost_field.coordinates[0,0,0]
    p2 = cost_field.coordinates[0,0,1]
    current_dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)**0.5
    sample_dist = max(round(min_point_dist/current_dist), 1)
    new_arr = cost_field.coordinates[::sample_dist, ::sample_dist, ::sample_dist]
    reshaped_array = new_arr.reshape(-1, 3)
    return reshaped_array

def interpolation(known_costs, new_cost_pos):
    #known_costs = np array with known cost values size n*4 [[coord_x, coord_y, coord_z, cost],...]
    #new_cost_pos = np array with new pos coordinates size m*3 [[coord_x, coord_y, coord_z],...]
    # Splitting known_costs array into positions and costs
    pos = known_costs[:, :3]  # positions
    costs = known_costs[:, 3]  # costs
    # Defining Kriging model
    sm = KRG(theta0=[1e-2]*3, print_global=False)
    # Training the model
    sm.set_training_values(pos, costs)
    sm.train()
    # Predicting the cost for the new positions
    predicted_costs = sm.predict_values(new_cost_pos)
    # Appending the predicted costs to new_cost_pos
    new_cost_pos_with_costs = np.hstack((new_cost_pos, predicted_costs))
    return new_cost_pos_with_costs

