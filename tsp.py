import pandas as pd
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(filepath, from_col='from', to_col='to', weight_col='distance_km', return_locations=False):
    # load the distance table
    data = {}
    
    df_distance_table = pd.read_csv(filepath)

    # generate distance matrix (distance only, duration is excluded)
    df_distance_matrix = df_distance_table.pivot_table(
        index=from_col,
        columns=to_col,
        values=weight_col,
    )

    data['distance_matrix'] = np.rint(df_distance_matrix.values) # the computations are done with integers
    data['num_vehicles'] = 1
    data['depot'] = 0 # start and end: home / first location

    if return_locations:
        locations = df_distance_table['from'].unique().tolist()
        return data, locations
    
    return data

def distance_callback(from_index, to_index):
    # return the distance between two locations
    # convert from routing variable index to distance matrix nodeindex
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)

    return distance_data['distance_matrix'][from_node][to_node]

def print_solution(manager, routing, solution, locations):
    # prints solution
    print('Objective: {} km'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0

    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(locations[manager.IndexToNode(index)])
        prev_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(prev_index, index, 0)
    
    plan_output += ' {}\n'.format(locations[manager.IndexToNode(index)])
    plan_output += 'Route distance: {} km\n'.format(route_distance)
    print(plan_output)


distance_filepath = 'distance.csv'
distance_data, location_names  = create_data_model(distance_filepath, return_locations=True)

# create the routing model
manager = pywrapcp.RoutingIndexManager(
    len(distance_data['distance_matrix']),
    distance_data['num_vehicles'], distance_data['depot']
)
routing = pywrapcp.RoutingModel(manager)
distance_callback_index = routing.RegisterTransitCallback(distance_callback)

# set the cost of travel
routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

# set search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
# search_parameters.first_solution_strategy = (
#     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
# )
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
search_parameters.time_limit.seconds = 40
search_parameters.log_search = True

# add the solution printer
solution = routing.SolveWithParameters(search_parameters)
if solution:
    print_solution(manager, routing, solution, location_names)