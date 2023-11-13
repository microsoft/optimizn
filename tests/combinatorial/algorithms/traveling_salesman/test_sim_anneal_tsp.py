# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pip install python-tsp
# https://github.com/fillipe-gsm/python-tsp
from python_tsp.heuristics import solve_tsp_local_search
## https://developers.google.com/optimization/routing/tsp
# Their solution didn't work, some cpp error.
from optimizn.combinatorial.algorithms.traveling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.algorithms.traveling_salesman.sim_anneal_tsp\
    import TravSalsmn
from tests.combinatorial.algorithms.check_sol_utils import\
    check_sol_optimality, check_sol_vs_init_sol


def test_sa_tsp():
    # create graph
    tt = CityGraph()

    # run external library algorithm
    #permutation, distance = solve_tsp_dynamic_programming(tt.dists)
    _, distance = solve_tsp_local_search(tt.dists)

    # run simulated annealing algorithm
    ts1 = TravSalsmn(tt)
    init_cost = ts1.best_cost
    ts1.anneal()

    # check final solution optimality
    check_sol_vs_init_sol(ts1.best_cost, init_cost)
    check_sol_optimality(ts1.best_cost, distance, 1.25)
