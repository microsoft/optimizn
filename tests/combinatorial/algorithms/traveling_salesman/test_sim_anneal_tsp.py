# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pip install python-tsp
# https://github.com/fillipe-gsm/python-tsp
from python_tsp.heuristics import solve_tsp_simulated_annealing
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
    _, distance = solve_tsp_simulated_annealing(
        tt.dists,
        max_processing_time=60,
        alpha=0.99, x0=list(range(tt.num_cities)),
        perturbation_scheme='ps2')

    # run simulated annealing algorithm
    ts = TravSalsmn(tt)
    init_cost = ts.best_cost
    ts.anneal(n_iter=int(1e20), reset_p=0, time_limit=60)

    # check final solution optimality
    check_sol_vs_init_sol(ts.best_cost, init_cost)
    check_sol_optimality(ts.best_cost, distance, 1.25)
