# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.traveling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.algorithms.traveling_salesman.sim_anneal_tsp\
    import TravSalsmn
from optimizn.combinatorial.algorithms.traveling_salesman.bnb_tsp\
    import TravelingSalesmanProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
from optimizn.combinatorial.opt_problem import load_latest_pckl
import time
import shutil
import os


def run_tsp_experiments(num_cities=50, compute_time_mins=1, num_trials=3):
    # specify maximum number of iterations
    MAX_ITERS = int(1e20)  # very high bound on iterations
    # since algorithms should use up all compute time

    # for collecting results
    results = dict()
    results['sa1'] = []
    results['sa1_time'] = []
    results['sa2'] = []
    results['sa2_time'] = []
    results['trad_bnb'] = []
    results['trad_bnb_time'] = []
    results['mod_bnb'] = []
    results['mod_bnb_time'] = []
    results['sa1_init_sol'] = None
    results['sa1_init_sol_cost'] = None
    results['sa2_init_sol'] = None
    results['sa2_init_sol_cost'] = None
    results['trad_bnb_init_sol'] = None
    results['trad_bnb_init_sol_cost'] = None
    results['mod_bnb_init_sol'] = None
    results['mod_bnb_init_sol_cost'] = None

    # create traveling salesman problem parameters
    city_graph = CityGraph(num_cities)

    # delete continuous training results from previous experiment runs
    if os.path.isdir('Data/'):
        shutil.rmtree('Data/')

    # run simulated annealing 1
    tsp_sa = TravSalsmn(city_graph)
    results['sa1_init_sol'] = tsp_sa.best_solution
    results['sa1_init_sol_cost'] = tsp_sa.best_cost
    s = time.time()
    tsp_sa.anneal(n_iter=MAX_ITERS, time_limit=compute_time_mins * 60)
    e = time.time()
    tsp_sa.persist()
    results['sa1'].append(tsp_sa.best_cost)
    results['sa1_time'].append(e - s)
    for _ in range(num_trials - 1):
        tsp_sa = load_latest_pckl(path1='Data/TravSalsmn/DailyOpt')
        if tsp_sa is None:
            raise Exception('No saved instance for TSP simulated annealing')
        s = time.time()
        tsp_sa.anneal(n_iter=MAX_ITERS, time_limit=compute_time_mins * 60)
        e = time.time()
        tsp_sa.persist()
        results['sa1'].append(tsp_sa.best_cost)
        results['sa1_time'].append(e - s)

    # run simulated annealing 2
    opt_permutation = list(results['sa1_init_sol'])
    opt_dist = results['sa1_init_sol_cost']
    results['sa2_init_sol'] = opt_permutation
    results['sa2_init_sol_cost'] = opt_dist
    s = time.time()
    e = time.time()
    while (e - s) < (compute_time_mins * 60):
        permutation, distance = solve_tsp_simulated_annealing(
            city_graph.dists,
            max_processing_time=(compute_time_mins * 60) - (e - s),
            alpha=0.99, x0=opt_permutation, perturbation_scheme='ps2')
        if opt_dist > distance:
            opt_dist = distance
            opt_permutation = permutation
        e = time.time()
    results['sa2'].append(opt_dist)
    results['sa2_time'].append(e - s)
    for _ in range(num_trials - 1):
        s = time.time()
        e = time.time()
        while (e - s) < (compute_time_mins * 60):
            permutation, distance = solve_tsp_simulated_annealing(
                city_graph.dists,
                max_processing_time=(compute_time_mins * 60) - (e - s),
                alpha=0.99, x0=opt_permutation, perturbation_scheme='ps2')
            if opt_dist > distance:
                opt_dist = distance
                opt_permutation = permutation
            e = time.time()
        results['sa2'].append(opt_dist)
        results['sa2_time'].append(e - s)

    # run modified branch and bound
    mod_tsp_bnb = TravelingSalesmanProblem({'input_graph': city_graph})
    results['mod_bnb_init_sol'] = mod_tsp_bnb.best_solution
    results['mod_bnb_init_sol_cost'] = mod_tsp_bnb.best_cost
    s = time.time()
    mod_tsp_bnb.solve(iters_limit=MAX_ITERS, print_iters=MAX_ITERS,
                      time_limit=compute_time_mins * 60, bnb_type=1)
    e = time.time()
    mod_tsp_bnb.persist()
    results['mod_bnb'].append(mod_tsp_bnb.best_cost)
    results['mod_bnb_time'].append(e - s)
    for _ in range(num_trials - 1):
        mod_tsp_bnb = load_latest_pckl(
            path1='Data/TravelingSalesmanProblem/DailyOpt')
        if mod_tsp_bnb is None:
            raise Exception('No saved instance for TSP branch and bound')
        s = time.time()
        mod_tsp_bnb.solve(iters_limit=MAX_ITERS, print_iters=200,
                          time_limit=compute_time_mins * 60)
        e = time.time()
        mod_tsp_bnb.persist()
        results['mod_bnb'].append(mod_tsp_bnb.best_cost)
        results['mod_bnb_time'].append(e - s)

    # clear continuous training data from previous branch and bound runs
    if os.path.isdir('Data/TravelingSalesmanProblem'):
        shutil.rmtree(path='Data/TravelingSalesmanProblem')

    # run traditional branch and bound
    trad_tsp_bnb = TravelingSalesmanProblem({'input_graph': city_graph})
    results['trad_bnb_init_sol'] = trad_tsp_bnb.best_solution
    results['trad_bnb_init_sol_cost'] = trad_tsp_bnb.best_cost
    s = time.time()
    trad_tsp_bnb.solve(iters_limit=MAX_ITERS, print_iters=MAX_ITERS,
                       time_limit=compute_time_mins * 60, bnb_type=0)
    e = time.time()
    trad_tsp_bnb.persist()
    results['trad_bnb'].append(trad_tsp_bnb.best_cost)
    results['trad_bnb_time'].append(e - s)
    for _ in range(num_trials - 1):
        trad_tsp_bnb = load_latest_pckl(
            path1='Data/TravelingSalesmanProblem/DailyOpt')
        if trad_tsp_bnb is None:
            raise Exception('No saved instance for TSP branch and bound')
        s = time.time()
        trad_tsp_bnb.solve(iters_limit=MAX_ITERS, print_iters=200,
                           time_limit=compute_time_mins * 60)
        e = time.time()
        trad_tsp_bnb.persist()
        results['trad_bnb'].append(trad_tsp_bnb.best_cost)
        results['trad_bnb_time'].append(e - s)

    # return results
    return results


if __name__ == '__main__':
    exp1_results = run_tsp_experiments(50, 1, 3)
    exp2_results = run_tsp_experiments(100, 2, 3)
    exp3_results = run_tsp_experiments(200, 4, 3)

    # print results
    results = [exp1_results, exp2_results, exp3_results]
    for i in range(len(results)):
        print(f'Results for Experiment {i}:\n')
        print(f'Simulated annealing:\n{results[i]["sa"]}')
        print(f'Branch and bound:\n{results[i]["bnb"]}')
        print(f'Local search heuristic:\n{results[i]["ls"]}')
