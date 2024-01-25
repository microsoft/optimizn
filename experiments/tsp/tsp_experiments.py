# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.traveling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.algorithms.traveling_salesman.sim_anneal_tsp\
    import TravSalsmn
from optimizn.combinatorial.algorithms.traveling_salesman.bnb_tsp\
    import TravelingSalesmanProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
from experiments.tsp.tsp_experiments_utils import run_python_tsp_bnb
from optimizn.combinatorial.opt_problem import load_latest_pckl
import time
import shutil
import os
import multiprocessing


def run_tsp_experiments(num_cities=50, compute_time_mins=1, num_trials=3,
                        reset_p=1/10000):
    # specify maximum number of iterations
    MAX_ITERS = int(1e20)  # very high bound on iterations
    # since algorithms should use up all compute time

    # for collecting results
    results = dict()

    # create traveling salesman problem parameters
    city_graph = CityGraph(num_cities)

    # delete continuous training results from previous experiment runs
    if os.path.isdir('Data/'):
        shutil.rmtree('Data/')

    # run simulated annealing 1
    results['sa1'] = []
    results['sa1_time'] = []
    tsp_sa1 = TravSalsmn(city_graph, temp_reduce_factor=0.99)
    results['sa1_init_sol'] = tsp_sa1.best_solution
    results['sa1_init_sol_cost'] = tsp_sa1.best_cost
    s = time.time()
    tsp_sa1.anneal(n_iter=MAX_ITERS, reset_p=reset_p,
                  time_limit=compute_time_mins * 60 * num_trials)
    e = time.time()
    results['sa1'].append(tsp_sa1.best_cost)
    results['sa1_time'].append(e - s)

    # run simulated annealing 2
    results['sa2'] = []
    results['sa2_time'] = []
    tsp_sa2 = TravSalsmn(city_graph, temp_reduce_factor=0.99)
    results['sa2_init_sol'] = tsp_sa2.best_solution
    results['sa2_init_sol_cost'] = tsp_sa2.best_cost
    s = time.time()
    tsp_sa2.anneal(n_iter=MAX_ITERS, reset_p=reset_p,
                  time_limit=compute_time_mins * 60)
    e = time.time()
    tsp_sa2.persist()
    results['sa2'].append(tsp_sa2.best_cost)
    results['sa2_time'].append(e - s)
    for _ in range(num_trials - 1):
        class_name = tsp_sa2.__class__.__name__
        prior_params = load_latest_pckl(
            path1=f'Data/{class_name}/DailyObj', logger=tsp_sa2.logger)
        if tsp_sa2.params == prior_params:
            tsp_sa2 = load_latest_pckl(
                path1=f'Data/{class_name}/DailyOpt', logger=tsp_sa2.logger)
            if tsp_sa2 is None:
                raise Exception(
                    'No saved instance for TSP simulated annealing')
        else:
            raise Exception('TSP simulated annealing parameters have changed')
        s = time.time()
        tsp_sa2.anneal(n_iter=MAX_ITERS, reset_p=reset_p,
                      time_limit=compute_time_mins * 60)
        e = time.time()
        tsp_sa2.persist()
        results['sa2'].append(tsp_sa2.best_cost)
        results['sa2_time'].append(e - s)

    # run simulated annealing 3
    results['sa3'] = []
    results['sa3_time'] = []
    opt_permutation = list(range(city_graph.num_cities))
    opt_dist = 0
    for i in range(1, len(opt_permutation)):
        opt_dist += city_graph.dists[opt_permutation[i], opt_permutation[i-1]]
    opt_dist += city_graph.dists[
        opt_permutation[0], opt_permutation[len(opt_permutation) - 1]]
    results['sa3_init_sol'] = opt_permutation
    results['sa3_init_sol_cost'] = opt_dist
    s = time.time()
    e = time.time()
    while (e - s) < (compute_time_mins * 60 * num_trials):
        permutation, distance = solve_tsp_simulated_annealing(
            city_graph.dists,
            max_processing_time=(
                compute_time_mins * 60 * num_trials) - (e - s),
            alpha=0.99, x0=opt_permutation, perturbation_scheme='ps2')
        if opt_dist > distance:
            opt_dist = distance
            opt_permutation = permutation
        e = time.time()
    results['sa3'].append(opt_dist)
    results['sa3_time'].append(e - s)

    # run simulated annealing 4
    results['sa4'] = []
    results['sa4_time'] = []
    opt_permutation = list(range(city_graph.num_cities))
    opt_dist = 0
    for i in range(1, len(opt_permutation)):
        opt_dist += city_graph.dists[opt_permutation[i], opt_permutation[i-1]]
    opt_dist += city_graph.dists[
        opt_permutation[0], opt_permutation[len(opt_permutation) - 1]]
    results['sa4_init_sol'] = opt_permutation
    results['sa4_init_sol_cost'] = opt_dist
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
    results['sa4'].append(opt_dist)
    results['sa4_time'].append(e - s)
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
        results['sa4'].append(opt_dist)
        results['sa4_time'].append(e - s)

    # run modified branch and bound 1
    results['mod_bnb1'] = []
    results['mod_bnb1_time'] = []
    mod_bnb1 = TravelingSalesmanProblem({'input_graph': city_graph})
    results['mod_bnb1_init_sol'] = mod_bnb1.best_solution
    results['mod_bnb1_init_sol_cost'] = mod_bnb1.best_cost
    s = time.time()
    mod_bnb1.solve(iters_limit=MAX_ITERS, log_iters=MAX_ITERS,
                       time_limit=compute_time_mins * 60 * num_trials,
                       bnb_type=1)
    e = time.time()
    results['mod_bnb1'].append(mod_bnb1.best_cost)
    results['mod_bnb1_time'].append(e - s)

    # run modified branch and bound 2
    results['mod_bnb2'] = []
    results['mod_bnb2_time'] = []
    mod_bnb2 = TravelingSalesmanProblem({'input_graph': city_graph})
    results['mod_bnb2_init_sol'] = mod_bnb2.best_solution
    results['mod_bnb2_init_sol_cost'] = mod_bnb2.best_cost
    s = time.time()
    mod_bnb2.solve(iters_limit=MAX_ITERS, log_iters=MAX_ITERS,
                      time_limit=compute_time_mins * 60, bnb_type=1)
    e = time.time()
    mod_bnb2.persist()
    results['mod_bnb2'].append(mod_bnb2.best_cost)
    results['mod_bnb2_time'].append(e - s)
    for _ in range(num_trials - 1):
        class_name = mod_bnb2.__class__.__name__
        prior_params = load_latest_pckl(
            path1=f'Data/{class_name}/DailyObj', logger=mod_bnb2.logger)
        if mod_bnb2.params == prior_params:
            mod_bnb2 = load_latest_pckl(
                path1=f'Data/{class_name}/DailyOpt', logger=mod_bnb2.logger)
            if mod_bnb2 is None:
                raise Exception(
                    'No saved instance for TSP modified branch '
                    + 'and bound')
        else:
            raise Exception(
                'TSP modified branch and bound parameters have changed')
        s = time.time()
        mod_bnb2.solve(iters_limit=MAX_ITERS, log_iters=200,
                          time_limit=compute_time_mins * 60)
        e = time.time()
        mod_bnb2.persist()
        results['mod_bnb2'].append(mod_bnb2.best_cost)
        results['mod_bnb2_time'].append(e - s)

    # clear continuous training data from previous branch and bound runs
    class_name = mod_bnb2.__class__.__name__
    if os.path.isdir(f'Data/{class_name}'):
        shutil.rmtree(path=f'Data/{class_name}')

    # run traditional branch and bound 1
    results['trad_bnb1'] = []
    results['trad_bnb1_time'] = []
    trad_bnb1 = TravelingSalesmanProblem({'input_graph': city_graph})
    results['trad_bnb1_init_sol'] = trad_bnb1.best_solution
    results['trad_bnb1_init_sol_cost'] = trad_bnb1.best_cost
    s = time.time()
    trad_bnb1.solve(iters_limit=MAX_ITERS, log_iters=MAX_ITERS,
                    time_limit=compute_time_mins * 60 * num_trials,
                    bnb_type=0)
    e = time.time()
    results['trad_bnb1'].append(trad_bnb1.best_cost)
    results['trad_bnb1_time'].append(e - s)

    # run traditional branch and bound 2
    results['trad_bnb2'] = []
    results['trad_bnb2_time'] = []
    trad_bnb2 = TravelingSalesmanProblem({'input_graph': city_graph})
    results['trad_bnb2_init_sol'] = trad_bnb2.best_solution
    results['trad_bnb2_init_sol_cost'] = trad_bnb2.best_cost
    s = time.time()
    trad_bnb2.solve(iters_limit=MAX_ITERS, log_iters=MAX_ITERS,
                    time_limit=compute_time_mins * 60, bnb_type=0)
    e = time.time()
    trad_bnb2.persist()
    results['trad_bnb2'].append(trad_bnb2.best_cost)
    results['trad_bnb2_time'].append(e - s)
    for _ in range(num_trials - 1):
        class_name = trad_bnb2.__class__.__name__
        prior_params = load_latest_pckl(
            path1=f'Data/{class_name}/DailyObj', logger=trad_bnb2.logger)
        if trad_bnb2.params == prior_params:
            trad_bnb2 = load_latest_pckl(
                path1=f'Data/{class_name}/DailyOpt', logger=trad_bnb2.logger)
            if trad_bnb2 == None:
                raise Exception(
                    'No saved instance for TSP traditional branch '
                    + 'and bound')
        else:
            raise Exception(
                'TSP traditional branch and bound parameters have changed')
        s = time.time()
        trad_bnb2.solve(iters_limit=MAX_ITERS, log_iters=200,
                        time_limit=compute_time_mins * 60)
        e = time.time()
        trad_bnb2.persist()
        results['trad_bnb2'].append(trad_bnb2.best_cost)
        results['trad_bnb2_time'].append(e - s)
    
    # run traditional branch and bound 3
    results['trad_bnb3'] = []
    results['trad_bnb3_time'] = []
    init_permutation = list(range(city_graph.num_cities))
    init_dist = 0
    for i in range(1, len(init_permutation)):
        init_dist += city_graph.dists[init_permutation[i], init_permutation[i-1]]
    init_dist += city_graph.dists[
        init_permutation[0], init_permutation[len(init_permutation) - 1]]
    results['trad_bnb3_init_sol'] = init_permutation
    results['trad_bnb3_init_sol_cost'] = init_dist
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_python_tsp_bnb, args=(q, city_graph.dists))
    s = time.time()
    p.start()
    p.join(compute_time_mins * 60 * num_trials)
    if p.is_alive():
        p.terminate()
        e = time.time()
        results['trad_bnb3'].append(init_dist)
    else:
        e = time.time()
        opt_dist = q.get()
        results['trad_bnb3'].append(opt_dist)
    results['trad_bnb3_time'].append(e - s)

    # return results
    return results
