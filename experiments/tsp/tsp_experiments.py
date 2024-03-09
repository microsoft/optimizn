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
import pickle


# specify maximum number of iterations
MAX_ITERS = int(1e20)  # very high bound on iterations
# since algorithms should use up all compute time

# specify number of iterations for logging (optimizn)
LOG_ITERS = int(1e7)


def clear_previous_data():
    # clear previous continuous training data
    if os.path.isdir('Data/'):
        shutil.rmtree(path='Data/')
        print('Cleared previous continuous training data')
    else:
        print('No previous continuous training data found')

    # clear previous city graph object
    if os.path.isfile('city_graph.obj'):
        os.remove('city_graph.obj')
        print('Cleared previous city graph object')
    else:
        print('No previous city graph object found')

    # clear previous experiment results object
    if os.path.isfile('exp_results.obj'):
        os.remove('exp_results.obj')
        print('Cleared previous experiment results dictionary')
    else:
        print('No previous experiment results dictionary found')

    print('Cleared previous experiment data')


def _clear_cont_train_data(opt_prob_obj):
    # clear continuous training data from previous runs
    if os.path.isdir(f'Data/{opt_prob_obj.name}'):
        shutil.rmtree(path=f'Data/{opt_prob_obj.name}')
        print('Cleared previous continuous training data for optimization '
              + f'problem class with name {opt_prob_obj.name}')


# function to get experiment graph and results dictionary
def get_exp_data(num_cities):
    # create/load experiment graph
    if os.path.isfile('city_graph.obj'):
        city_graph = pickle.load(open('city_graph.obj', 'rb'))
        print('Loaded saved city graph')
    else:
        city_graph = CityGraph(num_cities)
        pickle.dump(city_graph, open('city_graph.obj', 'wb'))
        print('Created and saved new city graph')

    # create/load experiment results dictionary
    if os.path.isfile('exp_results.obj'):
        exp_results = pickle.load(open('exp_results.obj', 'rb'))
        print('Loaded saved experiment results dictionary')
    else:
        exp_results = dict()
        print('Created new experiment results dictionary')
    return city_graph, exp_results


# function to save experiment results dictionary
def save_exp_results(exp_results):
    pickle.dump(exp_results, open('exp_results.obj', 'wb'))
    print('Saved experiment results dictionary')


def run_o_sa1(city_graph, results, compute_time_mins, num_trials, reset_p):
    results['o_sa1'] = []
    results['o_sa1_time'] = []
    tsp_o_sa1 = TravSalsmn(city_graph)
    results['o_sa1_init_sol'] = tsp_o_sa1.best_solution
    results['o_sa1_init_sol_cost'] = tsp_o_sa1.best_cost
    s = time.time()
    tsp_o_sa1.anneal(n_iter=MAX_ITERS, reset_p=reset_p,
                   time_limit=compute_time_mins * 60 * num_trials,
                   log_iters=LOG_ITERS)
    e = time.time()
    results['o_sa1'].append(tsp_o_sa1.best_cost)
    results['o_sa1_time'].append(e - s)


def run_o_sa2(city_graph, results, compute_time_mins, num_trials, reset_p):
    results['o_sa2'] = []
    results['o_sa2_time'] = []
    tsp_o_sa2 = TravSalsmn(city_graph)
    _clear_cont_train_data(tsp_o_sa2)
    results['o_sa2_init_sol'] = tsp_o_sa2.best_solution
    results['o_sa2_init_sol_cost'] = tsp_o_sa2.best_cost
    s = time.time()
    tsp_o_sa2.anneal(n_iter=MAX_ITERS, reset_p=reset_p,
                   time_limit=compute_time_mins * 60, log_iters=LOG_ITERS)
    e = time.time()
    tsp_o_sa2.persist()
    results['o_sa2'].append(tsp_o_sa2.best_cost)
    results['o_sa2_time'].append(e - s)
    for _ in range(num_trials - 1):
        class_name = tsp_o_sa2.name
        prior_params = load_latest_pckl(
            path1=f'Data/{class_name}/DailyObj', logger=tsp_o_sa2.logger)
        if tsp_o_sa2.params == prior_params:
            tsp_o_sa2 = load_latest_pckl(
                path1=f'Data/{class_name}/DailyOpt', logger=tsp_o_sa2.logger)
            if tsp_o_sa2 is None:
                raise Exception(
                    'No saved instance for TSP simulated annealing')
        else:
            raise Exception('TSP simulated annealing parameters have changed')
        s = time.time()
        tsp_o_sa2.anneal(n_iter=MAX_ITERS, reset_p=reset_p,
                       time_limit=compute_time_mins * 60,
                       log_iters=LOG_ITERS)
        e = time.time()
        tsp_o_sa2.persist()
        results['o_sa2'].append(tsp_o_sa2.best_cost)
        results['o_sa2_time'].append(e - s)


def run_pt_sa1(city_graph, results, compute_time_mins, num_trials):
    results['pt_sa1'] = []
    results['pt_sa1_time'] = []
    permutation = list(range(city_graph.num_cities))
    opt_dist = 0
    for i in range(1, len(permutation)):
        opt_dist += city_graph.dists[permutation[i], permutation[i-1]]
    opt_dist += city_graph.dists[
        permutation[0], permutation[len(permutation) - 1]]
    results['pt_sa1_init_sol'] = permutation
    results['pt_sa1_init_sol_cost'] = opt_dist
    s = time.time()
    e = time.time()
    while (e - s) < (compute_time_mins * 60 * num_trials):
        permutation, distance = solve_tsp_simulated_annealing(
            city_graph.dists,
            max_processing_time=(
                compute_time_mins * 60 * num_trials) - (e - s),
            alpha=0.99, x0=permutation, perturbation_scheme='ps2')
        if opt_dist > distance:
            opt_dist = distance
        e = time.time()
    results['pt_sa1'].append(opt_dist)
    results['pt_sa1_time'].append(e - s)


def run_pt_sa2(city_graph, results, compute_time_mins, num_trials):
    results['pt_sa2'] = []
    results['pt_sa2_time'] = []
    permutation = list(range(city_graph.num_cities))
    opt_dist = 0
    for i in range(1, len(permutation)):
        opt_dist += city_graph.dists[permutation[i], permutation[i-1]]
    opt_dist += city_graph.dists[
        permutation[0], permutation[len(permutation) - 1]]
    results['pt_sa2_init_sol'] = permutation
    results['pt_sa2_init_sol_cost'] = opt_dist
    s = time.time()
    e = time.time()
    while (e - s) < (compute_time_mins * 60):
        permutation, distance = solve_tsp_simulated_annealing(
            city_graph.dists,
            max_processing_time=(compute_time_mins * 60) - (e - s),
            alpha=0.99, x0=permutation, perturbation_scheme='ps2')
        if opt_dist > distance:
            opt_dist = distance
        e = time.time()
    results['pt_sa2'].append(opt_dist)
    results['pt_sa2_time'].append(e - s)
    for _ in range(num_trials - 1):
        s = time.time()
        e = time.time()
        while (e - s) < (compute_time_mins * 60):
            permutation, distance = solve_tsp_simulated_annealing(
                city_graph.dists,
                max_processing_time=(compute_time_mins * 60) - (e - s),
                alpha=0.99, x0=permutation, perturbation_scheme='ps2')
            if opt_dist > distance:
                opt_dist = distance
            e = time.time()
        results['pt_sa2'].append(opt_dist)
        results['pt_sa2_time'].append(e - s)


def run_mod_bnb1(city_graph, results, compute_time_mins, num_trials,
                 depth_first):
    if depth_first:
        alg_name = 'df_mod_bnb1'
    else:
        alg_name = 'mod_bnb1'
    results[f'{alg_name}'] = []
    results[f'{alg_name}_time'] = []
    mod_bnb1 = TravelingSalesmanProblem({'input_graph': city_graph})
    results[f'{alg_name}_init_sol'] = mod_bnb1.best_solution
    results[f'{alg_name}_init_sol_cost'] = mod_bnb1.best_cost
    s = time.time()
    mod_bnb1.solve(iters_limit=MAX_ITERS, log_iters=LOG_ITERS,
                   time_limit=compute_time_mins * 60 * num_trials,
                   bnb_type=1, depth_first=depth_first)
    e = time.time()
    results[f'{alg_name}'].append(mod_bnb1.best_cost)
    results[f'{alg_name}_time'].append(e - s)


def run_mod_bnb2(city_graph, results, compute_time_mins, num_trials,
                 depth_first):
    if depth_first:
        alg_name = 'df_mod_bnb2'
    else:
        alg_name = 'mod_bnb2'
    results[f'{alg_name}'] = []
    results[f'{alg_name}_time'] = []
    mod_bnb2 = TravelingSalesmanProblem({'input_graph': city_graph})
    _clear_cont_train_data(mod_bnb2)
    results[f'{alg_name}_init_sol'] = mod_bnb2.best_solution
    results[f'{alg_name}_init_sol_cost'] = mod_bnb2.best_cost
    s = time.time()
    mod_bnb2.solve(iters_limit=MAX_ITERS, log_iters=LOG_ITERS,
                   time_limit=compute_time_mins * 60, bnb_type=1,
                   depth_first=depth_first)
    e = time.time()
    mod_bnb2.persist()
    results[f'{alg_name}'].append(mod_bnb2.best_cost)
    results[f'{alg_name}_time'].append(e - s)
    for _ in range(num_trials - 1):
        class_name = mod_bnb2.name
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
        mod_bnb2.solve(iters_limit=MAX_ITERS, log_iters=LOG_ITERS,
                       time_limit=compute_time_mins * 60, bnb_type=1,
                       depth_first=depth_first)
        e = time.time()
        mod_bnb2.persist()
        results[f'{alg_name}'].append(mod_bnb2.best_cost)
        results[f'{alg_name}_time'].append(e - s)


def run_trad_bnb1(city_graph, results, compute_time_mins, num_trials,
                  depth_first):
    if depth_first:
        alg_name = 'df_trad_bnb1'
    else:
        alg_name = 'trad_bnb1'
    results[f'{alg_name}'] = []
    results[f'{alg_name}_time'] = []
    trad_bnb1 = TravelingSalesmanProblem({'input_graph': city_graph})
    results[f'{alg_name}_init_sol'] = trad_bnb1.best_solution
    results[f'{alg_name}_init_sol_cost'] = trad_bnb1.best_cost
    s = time.time()
    trad_bnb1.solve(iters_limit=MAX_ITERS, log_iters=LOG_ITERS,
                    time_limit=compute_time_mins * 60 * num_trials,
                    bnb_type=0, depth_first=depth_first)
    e = time.time()
    results[f'{alg_name}'].append(trad_bnb1.best_cost)
    results[f'{alg_name}_time'].append(e - s)


def run_trad_bnb2(city_graph, results, compute_time_mins, num_trials,
                  depth_first):
    if depth_first:
        alg_name = 'df_trad_bnb2'
    else:
        alg_name = 'trad_bnb2'
    results[f'{alg_name}'] = []
    results[f'{alg_name}_time'] = []
    trad_bnb2 = TravelingSalesmanProblem({'input_graph': city_graph})
    _clear_cont_train_data(trad_bnb2)
    results[f'{alg_name}_init_sol'] = trad_bnb2.best_solution
    results[f'{alg_name}_init_sol_cost'] = trad_bnb2.best_cost
    s = time.time()
    trad_bnb2.solve(iters_limit=MAX_ITERS, log_iters=LOG_ITERS,
                    time_limit=compute_time_mins * 60, bnb_type=0,
                    depth_first=depth_first)
    e = time.time()
    trad_bnb2.persist()
    results[f'{alg_name}'].append(trad_bnb2.best_cost)
    results[f'{alg_name}_time'].append(e - s)
    for _ in range(num_trials - 1):
        class_name = trad_bnb2.name
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
        trad_bnb2.solve(iters_limit=MAX_ITERS, log_iters=LOG_ITERS,
                        time_limit=compute_time_mins * 60,
                        depth_first=depth_first)
        e = time.time()
        trad_bnb2.persist()
        results[f'{alg_name}'].append(trad_bnb2.best_cost)
        results[f'{alg_name}_time'].append(e - s)


def run_tsp_experiments(num_cities=200, compute_time_mins=1, num_trials=3,
                        reset_p=1/1000000):
    # for collecting results
    results = dict()

    # create traveling salesman problem parameters
    city_graph = CityGraph(num_cities)

    # run optimizn simulated annealing (single stretch)
    run_o_sa1(city_graph, results, compute_time_mins, num_trials, reset_p)

    # run optimizn simulated annealing (successive runs)
    run_o_sa2(city_graph, results, compute_time_mins, num_trials, reset_p)

    # run python-tsp simulated annealing (single stretch)
    run_pt_sa1(city_graph, results, compute_time_mins, num_trials)

    # run python-tsp simulated annealing (successive runs)
    run_pt_sa2(city_graph, results, compute_time_mins, num_trials)

    # run optimizn modified branch and bound (single stretch)
    run_mod_bnb1(city_graph, results, compute_time_mins, num_trials)

    # run optimizn modified branch and bound (successive runs)
    run_mod_bnb2(city_graph, results, compute_time_mins, num_trials)

    # run optimizn traditional branch and bound (single stretch)
    run_trad_bnb1(city_graph, results, compute_time_mins, num_trials)

    # run optimizn traditional branch and bound (successive runs)
    run_trad_bnb2(city_graph, results, compute_time_mins, num_trials)

    # return results
    return city_graph, results
