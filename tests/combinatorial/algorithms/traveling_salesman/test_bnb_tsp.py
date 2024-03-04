# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.traveling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.algorithms.traveling_salesman.bnb_tsp import\
    TravelingSalesmanProblem
import numpy as np
from tests.combinatorial.algorithms.check_sol_utils import check_bnb_sol,\
    check_sol_vs_init_sol
from copy import deepcopy


class MockCityGraph:
    def __init__(self, dists):
        self.dists = dists
        self.num_cities = len(dists)


def test_is_feasible():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        # test case: (solution, boolean for whether solution is feasible)
        ([0, 1, 2, 3], True),
        ([1, 2], False),
        ([1, 2, 3], False),
        ([1, 2, 3, 0], True)
    ]
    for sol, is_feasible in TEST_CASES:
        feasible = tsp.is_feasible(sol)
        assert is_feasible == feasible, 'feasiblity check failed for solution '\
            + f'{sol}. Expected to be feasible: {is_feasible}. Actually '\
            + f'feasible: {feasible}'


def test_get_initial_solution_sorted_dists():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    exp_init_sol = [0, 1, 2, 3]
    exp_sorted_dists = [1, 2, 2, 3, 4, 4]
    assert tsp.best_solution == exp_init_sol, 'Invalid initial solution. '\
        + f'Expected: {exp_init_sol}. Actual: {tsp.best_solution}'
    assert tsp.sorted_dists == exp_sorted_dists, 'Invalid sorted distances. '\
        + f'Expected: {tsp.sorted_dists}. Actual: {exp_sorted_dists}'


def test_complete_solution():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        # test case: partial solution
        [], [0], [0, 1], [1, 3], [0, 3, 2, 1]
    ]
    for path in TEST_CASES:
        comp_path = tsp.complete_solution(path)
        assert len(comp_path) == mcg.num_cities,\
            'Incorrect length of completed path. Expected: '\
            + f'{mcg.num_cities}. Actual: {len(comp_path)}'
        assert set(comp_path) == set(range(mcg.num_cities)), 'Incorrect '\
            + f'coverage of cities. Expected: {set(range(mcg.num_cities))}'\
            + f'. Actual: {set(comp_path)}'


def test_cost():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        # test case: (solution, cost of solution)
        ([0, 3, 2, 1], 10),
        ([0, 3, 2, 1], 10),
        ([0, 1, 2, 3], 10),
        ([0, 1, 3, 2], 12)
    ]
    for sol, cost in TEST_CASES:
        sol_cost = tsp.cost(sol)
        assert sol_cost == cost, f'Incorrect cost for solution {sol}. '\
            + f'Expected: {cost}. Actual: {sol_cost}'


def test_lbound():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        # test case: (solution, lower bound of solution)
        ([0, 3, 2, 1], 10),
        ([0, 3], 6),
        ([0, 3, 2], 6),
        ([0, 1, 2, 3], 10),
        ([0, 1, 3, 2], 12),
        ([0], 8),
        ([], 8)
    ]
    for sol, lower_bound in TEST_CASES:
        lb = tsp.lbound(sol)
        assert lb == lower_bound, 'Incorrect lower bound for solution '\
            + f'{sol}. Expected: {lower_bound}. Actual: {lb}'


def test_branch():
    dists = np.array([
        [0, 4, 2, 1],
        [4, 0, 3, 4],
        [2, 3, 0, 2],
        [1, 4, 2, 0],
    ])
    mcg = MockCityGraph(dists)
    params = {
        'input_graph': mcg,
    }
    tsp = TravelingSalesmanProblem(params)
    TEST_CASES = [
        # test case: (solution, expected branched solutions)
        ([], [[0], [1], [2], [3]]),
        ([0, 3, 2, 1], []),
        ([0, 2, 1, 3], []),
        ([0, 2], [[0, 2, 1], [0, 2, 3]]),
        ([0, 1], [[0, 1, 2], [0, 1, 3]]),
        ([0, 1, 2], [[0, 1, 2, 3]]),
        ([1], [[1, 0], [1, 2], [1, 3]])
    ]
    for sol, branch_sols in TEST_CASES:
        new_sols = tsp.branch(sol)
        assert branch_sols == new_sols, 'Incorrect branched solutions for '\
            + f'solution: {sol}. Expected: {branch_sols}, Actual: {new_sols}'


def test_get_root():
    graph = CityGraph()
    params = {
        'input_graph': graph,
    }
    tsp = TravelingSalesmanProblem(params)
    root_sol = tsp.get_root()
    assert root_sol == [0], 'Incorrect root node solution. Expected: '\
        + f'[0], Actual: {root_sol}'


def test_bnb_tsp():
    graph = CityGraph()
    params = {
        'input_graph': graph,
    }
    tsp1 = TravelingSalesmanProblem(params)
    tsp1.solve(1e20, 1e20, 120, 1)
    tsp2 = TravelingSalesmanProblem(params)
    tsp2.solve(1e20, 1e20, 120, 1)

    # check final solutions
    check_bnb_sol(tsp1, 0, params)
    check_sol_vs_init_sol(tsp1.best_cost, tsp1.init_cost)
    check_bnb_sol(tsp2, 1, params)
    check_sol_vs_init_sol(tsp2.best_cost, tsp2.init_cost)


def test_depth_first():
    # input params
    graph = CityGraph()
    graph.num_cities = 4
    graph.dists = np.array([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
        [1, 2, 0, 1],
        [2, 1, 1, 0]
    ])
    params = {
        'input_graph': graph,
    }

    # depth first
    tsp = TravelingSalesmanProblem(params)
    assert not tsp.depth_first, 'Incorrect depth first setting. Expected: '\
        + 'False. Actual: True'
    tsp.solve(iters_limit=2, time_limit=1e20, log_iters=1, bnb_type=0,
              depth_first=True)
    assert tsp.depth_first, 'Incorrect depth first setting. Expected: True. '\
        + 'Actual: False'
    sols = []
    while not tsp.queue.empty():
        sols.append(tsp.queue.get()[-1])
    exp_sols = [[0, 1, 3], [0, 1, 2], [0, 2], [0, 3]]
    assert exp_sols == sols, 'Incorrect order of solutions. Expected: '\
        + f'{exp_sols}. Actual: {sols}'
    
    # not depth first
    tsp = TravelingSalesmanProblem(params)
    assert not tsp.depth_first, 'Incorrect depth first setting. Expected: '\
        + 'False. Actual: True'
    tsp.solve(iters_limit=2, time_limit=1e20, log_iters=1, bnb_type=0,
              depth_first=False)
    assert not tsp.depth_first, 'Incorrect depth first setting. Expected: '\
        + 'False. Actual: True'
    sols = []
    while not tsp.queue.empty():
        sols.append(tsp.queue.get()[-1])
    exp_sols = [[0, 1, 3], [0, 2], [0, 1, 2], [0, 3]]
    assert exp_sols == sols, 'Incorrect order of solutions. Expected: '\
        + f'{exp_sols}. Actual: {sols}'
    
    # check that queue ordering changes when depth first setting changes
    # from False to True
    tsp = TravelingSalesmanProblem(params)
    tsp.solve(iters_limit=2, time_limit=1e20, log_iters=1, bnb_type=0,
              depth_first=False)
    assert not tsp.depth_first, 'Incorrect depth first setting. Expected: '\
        + 'False. Actual: True'
    tsp.solve(iters_limit=0, time_limit=1e20, log_iters=1, bnb_type=0,
              depth_first=True)
    assert tsp.depth_first, 'Incorrect depth first setting. Expected: True. '\
        + 'Actual: False'
    sols = []
    while not tsp.queue.empty():
        sols.append(tsp.queue.get()[-1])
    exp_sols = [[0, 1, 3], [0, 1, 2], [0, 2], [0, 3]]
    assert exp_sols == sols, 'Incorrect order of solutions. Expected: '\
        + f'{exp_sols}. Actual: {sols}'

    # check that queue ordering changes when depth first setting changes
    # from True to False
    tsp = TravelingSalesmanProblem(params)
    tsp.solve(iters_limit=2, time_limit=1e20, log_iters=1, bnb_type=0,
              depth_first=True)
    assert tsp.depth_first, 'Incorrect depth first setting. Expected: True. '\
        + 'Actual: False'
    tsp.solve(iters_limit=0, time_limit=1e20, log_iters=1, bnb_type=0,
              depth_first=False)
    assert not tsp.depth_first, 'Incorrect depth first setting. Expected: '\
        + 'False. Actual: True'
    sols = []
    while not tsp.queue.empty():
        sols.append(tsp.queue.get()[-1])
    exp_sols = [[0, 1, 3], [0, 2], [0, 1, 2], [0, 3]]
    assert exp_sols == sols, 'Incorrect order of solutions. Expected: '\
        + f'{exp_sols}. Actual: {sols}'
