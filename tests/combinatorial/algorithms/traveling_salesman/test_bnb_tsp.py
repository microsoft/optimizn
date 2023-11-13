# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.traveling_salesman.city_graph\
    import CityGraph
from optimizn.combinatorial.algorithms.traveling_salesman.bnb_tsp import\
    TravelingSalesmanProblem
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np
from tests.combinatorial.algorithms.check_sol_utils import check_bnb_sol,\
    check_sol_optimality, check_sol_vs_init_sol


class MockCityGraph:
    def __init__(self, dists):
        self.dists = dists
        self.num_cities = len(dists)


def test_get_closest_city():
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
    TEST_CASES = [
        # test case: (current vertex, visited vertices, closest vertex to
        # current vertex)
        (0, {}, 3),
        (3, {0}, 2),
        (2, {0, 3}, 1),
        (1, {0, 3, 2}, None),
        (1, {0, 3, 2, 1}, None)
    ]
    for city, visited, true_closest in TEST_CASES:
        tsp = TravelingSalesmanProblem(params)
        closest = tsp._get_closest_city(city, visited)
        assert closest == true_closest, 'Incorrect closest vertex to '\
            + f'vertex {city}. Expected: {true_closest}. Actual: {closest}'


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
        (([0, 1, 2, 3], 3), True),
        (([1, 0, 2, 3], 2), True),
        (([1, 0, 2, 3], 4), False),
        (([1, 2, 2], 2), False),
        (([1, 2, 3], 2), True),
        (([1, 2, 3, 3], 2), False),
        (([1, 2, 3, 0, 1], 3), False)
    ]
    for sol, is_feasible in TEST_CASES:
        feasible = tsp.is_feasible(sol)
        assert is_feasible == feasible, 'Feasibility check failed '\
            + f'for solution {sol}. Expected to be feasible: {is_feasible}. '\
            + f'Actually feasible: {feasible}'


def test_is_complete():
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
        # test case: (solution, boolean for whether solution is complete)
        (([0, 1, 2, 3], 3), True),
        (([0, 1, 2, 3], 2), True),
        (([1, 0, 2, 3], 3), True),
        (([1, 0, 2, 3], 2), True),
        (([1, 2], 1), False),
        (([1, 2, 3], 2), False),
        (([1, 2, 3, 0], 3), True),
        (([1, 2, 3, 0, 1], 3), False)
    ]
    for sol, is_complete in TEST_CASES:
        complete = tsp.is_complete(sol)
        assert is_complete == complete, 'Completeness check failed '\
            + f'for solution {sol}. Expected to be complete: {complete}. '\
            + f'Actually complete: {complete}'


def test_complete_path():
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
        # test case: (path, completed path)
        ([], [0, 3, 2, 1]),
        ([0], [0, 3, 2, 1]),
        ([0, 1], [0, 1, 2, 3]),
        ([1, 3], [1, 3, 0, 2]),
        ([0, 3, 2, 1], [0, 3, 2, 1])
    ]
    for path, complete_path in TEST_CASES:
        comp_path = tsp._complete_path(path)
        assert comp_path == complete_path, 'Incorrect completed path formed '\
            + f'from path {path}. Expected: {complete_path}, Actual: '\
            + f'{comp_path}'


def test_get_candidate_sorted_dists():
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
    exp_init_sol = ([0, 3, 2, 1], -1)
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
        # test case: (partial solution, expected complete solution)
        (([], -1), ([0, 3, 2, 1], -1)),
        (([0], 0), ([0, 3, 2, 1], 0)),
        (([0, 1], 1), ([0, 1, 2, 3], 1)),
        (([1, 3], 1), ([1, 3, 0, 2], 1)),
        (([0, 3, 2, 1], 3), ([0, 3, 2, 1], 3))
    ]
    for path, complete_path in TEST_CASES:
        comp_path = tsp.complete_solution(path)
        assert comp_path == complete_path, 'Incorrect completed solution '\
            + f'formed from partial solution {path}. Expected: '\
            + f'{complete_path}, Actual: {comp_path}'


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
        (([0, 3, 2, 1], 2), 10),
        (([0, 3, 2, 1], 3), 10),
        (([0, 1, 2, 3], 1), 10),
        (([0, 1, 3, 2], 0), 12)
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
        (([0, 3, 2, 1], 3), 10),
        (([0, 3, 2, 1], 1), 6),
        (([0, 3, 2, 1], 2), 6),
        (([0, 1, 2, 3], 3), 10),
        (([0, 1, 3, 2], 3), 12),
        (([0, 3], 1), 6),
        (([0, 3, 2], 2), 6),
        (([0], 0), 8)
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
        (([0, 3, 2, 1], -1), [([0], 0), ([1], 0), ([2], 0), ([3], 0)]),
        (([0, 3, 2, 1], 3), []),
        (([0, 2, 1, 3], 4), []),
        (([0, 2, 1, 3], 1), [([0, 2, 1], 2), ([0, 2, 3], 2)]),
        (([0, 1], 1), [([0, 1, 2], 2), ([0, 1, 3], 2)]),
        (([0, 1, 2], 2), [([0, 1, 2, 3], 3)]),
        (([1], 0), [([1, 0], 1), ([1, 2], 1), ([1, 3], 1)])
    ]
    for sol, branch_sols in TEST_CASES:
        new_sols = tsp.branch(sol)
        assert branch_sols == new_sols, 'Incorrect branched solutions for '\
            + f'solution: {sol}. Expected: {branch_sols}, Actual: {new_sols}'


def test_bnb_tsp():
    graph = CityGraph()
    params = {
        'input_graph': graph,
    }
    tsp1 = TravelingSalesmanProblem(params)
    init_cost1 = tsp1.best_cost
    _, distance = solve_tsp_simulated_annealing(
        graph.dists, x0=tsp1.best_solution[0], perturbation_scheme='ps2',
        alpha=0.99)
    tsp1.solve(1e20, 1e20, 120, 0)
    tsp2 = TravelingSalesmanProblem(params)
    init_cost2 = tsp2.best_cost
    tsp2.solve(1e20, 1e20, 120, 1)

    # check final solutions
    check_bnb_sol(tsp1, 0, params)
    check_sol_vs_init_sol(tsp1.best_cost, init_cost1)
    check_bnb_sol(tsp2, 1, params)
    check_sol_vs_init_sol(tsp2.best_cost, init_cost2)
    check_sol_optimality(tsp2.best_cost, distance, 1.1)
