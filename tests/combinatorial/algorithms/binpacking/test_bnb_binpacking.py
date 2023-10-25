# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.binpacking.bnb_binpacking import\
    BinPackingParams, BinPackingProblem
from tests.combinatorial.algorithms.check_sol_utils import check_bnb_sol,\
    check_sol_optimality, check_sol_vs_init_sol


def test_param_equality():
    TEST_CASES = [
        # test case: (first instance of bin packing parameters, second instance
        # of bin packing parameters, boolean for whether instances are equal)
        (BinPackingParams([1, 2, 3, 4], [6]), None, False),
        (BinPackingParams([1, 2, 3, 4], [7]), BinPackingParams(
            [1, 2, 3, 4], [6]), False),
        (BinPackingParams([1, 2, 3, 4], [6]), BinPackingParams(
            [1, 2, 3, 4], [6]), True)
    ]
    for params1, params2, equal in TEST_CASES:
        assert (params1 == params2) == equal, 'Equality check failed for '\
            + f'BinPackingParams:\n{params1}\n{params2}\nExpected to be '\
            + f'equal: {equal}. Actually equal: {(params1 == params2)}'


def test_constructor():
    TEST_CASES = [
        # test case: (weights, capacity, expected initial solution)
        ([1, 2, 3], 3, {1: {3}, 2: {1, 2}}),
        ([7, 8, 2, 3], 15, {1: {2, 1}, 2: {4, 3}})
    ]
    for weights, capacity, expected in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check capacity
        assert bpp.capacity == capacity, 'Incorrect capacity. Expected: '\
            + f'{capacity}. Actual: {bpp.capacity}'

        # check item weights
        for i in range(len(weights)):
            assert bpp.item_weights[i + 1] == weights[i], 'Incorrect weight '\
                + f'for item {i + 1}. Expected: {weights[i]}. Actual: '\
                + f'{bpp.item_weights[i + 1]}'

        # check sorted item weights
        for i in range(len(bpp.sorted_item_weights)):
            weight, item = bpp.sorted_item_weights[i]
            assert bpp.item_weights[item] == weight, 'Incorrect weight for '\
                + f'item {item}. Expected {weight}. Actual: '\
                + f'{bpp.item_weights[item]}'
            if i > 0:
                assert weight <= bpp.sorted_item_weights[i - 1][0], 'Item '\
                    + f'weights not sorted in descending order. Element at '\
                    + f'index {i} in sorted list of item weights ({weight}) '\
                    + f'is greater than element at index {i - 1} '\
                    + f'({bpp.sorted_item_weights[i - 1][0]})'

        # check initial solution
        assert bpp.best_solution[0] == expected, 'Incorrect allocation of '\
            + f'items to bins in initial solution. Expected: {expected}. '\
            + f'Actual: {bpp.best_solution[0]}'
        assert bpp.best_solution[1] == -1, 'Incorrect value for index of '\
            + f'last allocated item in sorted-by-decreasing-weight list of '\
            + f'items. Expected: -1. Actual: {bpp.best_solution[1]}'


def test_is_feasible():
    TEST_CASES = [
        # test case: (weights, capacity, solution, boolean for whether
        # solution is feasible)
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}, 3: {1}}, 1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}}, 1), True),
        ([1, 2, 3], 3, ({1: {1, 2, 3}}, -1), False),
        ([1, 2, 3], 3, ({1: {3, 1}, 2: {2}}, -1), False),
        ([1, 2, 3], 3, ({1: {3, 2}, 2: {1}}, 1), False),
        ([1, 2, 3], 3, ({1: {3, 2}, 2: {1}}, 1), False)
    ]
    for weights, capacity, sol, feasible in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check feasibility
        is_feasible = bpp.is_feasible(sol)
        assert is_feasible == feasible, 'Feasibility check failed '\
            + f'for solution {sol}. Expected to be feasible: {feasible}. '\
            + f'Actually feasible: {is_feasible}'


def test_is_complete():
    TEST_CASES = [
        # test case: (weights, capacity, solution, boolean for whether
        # solution is complete)
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}, 3: {1}}, 1), True),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}}, 1), False),
        ([1, 2, 3], 3, ({1: {3}, 2: {1}}, 1), False),
        ([1, 2, 3], 3, ({1: {2}, 2: {1}}, 1), False)
    ]
    for weights, capacity, sol, complete in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check completeness
        is_complete = bpp.is_complete(sol)
        assert is_complete == complete, 'Completeness check failed '\
            + f'for solution {sol}. Expected to be complete: {complete}. '\
            + f'Actually complete: {is_complete}'


def test_cost():
    TEST_CASES = [
        # test cost: (weights, capacity, solution, expected solution cost)
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), 2),
        ([1, 2, 3], 3, ({1: {1, 2}, 2: {3}}, -1), 2),
        ([1, 2, 3], 3, ({1: {2}, 2: {3}, 3: {1}}, 2), 3),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}, 3: {1}}, 2), 3)
    ]
    for weights, capacity, sol, cost in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check cost
        sol_cost = bpp.cost(sol)
        assert sol_cost == cost, f'Incorrect cost of solution {sol}. '\
            + f'Expected: {cost}. Actual: {sol_cost}'


def test_lbound():
    TEST_CASES = [
        # test case: (weights, capacity, solution, expected lower bound
        # of solution)
        ([1, 2, 3], 3, ({1: {3}, 2: {1, 2}}, -1), 2),
        ([1, 2, 3], 3, ({1: {3}}, 0), 2),
        ([1, 2, 3], 3, ({1: {3}, 2: {2}}, 1), 2),
        ([1, 2, 3], 3, ({1: {3}, 2: {2, 1}}, 2), 2)
    ]
    for weights, capacity, sol, lb in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check lower bounds
        lbound = bpp.lbound(sol)
        assert lbound == lb, f'Incorrect lower bound for solution {sol}. '\
            + f'Expected: {lb}. Actual: {lbound}'


def test_branch():
    TEST_CASES = [
        # test case: (weights, capacity, expected list of branched solutions,
        # initial solution)
        ([1, 2, 3], 3, [({1: {3}, 2: {2}}, 1)], ({1: {3}}, 0)),
        ([7, 8, 2, 3], 15, [({1: {1, 2}}, 1), ({1: {2}, 2: {1}}, 1)],
         ({1: {2}}, 0)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16, [
            ({1: {6}, 2: {5, 10}, 3: {4}}, 3),
            ({1: {6}, 2: {5}, 3: {4, 10}}, 3),
            ({1: {6}, 2: {5}, 3: {4}, 4: {10}}, 3)
        ], ({1: {6}, 2: {5}, 3: {4}}, 2)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16, [({1: {6}}, 0)],
         ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, -1)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16, [({1: {6}, 2: {5}}, 1)],
         ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, 0))
    ]
    for weights, capacity, expected, init_sol in TEST_CASES:
        params = BinPackingParams(weights, capacity)
        bpp = BinPackingProblem(params)

        # check branched solutions
        new_sols = bpp.branch(init_sol)
        for new_sol in new_sols:
            assert new_sol in expected, 'Unexpected solution produced by '\
                + f'branching on solution {init_sol}: {new_sol}'
        for exp_sol in expected:
            assert exp_sol in new_sols, f'Expected solution {exp_sol} was '\
                + f'not produced by branching on solution {init_sol}'


def test_complete_solution():
    TEST_CASES = [
        # test case: (weights, capacity, incomplete solution, completed
        # solution)
        ([1, 2, 3], 3, ({1: {3}}, 0), ({1: {3}, 2: {1, 2}}, 0)),
        ([7, 8, 2, 3], 15, ({1: {2}}, 0), ({1: {2, 1}, 2: {3, 4}}, 0)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16,
         ({1: {6}, 2: {5}, 3: {4}}, 2),
         ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, 2)),
        ([1, 2, 3, 8, 9, 10, 4, 5, 6, 7], 16,
         (dict(), -1),
         ({1: {6, 9}, 2: {5, 10}, 3: {4, 8, 3}, 4: {7, 2, 1}}, -1))
    ]
    for weights, capacity, incomplete_sol, complete_sol in TEST_CASES:
        params = BinPackingParams(
            weights,
            capacity
        )
        bpp = BinPackingProblem(params)

        # check completed solution
        sol = bpp.complete_solution(incomplete_sol)
        assert sol == complete_sol, 'Incorrect completed solution created '\
            + f'from incomplete solution {incomplete_sol}. Expected: '\
            + f'{complete_sol}. Actual: {sol}'


def test_bnb_binpacking():
    # weights, capacity, min bins (optimal solution)
    TEST_CASES = [
        ([1, 2, 3], 3, 2),
        ([7, 8, 2, 3], 15, 2),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 12, 5),
        ([49, 41, 34, 33, 29, 26, 26, 22, 20, 19], 100, 3),
        ([49, 41, 34, 33, 29, 26, 26, 22, 20, 19] * 2, 100, 6)
    ]
    for weights, capacity, min_bins in TEST_CASES:
        for bnb_type in [0, 1]:
            params = BinPackingParams(weights, capacity)
            bpp = BinPackingProblem(params)
            init_cost = bpp.best_cost
            bpp.solve(1000, 100, 120, bnb_type)

            # check final solution
            check_bnb_sol(bpp, bnb_type, params)
            check_sol_vs_init_sol(bpp.best_cost, init_cost)

            # check if final solution was within 1.5 * optimal solution cost
            check_sol_optimality(bpp.best_cost, min_bins, 1.5)
