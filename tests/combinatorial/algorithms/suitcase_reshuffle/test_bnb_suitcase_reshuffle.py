# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.suitcase_reshuffle.bnb_suitcasereshuffle\
    import SuitcaseReshuffleProblem
from copy import deepcopy
from optimizn.combinatorial.algorithms.suitcase_reshuffle.suitcases\
    import SuitCases
from tests.combinatorial.algorithms.check_sol_utils import check_bnb_sol,\
    check_sol_optimality, check_sol_vs_init_sol


def test_constructor_get_root():
    TEST_CASES = [
        # test case: (suitcase configuration, expected suitcase capacities,
        # expected cost of initial solution)
        ([[7, 5, 1], [4, 6, 1]], [13, 11], -1),
        ([[7, 5, 0], [4, 6, 2]], [12, 12], -2),
        ([[7, 5, 1, 0], [4, 6, 0]], [13, 10], 0)
    ]
    for config, capacities, cost in TEST_CASES:
        srp = SuitcaseReshuffleProblem(SuitCases(config))
        init_sol = srp.best_solution

        # check config
        init_config = init_sol[0].config
        assert init_config == config, 'Incorrect initial solution configs. '\
            + f'Expected: {config}. Actual: {init_config}'
        
        # check capacities
        init_caps = init_sol[0].capacities
        assert init_caps == capacities, 'Incorrect initial solution '\
            + f'capacities. Expected: {capacities}. Actual: {init_caps}'
    
        # check initial solution
        init_suitcase_num = init_sol[1]
        assert srp.best_cost == cost, 'Incorrect initial solution cost. '\
            + f'Expected: {cost}. Actual: {srp.best_cost}'
        assert init_suitcase_num == -1, 'Incorrect suitcase number '\
            + 'in initial solution. Expected: -1. Actual: '\
            + f'{init_suitcase_num}'
        
        # check root node solution
        root_sol = srp.get_root()
        root_sol_cost = srp.cost(root_sol)
        assert root_sol_cost == cost, 'Incorrect root node solution cost. '\
            + f'Expected: {cost}. Actual: {root_sol_cost}'
        assert root_sol[1] == -1, 'Incorrect suitcase number (for '\
            + 'branching) in root node solution. Expected: -1. Actual: '\
            + f'{root_sol[1]}'


def test_cost():
    TEST_CASES = [
        # test case: (solution, expected cost)
        ((SuitCases([[7, 5, 1], [4, 6, 1]]), 0), -1),
        ((SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]), 1), -4),
        ((SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]), 2), -4)
    ]
    for sol, cost in TEST_CASES:
        srp = SuitcaseReshuffleProblem(sol[0])
        sol_cost = srp.cost(sol)
        assert sol_cost == cost, f'Computed cost of solution {sol} is '\
            + f'incorrect. Expected: {cost}. Actual: {sol_cost}'


def test_lbound():
    TEST_CASES = [
        # test case: (solution, expected lower bound)
        ((SuitCases([[7, 5, 1], [4, 6, 1]]), 0), -2),
        ((SuitCases([[7, 5, 3], [4, 6, 1]]), 0), -4),
        ((SuitCases([[7, 5, 1], [4, 6, 4]]), 0), -5)
    ]
    for sol, lbound in TEST_CASES:
        srp = SuitcaseReshuffleProblem(sol[0])
        sol_lb = srp.lbound(sol)
        assert sol_lb == lbound, f'Computed cost of solution {sol} is '\
            + f'incorrect. Expected: {lbound}. Actual: {sol_lb}'


def test_is_valid():
    TEST_CASES = [
        # test case: (initial suitcases, solution, boolean for whether solution
        # is complete)
        (SuitCases([[7, 5, 1], [4, 6, 1]]),
         (SuitCases([[7, 5, 1], [4, 6, 1]]), 0), True),
        (SuitCases([[7, 5, 1], [4, 6, 1]]),
         (SuitCases([[7, 6], [11]]), 0), False
         # not complete solution since item with weight 5 is not in a suitcase
         ),
        (SuitCases([[7, 5, 1], [4, 6, 1]]),
         (SuitCases([[7, 6], [11]]), 2), False
         # not complete solution since item with weight 5 is not in a suitcase
         ),
        (SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]),
         (SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]), 0),
         True),
        (SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]),
         (SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]), 4),
         True),
        (SuitCases([[7, 5, 1], [4, 6, 1], [12, 12, 4], [11, 10, 2]]),
         (SuitCases([[7, 6], [6, 4], [12, 12, 4], [11, 10, 2]]), 5),
         False
         # not complete solution since items with weights 4 and 5 are not in
         # suitcases
         )
    ]
    for init_sc, sol, valid_sol in TEST_CASES:
        srp = SuitcaseReshuffleProblem(init_sc)

        # check validity of solution
        is_valid = srp.is_valid(sol)
        assert valid_sol == is_valid, 'Validity check of solution '\
            + f'{sol} failed. Expected: {valid_sol}. Actual: {is_valid}'


def test_complete_solution():
    TEST_CASES = [
        # test case: (suitcase configuration, solution, expected 
        # complete solution)
        (SuitCases([[7, 5, 1], [4, 6, 1]]),
         (SuitCases([[7, 6], [11]]), 0),
         (SuitCases([[7, 6, 0], [5, 4, 2]]), 0)
         ),
        (SuitCases([[7, 5, 1], [4, 6, 1]]),
         (SuitCases([[7, 5, 1], [4, 6, 1]]), 3),
         (SuitCases([[7, 5, 1], [4, 6, 1]]), 3)
         ),
        (SuitCases([[7, 5, 1], [4, 6, 1], [3, 2, 1]]),
         (SuitCases([[7, 6, 0], [5, 6], [6]]), 2),
         (SuitCases([[7, 6, 0], [5, 4, 2, 0], [3, 3]]), 2)
         ),
        (SuitCases([[7, 5, 1], [4, 6, 1], [3, 2, 1]]),
         (SuitCases([[7, 5, 1], [4, 6, 1], [3, 2, 1]]), 5),
         (SuitCases([[7, 5, 1], [4, 6, 1], [3, 2, 1]]), 5)
         )
    ]
    for suitcases, sol, exp_comp_sol in TEST_CASES:
        srp = SuitcaseReshuffleProblem(suitcases)

        # check completed solution
        comp_sol = srp.complete_solution(sol)
        assert comp_sol == exp_comp_sol, 'Incorrect complete solution formed '\
            + f'from solution {sol}. Expected: {exp_comp_sol}. Actual: '\
            + f'{comp_sol}'


def test_branch():
    TEST_CASES = [
        # test case: (suitcase configuration, solution, expected branch
        # solutions)
        (
            [[7, 5, 1], [4, 6, 1]],
            (SuitCases([[7, 5, 1], [4, 6, 1]]), -1),
            [
                (SuitCases([[7, 6], [11]]), 0),
                (SuitCases([[13], [7, 4]]), 0)
            ]
        ),
        (
            [[7, 5, 1], [4, 6, 1]],
            (SuitCases([[7, 6], [11]]), 0),
            [
                (SuitCases([[7, 6, 0], [11]]), 1),
                (SuitCases([[7, 6], [6, 5]]), 1)
            ]
        ),
        (
            [[7, 5, 1], [4, 6, 1], [3, 2, 1]],
            (SuitCases([[7, 5, 1], [4, 6, 1], [3, 2, 1]]), -1),
            [
                (SuitCases([[7, 6], [11], [6]]), 0),
                (SuitCases([[13], [7, 4], [6]]), 0)
            ]
        ),
        (
            [[7, 5, 1], [4, 6, 1], [3, 2, 1]],
            (SuitCases([[13], [7, 4], [6]]), 0),
            [
                (SuitCases([[6, 7], [7, 4], [6]]), 1),
                (SuitCases([[13], [7, 4], [6, 0]]), 1)
            ]
        )
    ]
    for config, sol, branch_sols in TEST_CASES:
        srp = SuitcaseReshuffleProblem(SuitCases(config))

        # branch on solutions, check branched solutions
        new_sols = srp.branch(sol)
        assert new_sols == branch_sols, 'Incorrect branched solutions. '\
            + f'Expected: {branch_sols}. Actual: {new_sols}'


def test_bnb_suitcasereshuffle():
    TEST_CASES = [
        # test case: (suitcase configuration, optimal solution cost)
        ([[7, 5, 1], [4, 6, 1]], -2),
        ([[7, 5, 4], [4, 6, 1], [5, 5, 1]], -6),
        ([[1, 4, 3, 6, 4, 2], [2, 4, 7, 1, 0], [1, 7, 3, 8, 3, 4]], -6),
        ([
            [12, 52, 34, 23, 17, 18, 22, 10],
            [100, 21, 36, 77, 82, 44, 40],
            [1, 5, 2, 8, 22, 34, 50]
        ], -100)
    ]
    for config, opt_sol_cost in TEST_CASES:
        for bnb_type in [0, 1]:
            sc = SuitCases(config)
            params = {
                'init_sol': sc
            }
            srp = SuitcaseReshuffleProblem(sc)
            init_cost = srp.best_cost
            srp.solve(1000, 100, 120, bnb_type)

            # check final solution
            check_bnb_sol(srp, bnb_type, params)
            check_sol_vs_init_sol(srp.best_cost, init_cost)

            # check final solution optimality, if modified branch and bound
            # is used
            if bnb_type == 1:
                check_sol_optimality(srp.best_cost, opt_sol_cost, 9/10)
