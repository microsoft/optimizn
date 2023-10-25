# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def check_bnb_sol(bnb_instance, bnb_type, params):
    # determine BnB type
    if bnb_type == 0:
        bnb_alg = 'traditional'
    else:
        bnb_alg = 'modified'

    # check that final solution is complete and feasible
    assert bnb_instance.is_complete(bnb_instance.best_solution), 'Final '\
        + f'solution ({bnb_instance.best_solution}) is not complete. '\
        + f'Algorithm: {bnb_alg} branch and bound. Params: {params}'
    assert bnb_instance.is_feasible(bnb_instance.best_solution), 'Final '\
        + f'solution ({bnb_instance.best_solution}) is not feasible. '\
        + f'Algorithm: {bnb_alg} branch and bound. Params: {params}'


def check_sol_vs_init_sol(best_cost, init_cost):
    # check that final solution is not worse than initial solution
    assert best_cost <= init_cost, 'Final solution is less '\
        + f'optimal than initial solution. Cost of initial solution: '\
        + f'{init_cost}. Cost of final solution: {best_cost} '


def check_sol(sol, exp_sols):
    assert sol in exp_sols, 'Incorrect final solution. '\
    + f'Expected one of the following: {exp_sols}. Actual: {sol}'


def check_sol_optimality(sol_cost, opt_sol_cost, ratio=1.0):
    assert sol_cost <= opt_sol_cost * ratio, 'Final solution cost '\
        + f'({sol_cost}) is greater than {ratio} * '\
        + f'optimal solution cost, where optimal solution cost '\
        + f'= {opt_sol_cost}'
