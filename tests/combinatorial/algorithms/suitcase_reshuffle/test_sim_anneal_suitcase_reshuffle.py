# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.algorithms.suitcase_reshuffle.suitcases\
    import SuitCases
from optimizn.combinatorial.algorithms.suitcase_reshuffle\
    .sim_anneal_suitcase_reshuffle import SuitCaseReshuffle
from tests.combinatorial.algorithms.check_sol_utils import check_sol,\
	check_sol_vs_init_sol


def test_constructor():
	config = [[7,5,1],[4,6,1]]
	sc = SuitCases(config)
	scr = SuitCaseReshuffle(params=sc)

	# check initial solution
	assert config == scr.best_solution, 'Incorrect initial solution. '\
		+ f'Expected: {config}. Actual: {scr.best_solution}'
	
	# check initial solution cost
	exp_init_sol_cost = -1
	assert exp_init_sol_cost == scr.best_cost, 'Incorrect initial solution '\
		+ f'cost. Expected {exp_init_sol_cost}. Actual: {scr.best_cost}'

	# check params
	exp_params = SuitCases(config)
	assert exp_params == scr.params, 'Incorrect parameters. Expected: '\
		+ f'{exp_params}. Actual: {scr.params}'


def test_sa_suitcasereshuffle():
	config = [[7,5,1],[4,6,1]]
	sc = SuitCases(config)
	scr = SuitCaseReshuffle(params=sc)
	init_cost = scr.best_cost
	scr.anneal()

	# check final solution
	check_sol_vs_init_sol(scr.best_cost, init_cost)
	exp_sols = [
		[[7, 6, 0], [4, 5, 2]],
		[[7, 6, 0], [5, 4, 2]],
		[[6, 7, 0], [4, 5, 2]],
		[[6, 7, 0], [5, 4, 2]],
		[[7, 4, 2], [6, 5, 0]],
		[[7, 4, 2], [5, 6, 0]],
		[[4, 7, 2], [6, 5, 0]],
		[[4, 7, 2], [5, 6, 0]],
		[[6, 5, 2], [4, 7, 0]],
		[[6, 5, 2], [7, 4, 0]],
		[[5, 6, 2], [4, 7, 0]],
		[[5, 6, 2], [7, 4, 0]]
	]
	check_sol(scr.best_solution, exp_sols)
