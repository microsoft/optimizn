# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from graphing.special_graphs.neural_trigraph.neural_trigraph\
    import NeuralTriGraph
from graphing.special_graphs.neural_trigraph.rand_graph import rep_graph
from graphing.special_graphs.neural_trigraph.path_cover import \
    min_cover_trigraph
import os
from optimizn.combinatorial.algorithms.min_path_cover\
    .sim_anneal_min_path_cover import MinPathCover_NTG
from tests.combinatorial.algorithms.check_sol_utils import\
    check_sol_optimality, check_sol_vs_init_sol


def test_sa_minpathcover1(edges1=None, edges2=None, n_iter=20000, swtch=1):
    if edges1 is None:
        edges1, edges2 = rep_graph(8, 10, 14, reps=4)
    
    # get optimal solution
    opt_paths = min_cover_trigraph(edges1, edges2)
    opt_sol_cost = len(opt_paths)

    # get simulated annealing solution
    ntg = NeuralTriGraph(edges1, edges2)
    mpc = MinPathCover_NTG(ntg, swtch=swtch)
    init_cost = mpc.best_cost
    mpc.anneal(n_iter)

    # check optimality of simulated annealing solution
    check_sol_vs_init_sol(mpc.best_cost, init_cost)
    check_sol_optimality(mpc.best_cost, opt_sol_cost, 1.2)


def test_sa_minpathcover2(n_iter=20000, swtch=1):
    # read edges from file
    dirname = os.path.dirname(__file__)
    edges1_path = os.path.join(dirname, './edges1.csv')
    edges2_path = os.path.join(dirname, './edges2.csv')
    edges1 = np.loadtxt(edges1_path)
    edges1 = edges1.astype(int)
    edges2 = np.loadtxt(edges2_path)
    edges2 = edges2.astype(int)

    # test with edges read from file
    test_sa_minpathcover1(edges1, edges2, n_iter, swtch=swtch)
