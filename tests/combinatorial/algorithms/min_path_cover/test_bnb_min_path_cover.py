# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import numpy as np
from graphing.special_graphs.neural_trigraph.rand_graph import rep_graph
from graphing.special_graphs.neural_trigraph.path_cover import \
    min_cover_trigraph
from optimizn.combinatorial.algorithms.min_path_cover.bnb_min_path_cover\
    import MinPathCoverParams, MinPathCoverProblem1, MinPathCoverProblem2
from tests.combinatorial.algorithms.check_sol_utils import check_bnb_sol,\
    check_sol_optimality, check_sol_vs_init_sol


def test_bnb_minpathcover():
    EDGES = [
        (
            np.array([[1, 4], [2, 4], [2, 5], [3, 5]]),
            np.array([[4, 6], [4, 7], [5, 8]])
        ),
        rep_graph(8, 10, 14, reps=4),
        rep_graph(10, 14, 10, reps=4)
    ]
    TEST_CASES = [
        # test case: (edges, length of min path cover)
        (EDGES[0], 3),
        (EDGES[1], len(min_cover_trigraph(EDGES[1][0], EDGES[1][1]))),
        (EDGES[2], len(min_cover_trigraph(EDGES[2][0], EDGES[2][1])))
    ]
    for edges, mpc_len in TEST_CASES:
        for bnb_type in [0, 1]:
            edges1 = edges[0]
            edges2 = edges[1]

            # test min path cover algorithms
            params = MinPathCoverParams(edges1, edges2)
            mpc1 = MinPathCoverProblem1(params)
            mpc2 = MinPathCoverProblem2(params)
            for mpc in [mpc1, mpc2]:
                init_cost = mpc.best_cost
                mpc.solve(1000, 100, 120, bnb_type)

                # check final solutoin solution 
                check_bnb_sol(mpc, bnb_type, params)
                check_sol_vs_init_sol(mpc.best_cost, init_cost)

                # check final solution optimality if modified branch and bound
                # is used
                if bnb_type == 1:
                    check_sol_optimality(mpc.best_cost, mpc_len, 1.1)
