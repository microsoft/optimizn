# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from python_tsp.exact.branch_and_bound.solver import solve_tsp_branch_and_bound


def run_python_tsp_bnb(q, dists):
    _, opt_dist = solve_tsp_branch_and_bound(dists)
    q.put(opt_dist)
