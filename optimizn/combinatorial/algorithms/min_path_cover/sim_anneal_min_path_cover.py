# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from optimizn.combinatorial.simulated_annealing import SimAnnealProblem
from graphing.special_graphs.neural_trigraph.rand_graph import *
# from graphing.graph import Graph
# from graphing.traversal.clr_traversal import Graph1
from graphing.special_graphs.neural_trigraph.path_cover import\
    complete_paths
from copy import deepcopy


# For demonstration purposes.
# We pick the min path cover problem 
# where we have an algorithm for computing
# the optimal solution.
class MinPathCover_NTG(SimAnnealProblem):
    """
    Finding the min path cover of a neural trigraph.
    """
    def __init__(self, ntg, swtch=1):
        self.ntg = ntg
        self.params = ntg
        self.adj = ntg.g1.adj
        self.swtch = swtch
        self.name = "MinPathCover_NeuralTriGraph"
        super().__init__()

    def get_candidate(self):
        """
        A candidate is going to be an array of
        arrays, where each array is a full path
        from the left-most layer of the graph
        to the right-most layer.
        """
        paths = []
        self.covered = {}
        ixs = np.arange(1, self.ntg.max_ix+1)
        ixs = np.random.permutation(ixs)
        for i in ixs:
            if i not in self.covered:
                path = self.add_path(i)
                paths.append(path[0])
        self.candidate = paths
        return paths

    def next_candidate_v2(self, candidate, num_del_paths=1):
        self.candidate = deepcopy(candidate)
        paths = self.candidate
        covered = deepcopy(self.covered)
        del_paths = []
        for i in range(num_del_paths):
            ix = np.random.choice(range(len(paths)))
            del_paths.append(paths[ix])
            paths = np.delete(paths, ix, 0)
        for del_path in del_paths:
            for ixx in del_path:
                covered[ixx] -= 1
                if covered[ixx] == 0:
                    path = complete_paths([[ixx]],
                        self.ntg.left_edges,
                        self.ntg.right_edges)
                    path = path[0]
                    for pa in path:
                        covered[pa] += 1
                        #breakpoint()
                    paths = np.concatenate((paths, [path]))
                    #breakpoint()
        return paths

    def next_candidate(self, candidate, num_del_paths=1):
        if self.swtch == 0:
            return self.get_candidate()
        else:
            return self.next_candidate_v2(candidate, num_del_paths)

    def add_path(self, i):
        path = complete_paths([[i]],
            self.ntg.left_edges, self.ntg.right_edges)
        for j in path[0]:
            if j in self.covered:
                self.covered[j] += 1
            else:
                self.covered[j] = 1
        # self.candidate.append(path)
        return path

    def cost(self, candidate):
        '''
        Gets the cost for candidate solution.
        '''
        return (len(candidate))

    def update_candidate(self, candidate, cost):
        # TODO: This can be made more efficient by updating existing covered
        # set.
        self.covered = {}
        for path in candidate:
            for j in path:
                if j in self.covered:
                    self.covered[j] += 1
                else:
                    self.covered[j] = 1
        super().update_candidate(candidate, cost)


# Scipy simulated annealing can't be used because it expects a 1-d continuous
# array
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html

