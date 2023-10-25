# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from optimizn.combinatorial.branch_and_bound import BnBProblem


class MinPathCoverParams:
    def __init__(self, edges1, edges2):
        self.edges1 = edges1
        self.edges2 = edges2

    def __eq__(self, other):
        return (
            other is not None
            and self.edges1 == other.edges1
            and self.edges2 == other.edges2
        )


class MinPathCoverProblem1(BnBProblem):
    '''
    Solution format:
    1. Array of size p (p = number of complete paths in graph), values are 0
    if path is not used in cover, 1 if path is used in cover
    2. Last considered path index

    Branching strategy:
    Each level of the solution space tree corresponds to a path.
    Each solution branches into at most two solutions (the path is either used
    or omitted from the path cover). If the path does not cover any new
    vertices, then the branching only produces the solution where the path
    is omitted from the cover
    '''
    def __init__(self, params):
        self.edges1 = params.edges1
        self.edges2 = params.edges2
        self.vertices = set(params.edges1.flatten()).union(
            set(params.edges2.flatten()))
        self._get_all_paths()
        super().__init__(params)

    def get_candidate(self):
        return (np.ones(len(self.all_paths)), -1)

    def _get_all_paths(self):
        self.all_paths = []
        for u1, u2 in self.edges1:
            for v1, v2 in self.edges2:
                if u2 == v1:
                    self.all_paths.append([u1, v1, v2])
        self.all_paths = np.array(self.all_paths)

    def _pick_rem_paths(self, sol):
        # determine which vertices can still be covered
        choices = sol[0]
        last_idx = sol[1] + 1
        covered = set()
        for i in range(0, last_idx):
            if choices[i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        to_cover = set(self.all_paths[last_idx:].flatten()).difference(covered)

        # select paths that cover the most of the remaining vertices
        new_path_idxs = []
        new_covered = set()
        while len(to_cover.difference(new_covered)) != 0:
            path_idx = None
            path_covered = set()
            for i in range(last_idx, len(self.all_paths)):
                new_verts = set(self.all_paths[i]).intersection(
                    to_cover.difference(new_covered))
                if len(new_verts) > len(path_covered):
                    path_idx = i
                    path_covered = new_verts
            new_path_idxs.append(path_idx)
            new_covered = new_covered.union(path_covered)
        return new_path_idxs

    def complete_solution(self, sol):
        path_cover = list(sol[0])[:max(0, sol[1] + 1)]
        for _ in range(len(self.all_paths) - len(sol[0])):
            path_cover.append(0)
        for idx in self._pick_rem_paths(sol):
            path_cover[idx] = 1
        return (np.array(path_cover), sol[1])

    def lbound(self, sol):
        # sum of existing paths and (number of vertices left to cover / 3) 
        choices = sol[0]
        last_idx = sol[1] + 1
        covered = set()
        for i in range(0, last_idx):
            if choices[i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        return sum(choices[0: sol[1] + 1]) + len(self.vertices.difference(
            covered)) / 3

    def cost(self, sol):
        return sum(sol[0])

    def branch(self, sol):
        new_sols = []
        if sol[1] + 1 >= len(self.all_paths):
            return new_sols
        for val in [0, 1]:
            # do not include path in cover if no new vertices are covered
            if val == 1:
                covered = set()
                for i in range(0, sol[1] + 1):
                    for v in self.all_paths[i]:
                        covered.add(v)
                if len(set(self.all_paths[sol[1] + 1])
                        .difference(covered)) == 0:
                    continue
            new_sol = np.array(list(sol[0][:max(0, sol[1] + 1)]) + [val])
            new_sols.append((new_sol, sol[1] + 1))
        return new_sols

    def is_sol(self, sol):
        covered = set()
        for i in range(len(sol[0])):
            if sol[0][i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        return covered == self.vertices

    def is_complete(self, sol):
        # check length of solution
        check_length = len(sol[0]) == len(self.all_paths)
    
        # check that all vertices are covered
        covered = set()
        for i in range(len(sol[0])):
            if sol[0][i] == 1:
                covered = covered.union(set(self.all_paths[i]))
        check_coverage = covered == self.vertices
    
        return check_length and check_coverage

    def is_feasible(self, sol):
        # check that solution is not longer than list of paths
        check_length = len(sol[0]) <= len(self.all_paths)

        # check that all values in solution are 0 or 1
        check_vals = len(set(sol[0]).difference({0, 1})) == 0

        # check that remaining cliques are enough to cover the uncovered
        # vertices
        covered_verts = set()
        for i in range(len(sol[0])):
            if sol[0][i] == 1:
                covered_verts = covered_verts.union(set(self.all_paths[i]))
        uncovered_verts = self.vertices.difference(covered_verts)
        coverable_verts = set()
        for i in range(sol[1] + 1, len(self.all_paths)):
            coverable_verts = coverable_verts.union(set(self.all_paths[i]))
        uncoverable_verts = uncovered_verts.difference(coverable_verts)
        check_coverage = len(uncoverable_verts) == 0

        return check_length and check_vals and check_coverage


class MinPathCoverProblem2(BnBProblem):
    '''
    Solution format:
    1. Path cover (paths that cover the first vertex to the last covered
    vertex)
    2. Remaining paths (paths that cover the vertices after the last covered
    vertex)
    3. Last covered vertex

    Branching strategy:
    At a level of the solution space tree that corresponds to vertex X,
    the solution nodes correspond to path covers that cover vertex X.
    Branching any of these solutions produces a new set of solutions that
    correspond to path covers that cover vertex X+1. These path covers
    either remain the same (if the path cover already covered vertex X+1)
    or include one extra path (to cover vertex X+1).
    '''
    def __init__(self, params):
        self.edges1 = params.edges1
        self.edges2 = params.edges2
        self.vertices = set(params.edges1.flatten()).union(
            set(params.edges2.flatten()))
        self.all_paths = []
        self.cov_dict = {}
        for u1, u2 in self.edges1:
            for v1, v2 in self.edges2:
                path = (u1, v1, v2)
                if u2 == v1:
                    self.all_paths.append(path)
                    for vert in path:
                        if vert not in self.cov_dict.keys():
                            self.cov_dict[vert] = set()
                        self.cov_dict[vert].add(path)
        super().__init__(params)

    def get_candidate(self):
        return (np.zeros((0, 3)), np.array(self.all_paths),
                min(self.vertices) - 1)

    def lbound(self, sol):
        # sum of existing paths and (number of vertices left to cover / 3)
        path_cover = sol[0]
        rem_verts = self.vertices.difference(set(path_cover.flatten()))
        return len(path_cover) + (len(rem_verts) / 3)

    def cost(self, sol):
        path_cover = sol[0]
        rem_paths = sol[1]
        return len(path_cover) + len(rem_paths)

    def branch(self, sol):
        # get components of solution
        path_cover = sol[0]
        last_cov_vert = sol[2]

        # if next vertex to cover has already been covered, retain
        # solution and cover the vertex after that
        new_last_cov_vert = last_cov_vert + 1
        if new_last_cov_vert > len(self.vertices):
            return []
        new_sols = []
        covered = set(path_cover.flatten())
        if new_last_cov_vert in covered:
            new_sols.append((sol[0], sol[1], new_last_cov_vert))
        # otherwise, branch based on paths that can cover the next vertex,
        # complete solution by picking paths that greedily cover remaining
        # vertices
        else:
            cand_paths = np.array(list(self.cov_dict[new_last_cov_vert]))
            for cand_path in cand_paths:
                cand_path = np.array([cand_path])
                new_path_cover = np.concatenate(
                    (path_cover, cand_path), axis=0)
                new_sols.append((new_path_cover, np.zeros((0, 3)),
                                 new_last_cov_vert))
        return new_sols

    def complete_solution(self, sol):
        new_rem_paths = []
        rem_verts = self.vertices.difference(
            set(sol[0].flatten()).union(
                set(np.array(new_rem_paths).flatten())))
        while len(rem_verts) > 0:
            opt_path = None
            opt_cov = {}
            for path in self.all_paths:
                cov = rem_verts.intersection(path)
                if len(cov) > len(opt_cov):
                    opt_path = path
                    opt_cov = rem_verts.intersection(path)
            rem_verts = rem_verts.difference(set(opt_path))
            new_rem_paths.append(opt_path)
        new_rem_paths = np.array(new_rem_paths)
        return (sol[0], new_rem_paths, sol[2])
    
    def is_complete(self, sol):    
        # check that all vertices are covered
        path_cover = sol[0]
        rem_paths = sol[1]
        check_coverage = self.vertices == set(path_cover.flatten()).union(
            set(rem_paths.flatten()))
    
        return check_coverage

    def is_feasible(self, sol):
        # check that each path in solution is valid
        path_cover_set = set(map(lambda p: tuple(p.astype(int)), sol[0]))
        all_paths_set = set(self.all_paths)
        check_paths_valid = len(path_cover_set.difference(all_paths_set)) == 0
        
        # check that remaining paths are enough to cover the uncovered
        # vertices
        covered_verts = set(sol[0].flatten())
        uncovered_verts = self.vertices.difference(covered_verts)
        rem_paths = all_paths_set.difference(path_cover_set)
        coverable_verts = set()
        for path in rem_paths:
            coverable_verts = coverable_verts.union(set(path))
        check_coverage = len(uncovered_verts.difference(coverable_verts)) == 0

        return check_paths_valid and check_coverage
