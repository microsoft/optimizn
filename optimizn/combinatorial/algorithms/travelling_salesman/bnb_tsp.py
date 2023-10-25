# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
from optimizn.combinatorial.branch_and_bound import BnBProblem


class TravellingSalesmanProblem(BnBProblem):
    def __init__(self, params):
        self.input_graph = params['input_graph']
        # sort all distance values, for computing lower bounds
        self.sorted_dists = []
        for i in range(self.input_graph.dists.shape[0]):
            for j in range(0, i):
                self.sorted_dists.append(self.input_graph.dists[i, j])
        self.sorted_dists.sort()
        super().__init__(params)

    def _get_closest_city(self, city, visited):
        # get the unvisited city closest to the one provided
        min_city = None
        min_dist = float('inf')
        dists = self.input_graph.dists[city]
        for i in range(len(dists)):
            if i != city and i not in visited and dists[i] < min_dist:
                min_city = i
                min_dist = dists[i]
        return min_city

    def _complete_path(self, path):
        # complete the path greedily, iteratively adding the unvisited city
        # closest to the last city in the accumulated path
        visited = set(path)
        complete_path = deepcopy(path)
        while len(complete_path) != self.input_graph.dists.shape[0]:
            if len(complete_path) == 0:
                next_city = 0
            else:
                last_city_idx = 0 if len(complete_path) == 0 else\
                    complete_path[-1]
                next_city = self._get_closest_city(last_city_idx, visited)
            visited.add(next_city)
            complete_path.append(next_city)
        return complete_path

    def get_candidate(self):
        # greedily assemble a path from scratch
        # solution format is 2-tuple, first element is the path itself and
        # the second element is the index of the last confirmed city (last
        # confirmed index), which is used for branching
        return (self._complete_path([]), -1)

    def complete_solution(self, sol):
        # greedily complete the path using the remaining/unvisited cities
        return (self._complete_path(sol[0]), sol[1])

    def cost(self, sol):
        # sum of distances between adjacent cities in path, and from last
        # city to first city in path
        path = sol[0]
        path_cost = 0
        for i in range(self.input_graph.num_cities - 1):
            path_cost += self.input_graph.dists[path[i], path[i + 1]]
        path_cost += self.input_graph.dists[
            path[self.input_graph.num_cities - 1], path[0]]
        return path_cost

    def lbound(self, sol):
        # sum of distances between confirmed cities and smallest distances
        # to account for remaining cities and start city
        path = sol[0]
        last_confirmed_idx = sol[1]
        lb_path_cost = 0
        for i in range(last_confirmed_idx):
            lb_path_cost += self.input_graph.dists[path[i], path[i + 1]]
        if last_confirmed_idx + 1 == self.input_graph.num_cities:
            lb_path_cost += self.input_graph.dists[
                path[last_confirmed_idx], path[0]]
        else:
            lb_path_cost += sum(self.sorted_dists[
                :self.input_graph.num_cities - last_confirmed_idx])
        return lb_path_cost

    def is_complete(self, sol):
        # check that all cities covered once, path length is equal to the
        # number of cities
        path = sol[0]
        check_all_cities_covered = set(path) == set(
            range(self.input_graph.num_cities))
        check_cities_covered_once = len(path) == len(set(path))
        check_path_length = len(path) == self.input_graph.num_cities
        return (check_path_length and check_cities_covered_once and
                check_all_cities_covered)

    def is_feasible(self, sol):
        # check that covered cities are valid, covered cities are only covered
        # once, path length is less than or equal to the number of cities, and
        # last confirmed index is valid
        path = sol[0]
        last_confirmed_idx = sol[1]
        check_covered_cities = len(set(path).difference(
                set(range(self.input_graph.num_cities)))) == 0
        check_cities_covered_once = len(path) == len(set(path))
        check_path_length = len(path) <= self.input_graph.num_cities
        check_last_confirmed_index = last_confirmed_idx < len(path)\
            and last_confirmed_idx >= -1
        return (check_covered_cities and check_cities_covered_once and
                check_path_length and check_last_confirmed_index)

    def branch(self, sol):
        # build the path from the last confirmed city, by creating a new
        # solution where each uncovered city is the next confirmed city
        path = sol[0]
        last_confirmed_idx = sol[1]
        if last_confirmed_idx >= self.input_graph.num_cities - 1:
            return []
        visited = set(path[:last_confirmed_idx + 1])
        new_sols = []
        for new_city in range(self.input_graph.dists.shape[0]):
            if new_city not in visited:
                new_sols.append((path[:last_confirmed_idx + 1] + [new_city],
                                 last_confirmed_idx + 1))
        return new_sols
