# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from copy import deepcopy
from optimizn.combinatorial.simulated_annealing import SimAnnealProblem
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp


class TravSalsmn(SimAnnealProblem):
    '''
    This simulated annealing implementation for the traveling salesman
    problem is based on the following sources. The code presented in [2, 3] is
    licensed under the MIT License. The original license texts are shown in the
    NOTICE.md file.
        
    Sources:

    [1] T. W. Schneider, "The traveling salesman with simulated annealing,
    r, and shiny."
    https://toddwschneider.com/posts/traveling-salesman-with-simulated-annealing-r-and-shiny/,
    September 2014. Online; accessed 8-January-2023.

    [2] T. W. Schneider, "shiny-salesman/helpers.r."
    https://github.com/toddwschneider/shiny-salesman/blob/master/helpers.R,
    October 2014. Online; accessed 8-January-2023.

    [3] F. Goulart, T. Frick, and Luan, "python-tsp/python_tsp/heuristics/
    simulated_annealing.py."
    https://github.com/fillipe-gsm/python-tsp/blob/master/python_tsp/heuristics/simulated_annealing.py.
    Online; accessed 27-March-2023.
    '''
    def __init__(self, params, temp_reduce_factor=0.99):
        super().__init__(params)
        self.temp_reduce_factor = temp_reduce_factor
        
        # set initial temperature
        cost_diffs = []
        for _ in range(100):
            new_path = self.next_candidate(self.candidate)
            cost_diffs.append(self.cost(new_path) - self.best_cost)
        self.temperature = -1 * abs(
            sum(cost_diffs) / len(cost_diffs)) / np.log(0.5)

    def get_initial_solution(self):
        """
        A candidate is going to be an array
        representing the order of cities
        visited.
        """
        # path of cities in increasing, numerical order
        return np.arange(self.params.num_cities)

    def reset_candidate(self):
        # random path of cities
        return np.random.permutation(np.arange(self.params.num_cities))

    def cost(self, candidate):
        tour_d = 0
        for i in range(1, len(candidate)):
            tour_d += self.params.dists[candidate[i], candidate[i-1]]
        tour_d += self.params.dists[
            candidate[0], candidate[len(candidate) - 1]]
        return tour_d

    def next_candidate(self, candidate):
        nu_candidate = deepcopy(candidate)
        swaps = np.random.choice(
            np.arange(len(candidate)), size=2, replace=False)
        to_swap = nu_candidate[swaps]
        nu_candidate[swaps[0]] = to_swap[1]
        nu_candidate[swaps[1]] = to_swap[0]
        return nu_candidate
    
    def get_temperature(self, iters):
        return self.temperature * self.temp_reduce_factor


def dist_from_lat_long(lat1, long1, lat2, long2):
    """
    This was taken from the following source.

    Source:
    
    (1)
    Title: Latitude Longitude Distance Calculator
    Author: Luciano Miño
    Reviewer: Steven Wooding
    URL: https://www.omnicalculator.com/other/latitude-longitude-distance
    Date accessed: January 8, 2023

    Doesn't currently work. Need to debug (230108)
    """
    theta1 = lat1
    theta2 = lat2
    phi1 = long1
    phi2 = long2
    r = 6400
    dtheta1 = (theta2-theta1)/2
    dtheta1 = np.sin(dtheta1)**2
    dtheta2 = np.cos(theta1)*np.cos(theta1)
    dtheta2 *= np.sin((phi2-phi1)/2)**2
    d = np.sqrt(dtheta1+dtheta2)
    d = np.arcsin(d)
    dist = 2*r*d
    return d
