# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from numpy.random import uniform
# from numpy import e
import numpy as np
from copy import deepcopy
from optimizn.combinatorial.opt_problem import OptProblem
import time
import warnings


class SimAnnealProblem(OptProblem):
    def __init__(self, logger=None):
        ''' Initialize the problem '''
        super().__init__(logger)
        self.candidate = make_copy(self.init_solution)
        self.current_cost = make_copy(self.init_cost)

    def next_candidate(self):
        ''' Switch to the next candidate.'''
        raise NotImplementedError(
            "Implement a function to produce the next candidate solution "
            + "from the current candidate solution")

    def reset_candidate(self):
        '''
        Returns a new solution for when the candidate solution is reset.
        Defaults to get_initial_solution but can be overridden if needed
        '''
        return self.get_initial_solution()
    
    def get_temperature(self, iters):
        '''
        Calculates the temperature based on a given number of iterations.
        Defaults to current_temperature
        '''
        return current_temperature(iters)

    def anneal(self, n_iter=100000, reset_p=1/10000, time_limit=3600,
               log_iters=10000):
        '''
        This simulated annealing algorithm is based on the following
        sources. The code presented in source [2] is licensed under the MIT
        License. The original license text is shown in the NOTICE.md file.

        Sources:

        [1] T. W. Schneider, "The traveling salesman with simulated annealing,
        r, and shiny."
        https://toddwschneider.com/posts/traveling-salesman-with-simulated-annealing-r-and-shiny/,
        September 2014. Online; accessed 8-January-2023.

        [2] T. W. Schneider, "shiny-salesman/helpers.r."
        https://github.com/toddwschneider/shiny-salesman/blob/master/helpers.R,
        October 2014. Online; accessed 8-January-2023.

        [3] R. A. Rutenbar, "Simulated annealing algorithms: An overview," IEEE
        Circuits and Devices Magazine, vol. 5, pp. 19-26, January 1989.
        https://www.cs.amherst.edu/~ccmcgeoch/cs34/papers/rutenbar.pdf. Online;
        accessed 8-January-2024.
        '''
        reset = False
        j = -1
        start = time.time()
        for i in range(n_iter):
            # check if time limit exceeded
            if time.time() - start > time_limit:
                self.logger.info('Time limit exceeded, terminating algorithm')
                self.logger.info('Best solution: ' + str(self.best_cost))
                break
            j = j + 1
            self.temperature = self.get_temperature(j)
            if i % log_iters == 0:
                self.logger.info(
                    "Iteration: " + str(i) + " Current best solution: "
                    + str(self.best_cost))
            # eps = 0.3 * e**(-i/n_iter)
            if np.random.uniform() < reset_p:
                self.logger.info("Resetting candidate solution.")
                self.new_candidate = self.reset_candidate()
                self.new_cost = self.cost(self.new_candidate)
                self.logger.info("with cost: " + str(self.new_cost))
                j = 0
                reset = True
            else:
                self.new_candidate = self.next_candidate(self.candidate)
                self.new_cost = self.cost(self.new_candidate)
            cost_del = self.cost_delta(self.new_cost, self.current_cost)
            # treat runtime warnings like errors, to catch overflow warnings
            warnings.filterwarnings(
                "error", category=RuntimeWarning)
            try:
                # see if overflow occurs
                eps = np.exp(-1 * cost_del / self.temperature)
            except RuntimeWarning:
                # overflow occurred

                # if cost delta and temperature have the same sign, then
                # eps will be very close to 0, so eps is set to 0
                if (cost_del > 0 and self.temperature > 0) or\
                        (cost_del < 0 and self.temperature < 0):
                    eps = 0
                # if cost delta and temperature have opposite signs,
                # then eps will be very large (larger than any value sampled
                # from uniform distribution on [0, 1)), so eps is set to 1
                else:
                    eps = 1
            # reset warnings
            warnings.resetwarnings()

            if cost_del < 0 or uniform() < eps or reset:
                self.update_candidate(self.new_candidate, self.new_cost)
                if reset:
                    reset = False
            if self.cost_delta(self.new_cost, self.best_cost) < 0:
                self.update_best(self.new_candidate, self.new_cost)
                self.logger.info("Best cost updated to:" + str(self.new_cost))

    def update_candidate(self, candidate, cost):
        self.candidate = make_copy(candidate)
        self.current_cost = make_copy(cost)

    def update_best(self, candidate, cost):
        self.best_solution = make_copy(candidate)
        self.best_cost = make_copy(cost)


def make_copy(candidate):
    return deepcopy(candidate)


def s_curve(x, center, width):
    return 1 / (1 + np.exp((x - center) / width))


def current_temperature(iter, s_curve_amplitude=4000,
                        s_curve_center=0, s_curve_width=3000):
    return s_curve_amplitude * s_curve(iter, s_curve_center, s_curve_width)
