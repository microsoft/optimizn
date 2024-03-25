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
    def __init__(self, params, logger=None):
        ''' Initialize the problem '''
        super().__init__(params, logger)
        self.candidate = self._make_copy(self.init_solution)
        self.current_cost = self._make_copy(self.init_cost)
        self.total_iters = 0
        self.iters_since_reset = -1
        self.total_time_elapsed = 0

    def next_candidate(self, candidate):
        ''' Switch to the next candidate, given the current candidate. '''
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
        Defaults to s_curve, can be overridden
        '''
        return s_curve(iters)

    def _log_results(self, iters, time_elapsed):
        self.logger.info("Iterations (total): " + str(self.total_iters))
        self.logger.info("Iterations (current): " + str(iters))
        self.logger.info("Time elapsed (total): "
                         + str(self.total_time_elapsed) + " seconds")
        self.logger.info("Time elapsed (current): " + str(time_elapsed)
                         + " seconds")
        self.logger.info("Best solution: " + str(self.best_solution))
        self.logger.info("Best solution cost: " + str(self.best_cost))

    def anneal(self, n_iter=100000, reset_p=1/10000, time_limit=3600,
               log_iters=10000):
        '''
        This simulated annealing implementation is based on the following
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
        time_elapsed = 0
        iters = 0
        original_time_elapsed = self.total_time_elapsed
        start = time.time()
        for _ in range(n_iter):
            # check if time limit exceeded
            if time_elapsed >= time_limit:
                self.logger.info(
                    'Time limit reached/exceeded, terminating algorithm')
                break
            self.iters_since_reset = self.iters_since_reset + 1
            temp = self.get_temperature(self.iters_since_reset)
            # eps = 0.3 * e**(-i/n_iter)
            if np.random.uniform() < reset_p:
                self.logger.info("Resetting candidate solution.")
                self.new_candidate = self.reset_candidate()
                self.new_cost = self.cost(self.new_candidate)
                self.logger.info("with cost: " + str(self.new_cost))
                self.iters_since_reset = 0
                reset = True
            else:
                self.new_candidate = self.next_candidate(self.candidate)
                self.new_cost = self.cost(self.new_candidate)
            cost_del = self.cost_delta(self.new_cost, self.current_cost)
            if temp <= 0:
                # if temperature value is not greater than 0, candidate should
                # not be updated to a less optimal new candidate
                eps = 0
            else:
                # treat runtime warnings like errors, to catch overflow
                # warnings
                warnings.filterwarnings(
                    "error", category=RuntimeWarning)
                try:
                    # see if overflow occurs
                    eps = np.exp(-1 * cost_del / temp)
                except RuntimeWarning:
                    # overflow occurred

                    # if cost delta is positive, then eps will be very close to
                    # 0, so eps is set to 0
                    if cost_del > 0:
                        eps = 0
                    # if cost delta is negative, then eps will be very large
                    # (larger than any value sampled from uniform distribution
                    # on [0, 1)), so eps is set to 1
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
            
            self.total_iters += 1
            iters += 1
            time_elapsed = time.time() - start
            self.total_time_elapsed = original_time_elapsed + time_elapsed
            if iters == 1 or iters % int(log_iters) == 0:
                self._log_results(iters, time_elapsed)
        
        # log results, return best solution and best solution cost
        self._log_results(iters, time_elapsed)
        return self.best_solution, self.best_cost

    def update_candidate(self, candidate, cost):
        self.candidate = self._make_copy(candidate)
        self.current_cost = self._make_copy(cost)

    def update_best(self, candidate, cost):
        self.best_solution = self._make_copy(candidate)
        self.best_cost = self._make_copy(cost)

    def _make_copy(self, obj):
        return deepcopy(obj)


def s_curve(x, amplitude=4000, center=0, width=3000):
    # treat runtime warnings like errors, to catch overflow warnings
    warnings.filterwarnings("error", category=RuntimeWarning)
    try:
        res = 1 / (1 + np.exp((x - center) / width))
    except RuntimeWarning:
        # overflow occurred
        
        # np.exp term in the denominator is very large and the result is very
        # close to 0
        res = 0
    # reset warnings
    warnings.resetwarnings()
    return amplitude * res
