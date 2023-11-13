# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from numpy.random import uniform
# from numpy import e
import numpy as np
from copy import deepcopy
from optimizn.combinatorial.opt_problem import OptProblem
import time


class SimAnnealProblem(OptProblem):
    def __init__(self):
        ''' Initialize the problem '''
        # Instead of always stopping at a random solution, pick
        # the best one sometimes and the best daily one other times.
        super().__init__()
        self.candidate = make_copy(self.best_solution)
        self.current_cost = self.best_cost

    def next_candidate(self):
        ''' Switch to the next candidate.'''
        raise Exception("Not implemented")

    def reset_candidate(self):
        '''
        Returns a new solution for when the candidate solution is reset.
        Defaults to get_candidate but can be overridden if needed
        '''
        return self.get_candidate()

    def anneal(self, n_iter=100000, reset_p=1/10000, time_limit=3600):
        '''
        This simulated annealing implementation is based on the code and
        explanation of simulated annealing presented in the following sources.

        Sources:
        
        (1)
        Title: toddwschneider/shiny-salesman
        Author: Todd W. Schneider
        URL: https://github.com/toddwschneider/shiny-salesman/blob/master/helpers.R
        Date published: October 1, 2014
        Date accessed: January 8, 2023

        The code presented in this source is licensed under the MIT License.
        The original license text is shown in the NOTICE.md file.

        (2)
        Title: The Traveling Salesman with Simulated Annealing, R, and Shiny
        Author: Todd W. Schneider
        URL: https://toddwschneider.com/posts/traveling-salesman-with-simulated-annealing-r-and-shiny/
        Date published: October 1, 2014
        Date accessed: January 8, 2023
        '''
        reset = False
        j = -1
        start = time.time()
        for i in range(n_iter):
            # check if time limit exceeded
            if time.time() - start > time_limit:
                print('Time limit exceeded, terminating algorithm')
                print('Best solution: ', self.best_cost)
                break
            j = j + 1
            temprature = current_temperature(j)
            if i % 10000 == 0:
                print("Iteration: " + str(i) + " Current best solution: "
                      + str(self.best_cost))
            # eps = 0.3 * e**(-i/n_iter)
            if np.random.uniform() < reset_p:
                print("Resetting candidate solution.")
                self.new_candidate = self.reset_candidate()
                self.new_cost = self.cost(self.new_candidate)
                print("with cost: " + str(self.new_cost))
                j = 0
                reset = True
            else:
                self.new_candidate = self.next_candidate(self.candidate)
                self.new_cost = self.cost(self.new_candidate)
            cost_del = self.cost_delta(self.new_cost, self.current_cost)
            eps = np.exp(cost_del / temprature)

            if self.new_cost < self.current_cost or eps < uniform() or reset:
                self.update_candidate(self.new_candidate,
                                      self.new_cost)
                if reset:
                    reset = False
            if self.new_cost < self.best_cost:
                self.update_best(self.new_candidate, self.new_cost)
                print("Best cost updated to:" + str(self.new_cost))

    def update_candidate(self, candidate, cost):
        self.candidate = make_copy(candidate)
        self.current_cost = cost

    def update_best(self, candidate, cost):
        self.best_solution = make_copy(candidate)
        self.best_cost = cost


def make_copy(candidate):
    return deepcopy(candidate)


def s_curve(x, center, width):
    return 1 / (1 + np.exp((x - center) / width))


def current_temperature(iter, s_curve_amplitude=4000,
                        s_curve_center=0, s_curve_width=3000):
    return s_curve_amplitude * s_curve(iter, s_curve_center, s_curve_width)
