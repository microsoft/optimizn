# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from optimizn.combinatorial.branch_and_bound import BnBProblem
from copy import deepcopy


class KnapsackParams:
    def __init__(self, values, weights, capacity):
        self.values = values
        self.weights = weights
        self.capacity = capacity

    def __eq__(self, other):
        return (
            other is not None
            and (len(self.values) == len(other.values)
                 and (self.values == other.values).all())
            and (len(self.weights) == len(other.weights)
                 and (self.weights == other.weights).all())
            and self.capacity == other.capacity
        )


class ZeroOneKnapsackProblem(BnBProblem):
    '''
    Branch and bound implementation for the 0/1 knapsack problem, where each
    item is either taken or omitted in its entirety.

    This implementation is based on the demonstration shown in the following
    source.

    Source:

    (1)
    Title: 7.2 0/1 Knapsack using Branch and Bound
    Author: Abdul Bari
    URL: https://www.youtube.com/watch?v=yV1d-b_NeK8
    Date published: February 26, 2018
    Date accessed: December 16, 2022
    '''
    def __init__(self, params):
        self.values = params.values
        self.weights = params.weights
        self.capacity = params.capacity

        # value/weight ratios, in decreasing order
        vw_ratios = self.values / self.weights
        vw_ratios_ixs = []
        for i in range(len(vw_ratios)):
            vw_ratios_ixs.append((vw_ratios[i], i))
        self.sorted_vw_ratios = sorted(vw_ratios_ixs)
        self.sorted_vw_ratios.reverse()
        super().__init__(params)

    def get_initial_solution(self):
        return self.complete_solution([])
    
    def get_root(self):
        return []

    def lbound(self, sol):
        value = 0
        weight = 0

        # consider items already taken
        for i in range(0, len(sol)):
            if sol[i] == 1:
                value += self.values[i]
                weight += self.weights[i]

        # greedily take other items
        for vw_ratio, ix in self.sorted_vw_ratios:
            if ix < len(sol):
                continue
            rem_cap = self.capacity - weight
            if rem_cap <= 0:
                break
            item_weight = min(rem_cap, self.weights[ix])
            value += item_weight * vw_ratio
            weight += item_weight

        return -1 * value

    def cost(self, sol):
        return -1 * np.sum(np.array(sol) * np.array(self.values[:len(sol)]))

    def branch(self, sol):
        if len(sol) >= len(self.weights):
            return []

        new_sols = []
        for val in [0, 1]:
            new_sols.append(deepcopy(sol) + [val])
        return new_sols

    def is_feasible(self, sol):
        # check that array length is the same as the number of weights/values
        check_length1 = len(sol) == len(self.weights)
        check_length2 = len(sol) == len(self.values)
        check_length = check_length1 and check_length2

        # check that the only values in the array are 0 and 1
        check_values = len(set(sol).difference({0, 1})) == 0

        # check that the weight of the values in the array is not greater
        # than the capacity
        check_weight = np.sum(np.array(sol) * np.array(
            self.weights[:len(sol)])) <= self.capacity

        return check_length and check_values and check_weight

    def complete_solution(self, sol):
        # greedily add other items to array
        knapsack = [0] * len(self.weights)
        knapsack[0:len(sol)] = sol
        value = 0
        weight = 0
        for i in range(len(knapsack)):
            if knapsack[i] == 1:
                value += self.values[i]
                weight += self.weights[i]

        # greedily take other items
        for _, ix in self.sorted_vw_ratios:
            if ix < len(sol):
                continue
            rem_cap = self.capacity - weight
            if rem_cap < self.weights[ix]:
                continue
            value += self.values[ix]
            weight += self.weights[ix]
            knapsack[ix] = 1
        
        return knapsack
