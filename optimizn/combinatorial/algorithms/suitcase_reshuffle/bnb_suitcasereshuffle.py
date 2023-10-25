# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.branch_and_bound import BnBProblem
from copy import deepcopy
from functools import reduce
from optimizn.combinatorial.algorithms.suitcase_reshuffle.suitcases import\
    SuitCases


class SuitcaseReshuffleProblem(BnBProblem):
    '''
    Solution Format:
    2-tuple
    1. SuitCases object containing suitcases, weights, and empty space
    2. Index of last item (in list of sorted item weights) put in suitcase

    Branching strategy:
    Consider items in decreasing order of weight. Put item in each suitcase
    that can fit it
    '''

    def __init__(self, suitcases):
        self.config = suitcases.config
        self.capacities = suitcases.capacities
        self.suitcases = suitcases
        self.sorted_weights = self._get_weights(self.config, True)
        self.weight_counts = self._get_weight_counts(self.sorted_weights)
        super().__init__(suitcases)

    def _get_weights(self, suitcases, sort=False):
        weights = list(reduce(
            lambda l1, l2: l1 + l2[:-1], suitcases[1:], suitcases[0][:-1]))
        if sort:
            weights = sorted(weights, reverse=True)
        return weights

    def _get_weight_counts(self, weights):
        weight_counts = dict()
        for weight in weights:
            if weight not in weight_counts.keys():
                weight_counts[weight] = 1
            else:
                weight_counts[weight] += 1
        return weight_counts

    def get_candidate(self):
        return (deepcopy(self.suitcases), -1)

    def cost(self, sol):
        suitcases = sol[0]
        max_empty_space = float('-inf')
        for suitcase in suitcases.config:
            max_empty_space = max(max_empty_space, suitcase[-1])
        return -1 * max_empty_space

    def lbound(self, sol):
        suitcases = sol[0]
        empty_space = 0
        for suitcase in suitcases.config:
            empty_space += suitcase[-1]
        return -1 * empty_space
    
    def is_feasible(self, sol):
        suitcases = sol[0].config

        # check if suitcase number is valid index
        if sol[1] < -1 or sol[1] > len(self.sorted_weights) - 1:
            return False

        # for each suitcase, weights and extra space must be non-negative
        for i in range(len(suitcases)):
            suitcase = suitcases[i]
            suitcase_sum = 0
            for item in suitcase:
                suitcase_sum += item
                if item < 0:
                    return False

        # weights should not appear more often than in the original suitcase
        # configuration
        weight_counts = self._get_weight_counts(self._get_weights(suitcases))
        for weight, count in weight_counts.items():
            if weight not in self.weight_counts.keys():
                return False
            elif count > self.weight_counts[weight]:
                return False
            
        # check if solution can be completed (remaining items can be packed
        # into the suitcases)
        completed_sol = self.complete_solution(sol)
        if completed_sol is None:
            return False
        
        return True
        

    def is_complete(self, sol):
        suitcases = sol[0].config

        # for each suitcase, weights and extra space must equal original
        # capacity
        for i in range(len(suitcases)):
            if sum(suitcases[i]) != self.capacities[i]:
                return False
        
        # weights should appear exactly the same number of times as in
        # the original suitcase configs
        weight_counts = self._get_weight_counts(self._get_weights(suitcases))
        if weight_counts != self.weight_counts:
            return False
    
        return True

    def complete_solution(self, sol):
        suitcases = deepcopy(sol[0].config)

        # get remaining items to pack
        num_packed = len(list(reduce(
            lambda l1, l2: l1 + l2[:-1], suitcases[1:], suitcases[0][:-1])))
        items_to_pack = self.sorted_weights[num_packed:]

        # put each item in the suitcase with the least extra space that
        # can hold it
        for weight in items_to_pack:
            # find suitcase to pack item in
            min_space = None
            min_suitcase = None
            for i in range(len(suitcases)):
                extra_space = suitcases[i][-1]
                if extra_space >= weight:
                    if min_space is None and min_suitcase is None:
                        min_space = extra_space
                        min_suitcase = i
                    elif min_space > extra_space:
                        min_space = extra_space
                        min_suitcase = i
            
            # if item cannot be packed, solution cannot be completed
            if min_space is None and min_suitcase is None:
                return None
            
            # pack item in suitcase
            suitcases[min_suitcase] = suitcases[min_suitcase][:-1] + [weight]\
                + [suitcases[min_suitcase][-1] - weight]
        
        return (SuitCases(suitcases), sol[1])


    def branch(self, sol):
        last_item_idx = sol[1]
        if last_item_idx == -1:
            # if last item index is 0 (initial solution), start from empty
            # suitcases
            suitcases = []
            for capacity in self.capacities:
                suitcases.append([capacity])
        elif last_item_idx == len(self.sorted_weights) - 1:
            # if last item has been packed, no further branching can be done
            return []
        else:
            suitcases = sol[0].config

        # get next item weight
        next_item_idx = last_item_idx + 1
        next_item_weight = self.sorted_weights[next_item_idx]

        # pack next item in each suitcase that can fit it
        new_sols = []
        for i in range(len(suitcases)):
            extra_space = suitcases[i][-1]
            if extra_space >= next_item_weight:
                new_suitcases = deepcopy(suitcases)
                new_suitcases[i] = new_suitcases[i][:-1] + [next_item_weight]\
                    + [new_suitcases[i][-1] - next_item_weight]
                new_sols.append((SuitCases(new_suitcases), next_item_idx))

        return new_sols
