# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from optimizn.combinatorial.branch_and_bound import BnBProblem
from functools import reduce
import copy
import math


class BinPackingParams:
    def __init__(self, weights, capacity):
        self.weights = weights
        self.capacity = capacity

    def __eq__(self, other):
        return (
            other is not None
            and self.weights == other.weights
            and self.capacity == other.capacity
        )
    
    def __str__(self):
        return f'BinPackingParams - weights: {self.weights}, capacity: '\
            + f'{self.capacity}'


class BinPackingProblem(BnBProblem):
    '''
    Solution format: 2-tuple
    1. Allocation of items to bins (dict, keys are integers representing bins
    (starting from 1, so 1 represents the first bin, 2 is the second bin, etc.)
    and values are sets of integers that represent the items (1 represents
    first item in weights list, 2 represents second item in weights list, and
    so on))
    2. Index of last allocated item in sorted-by-decreasing-weight list of
    items (int)

    Branching strategy:
    Each level of the solution space tree corresponds to an item. Items are
    considered in order of decreasing weight. Each solution in a level 
    corresponds to the item being placed in a bin that it can fit in. The
    remaining items can be put in bins in decreasing order of weight, into
    the first bin that can fit it. New bins created as needed
    '''
    def __init__(self, params):
        self.item_weights = {}  # mapping of items to weights
        self.sorted_item_weights = []  # sorted (weight, item) tuples (desc)
        for i in range(1, len(params.weights) + 1):
            self.item_weights[i] = params.weights[i - 1]
            self.sorted_item_weights.append((params.weights[i - 1], i))
        self.sorted_item_weights.sort(reverse=True)
        self.capacity = params.capacity
        super().__init__(params)
    
    def get_candidate(self):
        return (self._pack_rem_items(dict(), -1), -1)

    def _pack_rem_items(self, bin_packing, last_item_idx):
        '''
        This function performs the first-fit decreasing algorithm presented in
        the following source.

        Source:

        (1)
        Title: Knapsack Problems, Algorithms and Computer Implementations
        Author: Silvano Martello, Paolo Toth
        URL: http://www.or.deis.unibo.it/knapsack.html, specifically the PDF for
        Chapter 8, http://www.or.deis.unibo.it/kp/Chapter8.pdf
        Date published: 1990
        Date accessed: January 11, 2023
        '''
        next_item_idx = last_item_idx + 1
        for i in range(next_item_idx, len(self.sorted_item_weights)):
            next_item_weight, next_item = self.sorted_item_weights[i]
            bins = set(bin_packing.keys())
            item_packed = False
            for bin in bins:
                # check if bin has space
                bin_weight = sum(
                    list(map(
                        lambda x: self.item_weights[x],
                        bin_packing[bin])
                    ))
                if next_item_weight > self.capacity - bin_weight:
                    continue

                # put item in bin
                bin_packing[bin].add(next_item)
                item_packed = True
                break

            # create new bin if needed
            if not item_packed:
                new_bin = 1
                if len(bins) != 0:
                    new_bin = max(bins) + 1
                bin_packing[new_bin] = set()
                bin_packing[new_bin].add(next_item)
        return bin_packing

    def _filter_items(self, bin_packing, last_item_idx):
        # remove items that have not been considered yet
        considered_items = set(map(
            lambda x: x[1], self.sorted_item_weights[0:last_item_idx + 1]))
        new_bin_packing = {}
        for bin in bin_packing.keys():
            new_bin = set(filter(
                lambda x: x in considered_items, bin_packing[bin]))
            if len(new_bin) != 0:
                new_bin_packing[bin] = new_bin
        return new_bin_packing

    def lbound(self, sol):
        '''
        This lower bound function is based on the lower bound function L_1
        presented in the following source.

        Source:

        (1) 
        Title: Knapsack Problems, Algorithms and Computer Implementations
        Author: Silvano Martello, Paolo Toth
        URL: http://www.or.deis.unibo.it/knapsack.html, specifically the PDF for
        Chapter 8, http://www.or.deis.unibo.it/kp/Chapter8.pdf
        Date published: 1990
        Date accessed: January 11, 2023
        '''
        bin_packing = sol[0]
        last_item_idx = sol[1]

        # remove items that have not been considered yet
        bin_packing = self._filter_items(bin_packing, last_item_idx)
        curr_bin_ct = len(bin_packing.keys())

        # get free capacity in bin packing
        curr_weight = sum(list(map(
            lambda x: self.sorted_item_weights[x][0],
            list(range(last_item_idx + 1))
        )))
        free_capacity = self.capacity * curr_bin_ct - curr_weight

        # get weights of remaining items
        rem_weight = sum(list(map(
            lambda x: self.sorted_item_weights[x][0],
            list(range(last_item_idx + 1, len(self.sorted_item_weights)))
        )))        

        return curr_bin_ct + math.ceil(
            (rem_weight - free_capacity) / self.capacity)

    def cost(self, sol):
        bin_packing = sol[0]
        return len(bin_packing.keys())

    def branch(self, sol):
        # determine next item and its weight
        bin_packing = sol[0]
        last_item_idx = sol[1]
        next_item_idx = last_item_idx + 1
        if next_item_idx >= len(self.sorted_item_weights):
            return []
        next_item_weight, next_item = self.sorted_item_weights[
            next_item_idx]

        # remove items that have not been considered yet
        bin_packing = self._filter_items(bin_packing, last_item_idx)

        # pack items in bins
        new_sols = []
        extra_bin = 1
        if len(bin_packing.keys()) != 0:
            extra_bin = max(bin_packing.keys()) + 1
        bins = set(bin_packing.keys()).union({extra_bin})
        for bin in bins:
            # create new bin if considering new bin index
            new_bin_packing = copy.deepcopy(bin_packing)
            if bin not in new_bin_packing.keys():
                new_bin_packing[bin] = set()

            # check if bin has space
            bin_weight = sum(
                list(map(
                    lambda x: self.item_weights[x],
                    new_bin_packing[bin])
                ))
            if next_item_weight > self.capacity - bin_weight:
                continue

            # pack item in bin
            new_bin_packing[bin].add(next_item)
            new_sols.append((new_bin_packing, next_item_idx))
        return new_sols

    def is_feasible(self, sol):
        bin_packing = sol[0]

        # check that packed items are valid
        items = set(reduce(
            (lambda s1, s2: s1.union(s2)),
            list(map(lambda b: bin_packing[b], bin_packing.keys()))
        ))
        if len(items.difference(set(range(1, len(self.item_weights)+1)))) != 0:
            return False

        # check that for each bin, the weight is not exceeded
        for bin in bin_packing.keys():
            bin_weight = sum(
                list(map(lambda x: self.item_weights[x], bin_packing[bin])))
            if bin_weight > self.capacity:
                return False

        return True

    def is_complete(self, sol):
        bin_packing = sol[0]

        # check that all items are packed
        items = set(reduce(
            (lambda s1, s2: s1.union(s2)),
            list(map(lambda b: bin_packing[b], bin_packing.keys()))
        ))
        if items != set(range(1, len(self.item_weights)+1)):
            return False

        # check that for each bin, the weight is not exceeded
        for bin in bin_packing.keys():
            bin_weight = sum(
                list(map(lambda x: self.item_weights[x], bin_packing[bin])))
            if bin_weight > self.capacity:
                return False

        return True

    def complete_solution(self, sol):
        return (self._pack_rem_items(copy.deepcopy(sol[0]), sol[1]), sol[1])
