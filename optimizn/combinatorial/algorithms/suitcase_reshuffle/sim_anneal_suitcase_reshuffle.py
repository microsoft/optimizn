# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from optimizn.combinatorial.simulated_annealing import SimAnnealProblem
from copy import deepcopy


class SuitCaseReshuffle(SimAnnealProblem):
	def __init__(self, params):
		self.params = params
		self.name = "SuitcaseReshuffling"
		super().__init__()

	def get_candidate(self):
		self.candidate = self.params.config
		return self.candidate

	def cost(self, candidate):
		maxx = 0
		for ar in candidate:
			maxx = max(maxx, ar[len(ar)-1])
		return -maxx

	def next_candidate(self, candidate):
		# Find two items to swap from two different
		# suitcases.
		keep_going = True
		while keep_going:
			candidate1 = deepcopy(candidate)
			l = np.arange(len(candidate))
			cases = np.random.choice(l, size=2, replace=False)
			ix1 = np.random.choice(len(candidate[cases[0]]) - 1)
			ix2 = np.random.choice(len(candidate[cases[1]]) - 1)
			size1 = candidate[cases[0]][ix1]
			size2 = candidate[cases[1]][ix2]
			candidate1[cases[0]][ix1] = size2
			candidate1[cases[1]][ix2] = size1
			arr1 = candidate1[cases[0]]
			arr2 = candidate1[cases[1]]
			caps = self.params.capacities
			if caps[cases[0]] < sum(arr1[:len(arr1)-1])\
				or caps[cases[1]] < sum(arr2[:len(arr2)-1]):
				continue
			else:
				keep_going = False
				arr1[len(arr1)-1] = caps[cases[0]]\
									- sum(arr1[:len(arr1)-1])
				arr2[len(arr2)-1] = caps[cases[1]]\
									- sum(arr2[:len(arr2)-1])
		return candidate1
