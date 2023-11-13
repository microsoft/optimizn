# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np


class CityGraph():
    def __init__(self, num_cities=50):
        # Generate x-y coordinates of some cities.
        # Here, we just draw them from a normal dist.
        self.xs = np.random.normal(loc=0,scale=5,size=(num_cities,2))
        self.num_cities = len(self.xs)
        self.dists = np.zeros((len(self.xs), len(self.xs)))
        # Populate the matrix of euclidean distances.
        for i in range(len(self.xs)):
            for j in range(i+1, len(self.xs)):
                dist = (self.xs[i][0]-self.xs[j][0])**2
                dist += (self.xs[i][1]-self.xs[j][1])**2
                dist = np.sqrt(dist)
                self.dists[i,j] = dist
        for i in range(len(self.xs)):
            for j in range(i):
                self.dists[i,j] = self.dists[j,i]
