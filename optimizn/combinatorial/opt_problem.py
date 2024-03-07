# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import os
from datetime import datetime
from optimizn.utils import get_logger
from copy import deepcopy


class OptProblem():
    def __init__(self, params, logger=None):
        ''' Initialize the problem '''
        self.name = self.__class__.__name__
        self.params = params
        if logger is None:
            self.logger = get_logger(f'{self.name}_logger')
        else:
            self.logger = logger
        self.init_time = datetime.now()
        self.init_secs = int(self.init_time.timestamp())
        self.init_solution = self.get_initial_solution()
        if self.init_solution is not None:
            self.init_cost = self.cost(self.init_solution)
        else:
            self.init_cost = float('inf')
        self.best_solution = deepcopy(self.init_solution)
        self.best_cost = deepcopy(self.init_cost)
        self.logger.info(f'Initial solution: {self.init_solution}')
        self.logger.info(f'Initial solution cost: {self.init_cost}')

    def get_initial_solution(self):
        ''' Gets the initial solution.'''
        raise NotImplementedError(
            "Implement a function to get the initial solution")

    def cost(self, sol):
        ''' Gets the cost for a given solution.'''
        raise NotImplementedError(
            "Implement a function to compute the cost of a given solution")

    def cost_delta(self, cost1, cost2):
        return cost1 - cost2

    def persist(self):
        create_folders(self.name)
        existing_obj = load_latest_pckl(
            "Data//" + self.name + "//DailyObj", self.logger)
        if existing_obj is None:
            self.obj_changed = True
        else:
            self.obj_changed = (existing_obj != self.params)
        if self.obj_changed:
            # Write the latest input object that has changed.
            f_name = "Data//" + self.name + "//DailyObj//" +\
                        str(self.init_secs) + ".obj"
            file1 = open(f_name, 'wb')
            pickle.dump(self.params, file1)
            file1.close()
            self.logger.info("Wrote to DailyObj")
        # Write the optimization object.
        f_name = "Data//" + self.name + "//DailyOpt//" + str(self.init_secs)\
            + ".obj"
        file1 = open(f_name, 'wb')
        pickle.dump(self, file1)
        file1.close()
        self.logger.info("Wrote to DailyOpt")

        # Now check if the current best is better than the global best
        existing_best = load_latest_pckl(
            "Data//" + self.name + "//GlobalOpt", self.logger)
        if existing_best is None or self.cost_delta(
                self.best_cost, existing_best.best_cost) < 0\
                or self.obj_changed:
            f_name = "Data//" + self.name + "//GlobalOpt//" +\
                        str(self.init_secs) + ".obj"
            file1 = open(f_name, 'wb')
            pickle.dump(self, file1)
            file1.close()
            self.logger.info("Wrote to GlobalOpt")


def create_folders(name):
    if not os.path.exists("Data//"):
        os.mkdir("Data//")
    if not os.path.exists("Data//" + name + "//"):
        os.mkdir("Data//" + name + "//")
    if not os.path.exists("Data//" + name + "//DailyObj//"):
        os.mkdir("Data//" + name + "//DailyObj//")
    if not os.path.exists("Data//" + name + "//DailyOpt//"):
        os.mkdir("Data//" + name + "//DailyOpt//")
    if not os.path.exists("Data//" + name + "//GlobalOpt//"):
        os.mkdir("Data//" + name + "//GlobalOpt//")


def load_latest_pckl(path1="Data/DailyObj", logger=None):
    if logger is None:
        logger = get_logger('optimizn_logger')
    if not os.path.exists(path1):
        logger.warning(f'No file located at {path1}')
        return None
    msh_files = os.listdir(path1)
    msh_files = [i for i in msh_files if not i.startswith('.')]
    msh_files = sorted(msh_files)
    if len(msh_files) > 0:
        latest_file = msh_files[len(msh_files)-1]
        filepath = path1 + "//" + latest_file
        if os.path.getsize(filepath) == 0:
            logger.warning(f'File located at {filepath} is empty')
        else:
            filehandler = open(filepath, 'rb')
            existing_obj = pickle.load(filehandler)
            filehandler.close()
            return existing_obj
    return None
