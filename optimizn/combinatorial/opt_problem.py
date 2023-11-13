# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import os
from datetime import datetime


class OptProblem():
    def __init__(self):
        ''' Initialize the problem '''
        self.init_time = datetime.now()
        self.init_secs = int(self.init_time.timestamp())
        self.best_solution = self.get_candidate()
        self.best_cost = self.cost(self.best_solution)
        print(f'Initial solution: {self.best_solution}')
        print(f'Initial solution cost: {self.best_cost}')
        self.name = self.__class__.__name__
        if not hasattr(self, 'params'):
            raise Exception(
                'All problem class instances must have a "params" attribute, '
                + 'which is an object that contains the input parameters '
                + 'to the problem class')

    def get_candidate(self):
        ''' Gets a feasible candidate.'''
        raise Exception("Not implemented")

    def cost(self, sol):
        ''' Gets the cost for candidate solution.'''
        raise Exception("Not implemented")

    def cost_delta(self, cost1, cost2):
        return cost1 - cost2

    def persist(self):
        create_folders(self.name)
        existing_obj = load_latest_pckl("Data//" + self.name + "//DailyObj")
        self.obj_changed = (existing_obj != self.params)
        if self.obj_changed or existing_obj is None:
            # Write the latest input object that has changed.
            f_name = "Data//" + self.name + "//DailyObj//" +\
                        str(self.init_secs) + ".obj"
            file1 = open(f_name, 'wb')
            pickle.dump(self.params, file1)
            print("Wrote to DailyObj")
        # Write the optimization object.
        f_name = "Data//" + self.name + "//DailyOpt//" + str(self.init_secs)\
            + ".obj"
        file1 = open(f_name, 'wb')
        pickle.dump(self, file1)
        print("Wrote to DailyOpt")

        # Now check if the current best is better
        # than the global best
        existing_best = load_latest_pckl("Data//" + self.name + "//GlobalOpt")
        if existing_best is None or self.best_cost > existing_best.best_cost\
                or self.obj_changed:
            f_name = "Data//" + self.name + "//GlobalOpt//" +\
                        str(self.init_secs) + ".obj"
            file1 = open(f_name, 'wb')
            pickle.dump(self, file1)
            print("Wrote to GlobalOpt")


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


def load_latest_pckl(path1="Data/DailyObj"):
    if not os.path.exists(path1):
        return None
    msh_files = os.listdir(path1)
    msh_files = [i for i in msh_files if not i.startswith('.')]
    msh_files = sorted(msh_files)
    if len(msh_files) > 0:
        latest_file = msh_files[len(msh_files)-1]
        filepath = path1 + "//" + latest_file
        if os.path.getsize(filepath) == 0:
            print('File located at', filepath, 'is empty')
        else:
            filehandler = open(filepath, 'rb')
            existing_obj = pickle.load(filehandler)
            return existing_obj
    return None
