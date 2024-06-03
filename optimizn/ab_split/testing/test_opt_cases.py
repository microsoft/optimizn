import numpy as np
import multiprocessing
import time
import sys
from optimizn.ab_split.opt_split2 import OptProblm
from optimizn.ab_split.opt_split import optimize2
from optimizn.ab_split.opt_split2 import optimize1, optimize3
from optimizn.ab_split.ABSplitDuringDP import ABTestSituation


class TstCases():
    def __init__(self, fn, assrt_opt=False):
        """
        fn has to take as input a two dimensional array. Each
        entry in that array is the input arrays.
        """
        self.fn = fn
        self.assrt_opt = assrt_opt
        op = OptProblm()
        self.inputoutput = {
            "problem1: A case with conflict.": {
                "input": [
                    [2, 5, 9, 3, 1],
                    [2, 3, 4, 4, 3]
                ],
                "delta": 2
            },
            "problem2: A case where there are conflicts": {
                "input": [
                    [2, 4, 7, 9],
                    [1, 2, 3, 2],
                    [4, 7, 5, 2]
                ],
                "delta": 4
            },
            "problem3: A test case with zeros": {
                "input": [
                    [7, 0, 7, 0],
                    [0, 5, 0, 4]
                ],
                "delta": 1
            },
            "problem4: A real world test case": {
                "input": op.arrays,
                "delta": 1881
            },
            "problem5: All arrays agree": {
                "input": [
                    [3, 34, 4, 12, 5, 2],
                    [0, 25, 4, 12, 5, 2],
                    [22, 10, 4, 12, 5, 2],
                ],
                "delta": 25
            }
        }

    def tst_all(self):
        for k in self.inputoutput.keys():
            print("## Trying problem " + k + "\n#######\n")
            arr = self.inputoutput[k]["input"]
            target_delta = self.inputoutput[k]["delta"]
            split = self.fn(arr)
            total_delta = calc_sol_delta(arr, split)
            print("Model delta: " + str(total_delta) + ","
                  + " Best known delta: "
                  + str(target_delta))
            if self.assrt_opt:
                assert total_delta <= target_delta
            print("## Passed: " + k + "\n")


def calc_sol_delta(arr, split):
    total_delta = 0
    for i in range(len(arr)):
        ar = arr[i]
        sum1 = sum(ar)
        cnt1 = sum([ar[ix] for ix in split])
        total_delta += abs(sum1 - 2*cnt1)
    return total_delta


def tst1():
    tc = TstCases(ABTestSituation, False)
    # p = multiprocessing.Process(target=tc.tst_all, name="Foo")
    # p.start()
    tc.tst_all()


if __name__ == "__main__":
    tst1()
