# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## Binary assignment for equal spoils.
import numpy as np
import cvxpy as cp
import optimizn.problems.sample_ci as sci


def formultn1(ci=sci.candy_preferences):
    #ci = np.array([10,7,6,3])
    x = cp.Variable(len(ci),boolean=True)
    objective = cp.Minimize(cp.sum_squares(ci@(2*x-1)))
    problm = cp.Problem(objective)
    #res = problm.solve(solver='GLPK')
    #_ = problm.solve(solver=cp.GLPK_MI)
    _ = problm.solve()
    return x.value


def formulatn2(ci=sci.candy_preferences):  
    '''
    This formulation is from the following source.

    Source:

    (1)
    Title: Binary assignment - distributing candies.
    Author: Rohit Pandey (author of question), RobPratt (author of answer)
    URL: https://math.stackexchange.com/questions/3515223/binary-assignment-distributing-candies/3515391#3515391
    Date published: January 19, 2020
    Date accessed: January 19, 2020
    '''  
    #ci = np.array([10,7,6,3])
    z = cp.Variable()
    x = cp.Variable(len(ci),boolean=True)
    constraints = [ci@x<=z, ci@(1-x)<=z]
    #constraints = [sum(ci*x)<=z,sum(ci*(1-x))<=z]
    objective = cp.Minimize(z+0*sum(x))
    problm = cp.Problem(objective,constraints)
    #_ = problm.solve(solver=cp.GLPK_MI)
    _ = problm.solve()
    return x.value


def formultn3(ri=np.array([1,3,5,2,4,6]),\
            ui=np.array([1,1,1,1,1,1])):
    z1 = cp.Variable()
    z2 = cp.Variable()
    x = cp.Variable(len(ri),boolean=True)
    constraints = [ri@x<=z1, ri@(1-x)<=z1,\
                    ui@x<=z1, ui@(1-x)<=z2]
    objective = cp.Minimize(z1+z2)
    problm = cp.Problem(objective,constraints)
    _ = problm.solve()
    return x.value

