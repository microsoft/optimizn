# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sympy import *


x, y, z, l = symbols('x y z l')
#k, m, n = symbols('k m n', integer=True)
#f, g, h = map(Function, 'fgh')

((x+y)**2 * (x+1)).expand()


## Unfortunately, produces just one solution, not all.
solve([Eq(3*x**2+2*y*z-2*x*l,0),
       Eq(2*x*z-2*y*l,0),
       Eq(2*x**2-2*z-2*z*l,0),
       Eq(x**2+y**2+z**2,1)], [x,y,z,l])

F = [f1, f2, f3, f4] = [3*x**2+2*y*z-2*x*l, 2*x*z-2*y*l, 2*x**2-2*z-2*z*l, x**2+y**2+z**2-1]

gb = groebner([f1, f2, f3, f4], l, x, y, z, order='lex')

gb[len(gb)-1]


