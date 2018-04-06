import numba as nb
import numpy as np

x0 = np.zeros((0,0), 'int32')
x0.setflags(write=False)

int32_2d_const = nb.typeof(x0)

d0 = np.zeros((0,0), 'float64')
d0.setflags(write=False)

float64_2d_const = nb.typeof(d0)