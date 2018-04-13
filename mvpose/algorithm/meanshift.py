import numpy as np
import mvpose.math as mvmath
from numba import jit, float64


@jit([float64[:](float64[:], float64[: ,:], float64)], nopython=True, nogil=True)
def m(y, Nx, sigma):
    num = len(Nx)
    G = mvmath.gauss3d(y[0]-Nx[: ,0], y[1]-Nx[: ,1], y[2]-Nx[: ,2], sigma)

    result = np.zeros((3,))
    div = 0

    for i in range(num):
        result += Nx[i ,0:3] * G[i] * Nx[i,3]
        div += G[i] * Nx[i,3]

    return result /div