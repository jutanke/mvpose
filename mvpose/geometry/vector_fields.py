import numpy as np
from mvpose.plot.limbs import draw_vector_field


def clamp_to_1(U, V):
    """
        clamps the vector field to have only vectors of max length 1
        The neural network sometimes outputs vectors which are larger
        than 1.. clamp them
    :param U: vectors for x
    :param V: vectors for y
    :return:
    """
    assert U.shape == V.shape

    L = draw_vector_field(U, V)
    Mask = (L > 1) * 1

    XY = Mask.nonzero()

    Div = np.ones_like(U)
    Div[XY] = L[XY]

    return U/Div, V/Div