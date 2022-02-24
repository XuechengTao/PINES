import defs
import numpy as np

def mprint(matrix):
    for row in matrix:
        for val in row:
            print(defs.fprint_format % val, end='')
        print()

def momentum_sampling(m: float, T: float):
    """
    sample momentum states from Boltzmann distribution
    f(v) = \frac{m}{2 \pi k_b T}^{3/2} e^{- \frac{m v^2}{2 k_B T}
    :param T: temperature in K
    :return: momentum p sampled from gaussian
    """
    kb = 1.38064852e-23
    return np.random.normal(np.sqrt(2*kb*T / m))