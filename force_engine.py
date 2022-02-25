import numpy as np
import utils

# Harmonic force engine
def harmonic(position):
# V = 1/2 m omega^2 x^2, return -dV/dx = -m omega^2 x
    omega = 3333.3333 * utils.inv_cm_to_freq_au
    return -position * utils.amu_to_au * omega**2
