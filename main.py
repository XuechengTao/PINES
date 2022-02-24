# Driver for Path-Integral calculations iNvolving Excited States (PINES)

import defs, utils
import numpy as np


def ho_force_engine(position):
    # V = 1/2 x^2, return -dV/dx = -x
    return -position


# Propagating the system for a period of dt
def integrator(state, dt, force_engine):
    state.time += dt
    state.nuclei.velocity += 0.5 * dt * force_engine(state.nuclei.position) / state.nuclei.mass[:, None]
    state.nuclei.position += dt * state.nuclei.velocity
    state.nuclei.velocity += 0.5 * dt * force_engine(state.nuclei.position) / state.nuclei.mass[:, None]


# The main driver for the dynamics simulation
def pines(dt, nsteps):
    print("\n ======= RUNNING PINES ====== \n")
    #                       MASS        POSITION                        VELOCITY
    state = defs.State([np.array([1.]), np.array([[1.], [0.], [0.]]).T, np.array([[0.], [0.], [0.]]).T])

    print("Simulated trajectory:")
    print("%16s%32s%16s" % ("#time", "position", "(x,y,z)"))
    for istep in range(nsteps):
        print(defs.fprint_format % state.time, end='')
        integrator(state, dt, ho_force_engine)
        utils.mprint(state.nuclei.position)


if __name__ == "__main__":
    pines(0.1, 100)
