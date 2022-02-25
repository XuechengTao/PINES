# Driver for Path-Integral calculations iNvolving Excited States (PINES)

import defs, utils, force_engine
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import filecmp

def parse_args():
    parser = argparse.ArgumentParser(description='Read in customized system parameters', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dt', dest='dt', type=float, default=0.1, help="Propagation timestep in fs")
    parser.add_argument('--nsteps', dest='nsteps', type=int, default=25, help="Number of steps in integers")
    parser.add_argument('--temp', dest='temperature', type=float, default=0., help="System temperature in K")
    parser.add_argument('--tau', dest='coupling_time', type=float, default=100., help="Coupling time in fs")
    return parser.parse_args()

def unit_to_au(args):
    args.dt *= utils.fs_to_autime
    args.coupling_time *= utils.fs_to_autime
    args.temperature *= utils.kboltz

def thermostat(state, temp):
    sigma = np.sqrt(temp / state.nuclei.mass)
    print (sigma)
    for iatom in range(len(state.nuclei.mass)):
        state.nuclei.velocity[iatom,:] = np.random.normal(0., sigma[iatom], 3)

    # utils.mprint(state.nuclei.velocity)
    return True

# Propagating the system for a period of dt
def integrator(state, system, dt, force_engine):
    state.time += dt
    force = force_engine(state.nuclei.position)
    state.nuclei.velocity += 0.5 * dt * force / state.nuclei.mass[:, None]
    state.nuclei.position += dt * state.nuclei.velocity
    force = force_engine(state.nuclei.position)
    state.nuclei.velocity += 0.5 * dt * force / state.nuclei.mass[:, None]
    if (system.temperature > utils.tol and (state.time % system.coupling_time) < dt):
        thermostat(state, system.temperature)

# The main driver for the dynamics simulation
def pines(args):
    unit_to_au(args)

    print("\n ======= RUNNING PINES ====== \n")
    system = defs.System(args.temperature, args.coupling_time)
    # MASS, POSITION, VELOCITY in amu, A, A / fs
    state = defs.State([np.array([1., 2.]) * utils.amu_to_au, \
                        (np.array([[1., 0., 0.], [0., 1., 0.]])) * utils.angstrom_to_bohr, \
                        (np.array([[0., 0., 0.], [0., 0., 0.]])) * utils.angstrom_to_bohr * utils.autime_to_fs])

    print ("Simulated trajectory:")
    print ("%16s%16s" % ("#time (fs)", "position x_1 (Bohr)"))
    utils.inter_print(state)
    with open('trajectory.xyz', "w") as traj_file:
        utils.xyzprint(state, traj_file)
        for istep in range(args.nsteps):
            integrator(state, system, args.dt, force_engine.harmonic)
            utils.inter_print(state)
            utils.xyzprint(state, traj_file)
    print("\n Assertion", filecmp.cmp('trajectory.xyz', 'examples/trajectory.backup'), "\n")

if __name__ == "__main__":
    args = parse_args()
    args.nsteps = 150
    args.temperature = 0.

    pines(args)
