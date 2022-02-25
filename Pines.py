# Driver for Path-Integral calculations iNvolving Excited States (PINES)

import Defs, Utils, ForceEngine
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

def parse_args():
    parser = argparse.ArgumentParser(description='Read in customized system parameters', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dt', dest='dt', type=float, default=0.1, help="Propagation timestep in fs")
    parser.add_argument('--nsteps', dest='nsteps', type=int, default=25, help="Number of steps in integers")
    parser.add_argument('--temp', dest='temperature', type=float, default=0., help="System temperature in K")
    parser.add_argument('--tau', dest='coupling_time', type=float, default=100., help="Coupling time in fs")
    return parser.parse_args()

def unit_to_au(args):
    args.dt *= Utils.fs_to_autime
    args.coupling_time *= Utils.fs_to_autime
    args.temperature *= Utils.kboltz

def thermostat(state, temp):
    sigma = np.sqrt(temp / state.nuclei.mass)
    print (sigma)
    for iatom in range(len(state.nuclei.mass)):
        state.nuclei.velocity[iatom,:] = np.random.normal(0., sigma[iatom], 3)
    # Utils.mprint(state.nuclei.velocity)
    return True

# Propagating the system for a period of dt
def integrator(state, system, dt, force_engine):
    state.time += dt

    pot_energy, force = force_engine(state.nuclei.position)
    state.nuclei.velocity += 0.5 * dt * force / state.nuclei.mass[:, None]

    state.nuclei.position += dt * state.nuclei.velocity

    pot_energy, force = force_engine(state.nuclei.position)
    state.nuclei.velocity += 0.5 * dt * force / state.nuclei.mass[:, None]

    if (system.temperature > Utils.tol and (state.time % system.coupling_time) < dt):
        thermostat(state, system.temperature)

    ForceEngine.update_state_energy(state, pot_energy)

# The main driver for the dynamics simulation
def pines(args, system, state):
    unit_to_au(args)

    print("\n ======= RUNNING PINES ====== \n")
    print ("Simulated trajectory:")
    print ("%16s%16s%16s%16s" % ("#time (fs)", "kinetic", "potential", "total energy"))
    Utils.inter_print(state)
    with open('trajectory.xyz', "w") as traj_file:
        Utils.xyzprint(state, traj_file)
        for istep in range(args.nsteps):
            integrator(state, system, args.dt, ForceEngine.harmonic)
            Utils.inter_print(state)
            Utils.xyzprint(state, traj_file)
