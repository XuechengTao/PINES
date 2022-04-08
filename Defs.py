import numpy as np
from inspect import isfunction

def propagator_prototype(state, system, rng=[]):
    return True

def force_engine_prototype(position):
    energy, force = 0., np.array([0., 0., 0.])
    return [energy, force]

"""
System is the class that consists of global system parameters.
"""
class System():

    def __init__(self, args, force_engine, rngseed=42, print_mode="all"):
        self.dt = args.dt
        self.n_steps = args.n_steps
        self.temperature = args.temperature
        self.coupling_time = args.coupling_time
        self.propagator_type_name = args.propagator
        self.propagator = propagator_prototype
        self.thermostat_type_name = args.thermostat
        self.thermostat = propagator_prototype
        self.rngseed = rngseed
        self.print_mode = print_mode

        if (isfunction(force_engine)):
            self.use_external_force_engine = False
            self.force_engine = force_engine
        else:
            self.use_external_force_engine = True
            self.force_engine = force_engine_prototype

        self.n_beads = args.n_beads
        self.cartesian_to_normalmode = np.eye(args.n_beads)
        self.normalmode_to_cartesian = np.eye(args.n_beads)
        self.normalmode_frequency = np.zeros(args.n_beads)
        self.internal_mode_friction = args.trpmd_internal_mode_friction
        self.langevin_damping_weights = np.zeros(args.n_beads)
        self.langevin_stochastic_weights = np.zeros(args.n_beads)

"""
State is the class that consists of information for a snapshot of one dynamics simulation.
"""

class State():

    def __init__(self, system, nuc_mpv):
        self.time = 0.
        self.kin_energy = 0.
        self.pot_energy = 0.
        self.bead_kin_energy = np.zeros(system.n_beads)
        self.bead_pot_energy = np.zeros(system.n_beads)
        self.tot_energy = 0.
        self.bead_force = np.zeros((nuc_mpv[1].shape[0], 3, system.n_beads))

        n_input_dims = len(nuc_mpv[1].shape)
        if (n_input_dims == 2 and system.n_beads == 1):
            print(" Initializing ring-polymers with *centroid* input")
            self.nuclei = self.Nuclei(nuc_mpv[0], np.expand_dims(nuc_mpv[1], axis=2), np.expand_dims(nuc_mpv[2], axis=2))
            self.rdmelectrons = self.RDMElectrons()
        elif (n_input_dims == 3):
            print(" Initializing ring-polymers with *bead* input")
            self.nuclei = self.Nuclei(nuc_mpv[0], nuc_mpv[1], nuc_mpv[2])
            self.rdmelectrons = self.RDMElectrons()
        else:
            exit(" Wrong dimensions of the input of the positions")

    class Nuclei:
        def __init__(self, mass, position, velocity, symbols=''):
            self.mass = mass
            self.position = position
            self.velocity = velocity
            if (len(symbols) > 0):
                self.symbols = symbols
            else:
                self.symbols = ["H" for xatom in range(len(mass))]

        def get_centroid_position(self):
            return np.mean(self.position, axis=2)
        def get_centroid_velocity(self):
            return np.mean(self.velocity, axis=2)

    class RDMElectrons:
        def __init__(self):
            self.rdm =  np.eye(2)
