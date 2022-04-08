# Driver for Path-Integral calculations iNvolving Excited States (PINES)
# Xuecheng Tao, xtao@caltech.edu

import Defs, Utils, ForceEngine

import numpy as np
import argparse, os
from argparse import RawTextHelpFormatter

def parse_args():
    parser = argparse.ArgumentParser(description='Read in customized system parameters', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dt',     dest='dt', type=float, default=0.1, help="Propagation timestep in fs")
    parser.add_argument('--nsteps', dest='n_steps', type=int, default=25, help="Number of steps in integers")
    parser.add_argument('--temp',   dest='temperature', type=float, default=300., help="System temperature in K")
    parser.add_argument('--tau',    dest='coupling_time', type=float, default=100., help="Coupling time in fs")
    parser.add_argument('--nbeads', dest='n_beads', type=int, default=1, help="Number of ring-polymer beads")
    parser.add_argument('--therm',  dest='thermostat', type=str, choices=["none", "andersen", "internal", "langevin"], \
                                    default="none", help="Whether to turn on the thermostats for centroid and internal modes DOFs")
    parser.add_argument('--prop',   dest='propagator', type=str, choices=["exactho", "cayley"], \
                                    default="exactho", help="The type of the propagator")
    parser.add_argument('--lambda', dest='trpmd_internal_mode_friction', type=float,
                                    default=0.5, help="parameter to define the friction matrix for T-RPMD internal mode thermostating \n \
                                                       if lambda = 0, it reduces to the conventional NVE RPMD")
                                    # This parameter is introduced as \lambda in eq. 30 of Rossi et. al. JCP 140(23), 234116 (2014).
    return parser.parse_args()

def initialize_system(system, chem_symbols, cwd):

    n_atoms = len(chem_symbols)

    def CartesianToNormalmodeMatrix(n_beads):
        if (n_beads == 1):
            return np.eye(1)
        if (n_beads % 2 != 0):
            exit("Only even number of ring-polymer beads is allowed")

        transformation_matrix = np.eye(n_beads)
        normalization_factor = np.sqrt(1. / n_beads)
        phase = 2. * np.pi / n_beads
        for irow in range(n_beads):
            transformation_matrix[irow, 0] = normalization_factor
            transformation_matrix[irow, n_beads // 2] = normalization_factor * np.power(-1, irow+1)
            for icol in range(1, n_beads // 2):
                transformation_matrix[irow, icol] = np.sqrt(2.) * normalization_factor * np.cos(phase * (irow+1) *icol)
                transformation_matrix[irow, icol+ n_beads // 2] = np.sqrt(2.) * normalization_factor * np.sin(phase * (irow+1) *icol)
        return transformation_matrix

    def NormalModeFrequency(n_beads, temperature):
        omega_n =  temperature * n_beads
        return 2. * omega_n * np.sin(np.arange(n_beads) * np.pi / n_beads)

    def LangevinDampingWeights(modes_friction, thermostat_type, dt, coupling_time):
        if (thermostat_type == 'langevin'):
            modes_friction[0] = 1. / coupling_time
        weight_damping    = np.exp(- modes_friction * dt)
        weight_stochastic = np.sqrt(1. - np.square(weight_damping))
        return [weight_damping, weight_stochastic]

    system.cartesian_to_normalmode = CartesianToNormalmodeMatrix(system.n_beads)
    system.normalmode_to_cartesian = np.transpose(system.cartesian_to_normalmode)
    system.normalmode_frequency = NormalModeFrequency(system.n_beads, system.temperature)
    system.langevin_damping_weights, system.langevin_stochastic_weights = \
           LangevinDampingWeights(system.normalmode_frequency * 2. * system.internal_mode_friction, \
                                  system.thermostat_type_name, system.dt, system.coupling_time)

    if (system.propagator_type_name == "exactho"):
        system.propagator = free_ring_polymer_propagation_exactho
    elif (system.propagator_type_name == "cayley"):
        system.propagator = free_ring_polymer_propagation_sqrtcayley

    if (system.thermostat_type_name == "none"):
        system.thermostat = Defs.propagator_prototype
    elif (system.thermostat_type_name == "andersen"):
        system.thermostat = thermostat_andersen
    elif (system.thermostat_type_name == "internal" or system.thermostat_type_name == "langevin"):
        system.thermostat = thermostat_langevin

    if (system.use_external_force_engine):
        print(" Buddy call for *ORCA Force Engine* has been sent out. \n")
        system.force_engine = ForceEngine.make_orca_force_engine(chem_symbols)
        orca_log_file = cwd + '/orca_tmp/orca.log'
        if os.path.exists(orca_log_file):
            os.remove(orca_log_file)

# Andersen (1980) JCP 72(4), 2384–2393
def thermostat_andersen(state, system, rng):
    if (state.time % system.coupling_time) < system.dt:
        sigma = np.sqrt(system.temperature * system.n_beads / state.nuclei.mass)
        for iatom in range(len(state.nuclei.mass)):
            state.nuclei.velocity[iatom,:,:] = rng.normal(0., sigma[iatom], (3, system.n_beads))

# Bussi and Parrinello (2007) Phys. Rev. E, 75, 056707.
# Ceriotti et.al. (2010) JCP 133, 124104.
# Rossi et. al. JCP 140(23), 234116 (2014).
def thermostat_langevin(state, system, rng):
    nm_velocity = np.einsum('ijk,kl->ijl', state.nuclei.velocity, system.cartesian_to_normalmode)
    nm_velocity = nm_velocity * system.langevin_damping_weights[None,None,:]

    sigma = np.sqrt(system.temperature * system.n_beads / state.nuclei.mass)
    for iatom in range(len(state.nuclei.mass)):
        nm_velocity[iatom,:,:] += rng.normal(0., sigma[iatom], (3, system.n_beads)) * system.langevin_stochastic_weights[None, :]
    state.nuclei.velocity = np.einsum('ijk,kl->ijl', nm_velocity, system.normalmode_to_cartesian)

# Harmonic equation of motion
# q(t) = q(0) cos(wt) + v sin(wt) / w; v(t) = v(0) cos (wt) - w q(0) sin(wt)
def free_ring_polymer_propagation_exactho(state, system, evolution_time):
    nm_position = np.einsum('ijk,kl->ijl', state.nuclei.position, system.cartesian_to_normalmode)
    nm_velocity = np.einsum('ijk,kl->ijl', state.nuclei.velocity, system.cartesian_to_normalmode)

    if (system.n_beads == 1):
        nm_position_new = nm_position + evolution_time * nm_velocity
        nm_velocity_new = nm_velocity
    else:
        cos_freqt = np.cos(system.normalmode_frequency * evolution_time)
        sin_freqt = np.sin(system.normalmode_frequency * evolution_time)
        nm_position_new = np.empty_like(nm_position)
        nm_position_new[:,:,0] = nm_position[:,:,0] + evolution_time * nm_velocity[:,:,0]
        nm_position_new[:,:,1:] = nm_position[:,:,1:] * cos_freqt[None,None,1:] \
                               + nm_velocity[:,:,1:] * (sin_freqt[1:] / system.normalmode_frequency[1:])[None,None,:]
        nm_velocity_new = nm_velocity[:,:,:] * cos_freqt[None,None,:] \
                        - nm_position[:,:,:] * (sin_freqt * system.normalmode_frequency)[None,None,:]

    state.nuclei.position = np.einsum('ijk,kl->ijl', nm_position_new, system.normalmode_to_cartesian)
    state.nuclei.velocity = np.einsum('ijk,kl->ijl', nm_velocity_new, system.normalmode_to_cartesian)

# Propagator (I - A dt)^(-1) . (I + A dt), A = [[0, 1], [-w^2, 1]]
# The evolution matrix is [[1 - 0.25 dt^2 w^2, dt], [-dt w^2, 1 - 0.25 dt^2 w^2]] / (1 + 0.25 dt^2 w^2)
def free_ring_polymer_propagation_cayley(state, system, evolution_time):
    nm_position = np.einsum('ijk,kl->ijl', state.nuclei.position, system.cartesian_to_normalmode)
    nm_velocity = np.einsum('ijk,kl->ijl', state.nuclei.velocity, system.cartesian_to_normalmode)

    # Harmonic equation of motion: approximated by Cayley transformation
    cayley_denominator = np.reciprocal(np.square(0.5 * evolution_time * system.normalmode_frequency) + np.ones(system.n_beads))
    cos_component = 2. * cayley_denominator - np.ones(system.n_beads)
    nm_position_new = nm_position * cos_component[None,None,:] \
                    + nm_velocity * (cayley_denominator * evolution_time)[None,None,:]
    nm_velocity_new = nm_velocity * cos_component[None,None,:] \
                    - nm_position * (cayley_denominator * evolution_time * np.square(system.normalmode_frequency))[None,None,:]

    state.nuclei.position = np.einsum('ijk,kl->ijl', nm_position_new, system.normalmode_to_cartesian)
    state.nuclei.velocity = np.einsum('ijk,kl->ijl', nm_velocity_new, system.normalmode_to_cartesian)

# The evolution matrix is [[1, dt/2], [-dt w^2, 1]] / sqrt(1 + 0.25 dt^2 w^2)
def free_ring_polymer_propagation_sqrtcayley(state, system, half_evolution_time):
    nm_position = np.einsum('ijk,kl->ijl', state.nuclei.position, system.cartesian_to_normalmode)
    nm_velocity = np.einsum('ijk,kl->ijl', state.nuclei.velocity, system.cartesian_to_normalmode)

    # Harmonic equation of motion: approximated by Cayley transformation
    sqrtcayley_denominator = np.reciprocal(np.sqrt(np.square(half_evolution_time * system.normalmode_frequency) + np.ones(system.n_beads)))
    nm_position_new = nm_position * sqrtcayley_denominator[None,None,:] \
                    + nm_velocity * (sqrtcayley_denominator * half_evolution_time)[None,None,:]
    nm_velocity_new = nm_velocity * sqrtcayley_denominator[None,None,:] \
                    - nm_position * (sqrtcayley_denominator * half_evolution_time * np.square(system.normalmode_frequency))[None,None,:]

    state.nuclei.position = np.einsum('ijk,kl->ijl', nm_position_new, system.normalmode_to_cartesian)
    state.nuclei.velocity = np.einsum('ijk,kl->ijl', nm_velocity_new, system.normalmode_to_cartesian)

# Propagating the ring-polymer system for a period of dt,
# With the total Liouvillian split into (1) velocity update part (according to potential, denoted as B),
#                                       (2) exact free ring-polymer evolution (denoted as A),
#                                       (3) and the Cayley modification of propagator (2), (denoted as C);
# Three propagation schemes are available: BA[O]AB, BC[O]CB, with or without thermostat.
# Leimkuhler and Matthews (2013) AMREX, 34–56.
# Korol, Bou-Rabee, Miller JCP 151, 124103 (2019)
# Korol, Rosa-Raices, Bou-Rabee, Miller (2020) JCP 152, 104102
# Rosa-Raices, Sun, Bou-Rabee, Miller (2021) JCP 154, 024106

def integrator(state, system, rng):
    state.time += system.dt
    reciprocal_mass = np.reciprocal(state.nuclei.mass)

    # Step "B"
    state.nuclei.velocity[:,:,:] += 0.5 * system.dt * state.bead_force[:,:,:] * reciprocal_mass[:,None,None]
    # Step "A"/"C"
    system.propagator(state, system, 0.5 * system.dt)

    # Step "O"
    system.thermostat(state, system, rng)

    # Step "A"/"C"
    system.propagator(state, system, 0.5 * system.dt)

    # Force evaluation from quantum chemistry, AVAILABLE FOR PARALLELIZATION
    ###################################
    for ibead in range(system.n_beads):
        state.bead_pot_energy[ibead], state.bead_force[:,:,ibead] = system.force_engine(state.nuclei.position[:,:,ibead])
    ###################################

    # Step "B"
    state.nuclei.velocity[:,:,:] += 0.5 * system.dt * state.bead_force[:,:,:] * reciprocal_mass[:,None,None]

    Utils.update_state_energy(state, system)

# The main driver for the dynamics simulation
def pines(system, state):
    print("\n ======= RUNNING PINES ====== \n")

    cwd = os.getcwd()
    initialize_system(system, state.nuclei.symbols, cwd)
    # Prepare the system and the output for the t=0 step
    print (" Trajectory simulation starts successfully. \n")
    rng =  np.random.default_rng(system.rngseed)
    results_set = []
    for ibead in range(system.n_beads):
        state.bead_pot_energy[ibead], state.bead_force[:,:,ibead] = system.force_engine(state.nuclei.position[:,:,ibead])
    Utils.update_state_energy(state, system)

    traj_file_name = cwd + '/trajectory.xyz'
    with open(traj_file_name, "w") as traj_file:
        if ((system.print_mode == "all")):
            print ("%16s%16s%16s%16s" % ("#time (fs)", "kinetic", "potential", "total energy"))
            Utils.inter_print(state, system)
        if ((system.print_mode == "all") or (system.print_mode == "traj_only")):
            Utils.state_centroid_xyzprint(state, traj_file)

        # Time evolution of the system
        for istep in range(system.n_steps):
            integrator(state, system, rng)
            results_set.append(Utils.get_state_properties(state, system))

            if ((system.print_mode == "all")):
                Utils.inter_print(state, system)
            if ((system.print_mode == "all") or (system.print_mode == "traj_only")):
                Utils.state_centroid_xyzprint(state, traj_file)

    return [state, np.array(results_set)]
