import numpy as np
import Utils

def calc_distance(p1, p2):
    return np.sqrt(np.dot(p1-p2, p1-p2))

# Harmonic force engine
def harmonic(position):
# V = 1/2 m omega^2 x^2, return -dV/dx = -m omega^2 x
# w = 3333.33 cm^-1
    omega = 3333.3333 * Utils.inv_cm_to_freq_au
    return [0.5 * Utils.amu_to_au * omega**2 * np.sum(np.square(position)),
            -position * Utils.amu_to_au * omega**2]

# Single bond (Diatomic) Morse potential
def morse(position):
    assert(position.shape[0] == 2)

    De, gamma, re = 0.24456, 1.208, 1.732
    bond_distance = calc_distance(position[0, :], position[1, :])
    tmp = np.exp(-gamma * (bond_distance - re))
    energy = De * (1. - tmp) * (1. - tmp)

    derivative = 2. * gamma * De * (1. - tmp) * tmp
    force = np.empty_like(position)
    force[0, :] = -derivative * (position[0, :] - position[1, :]) / bond_distance
    force[1, :] = -force[0, :]

    return [energy, force]

def update_state_energy(state, system):

    for ibead in range(system.n_beads):
        state.bead_kin_energy[ibead] = 0.5 * np.sum(np.matmul(state.nuclei.mass, np.square(state.nuclei.velocity[:,:,ibead])))
    state.kin_energy = np.average(state.bead_kin_energy)

    state.spr_energy = 0.
    if (system.n_beads > 1):
        for ibead in range(system.n_beads):
            state.spr_energy += np.sum(np.matmul(state.nuclei.mass, \
                                                 np.square(state.nuclei.position[:,:,ibead-1] - state.nuclei.position[:,:,ibead])))
        state.spr_energy *= 0.5 * system.n_beads * system.temperature * system.temperature

    state.pot_energy = np.average(state.bead_pot_energy)
    state.tot_energy = state.kin_energy + state.spr_energy + state.pot_energy
