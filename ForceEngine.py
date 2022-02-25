import numpy as np
import Utils

# Harmonic force engine
def harmonic(position):
# V = 1/2 m omega^2 x^2, return -dV/dx = -m omega^2 x
    omega = 3333.3333 * Utils.inv_cm_to_freq_au
    return [0.5 * Utils.amu_to_au * omega**2 * np.sum(np.square(position)),
            -position * Utils.amu_to_au * omega**2]

def update_state_energy(state, pot_energy):
    state.kin_energy = 0.5 * np.sum(np.matmul(state.nuclei.mass, np.square(state.nuclei.velocity)))
    state.pot_energy = pot_energy
    state.tot_energy = state.kin_energy + state.pot_energy