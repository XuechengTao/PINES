import Utils

import numpy as np
import scipy.linalg as linalg

from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad as jgrad
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

# INPUT arguments
# multistate_potential[:,:,:], n_beads x n_states x n_states
# multistate_gradient_tensor[:,:,:,:,:], n_atoms x n_dims(3) x n_beads x n_states x n_states 
def mean_field_force_engine(multistate_potential, multistate_gradient_tensor, system):
    n_beads = system.n_beads
    beta_n = 1. / system.temperature / n_beads
    bead_weight = []
    for ibead in range(n_beads):
        bead_weight.append(linalg.expm(multistate_potential[ibead] * (-beta_n)))
    mean_field_weight = np.trace(np.linalg.multi_dot(np.array(bead_weight)))
    mean_field_energy = - system.temperature * np.log(mean_field_weight)
    return [mean_field_energy, []]

def jax_mean_field_energy(multistate_potential, system):
    n_beads = system.n_beads
    beta_n = 1. / system.temperature / n_beads
    
    def single_bead_weight(bead_multistate_potential):
        return jlinalg.expm(bead_multistate_potential * (-beta_n))
    bead_weight = jnp.array([single_bead_weight(pmatrix) for pmatrix in multistate_potential])
    mean_field_weight = jnp.trace(jnp.linalg.multi_dot(bead_weight))
    mean_field_energy = - system.temperature * jnp.log(mean_field_weight)
    return mean_field_energy

def jax_mean_field_force_engine(multistate_potential, multistate_gradient_tensor, system):
    mean_field_energy = jax_mean_field_energy(multistate_potential, system)
    operation_gradient = jgrad(jax_mean_field_energy)(multistate_potential, system)
    mean_field_gradient = np.einsum('rij,pqrij->pqr', operation_gradient, multistate_gradient_tensor)
    #                                                    [ibead,:,:]        [iatom,idim,ibead,:,:]
    return [mean_field_energy, mean_field_gradient]