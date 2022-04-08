import Defs
import numpy as np
import sys

# Set up the global settings calculation environment
fprint_format = "%16.8f"
eprint_format = "%16.4e"
tol = 1.e-9

# Set up physical constants
hplanck_si = 6.62607015e-34
hbar_si = hplanck_si / (2. * np.pi)
emass_si = 9.10938370e-31
echarge_si = 1.602176634e-19
clight_si = 299792458.0
kboltz_si = 1.380649e-23
afine = 7.29735257e-3
avogadro = 6.02214076e23

# Set up the unit conversion constants
bohr_to_meter = hbar_si / (emass_si * clight_si * afine)
meter_to_bohr = 1. / bohr_to_meter
bohr_to_angstrom = bohr_to_meter * 1.e10
angstrom_to_bohr = 1. / bohr_to_angstrom

hartree_to_joule = (afine * clight_si) * (afine * clight_si) * emass_si
joule_to_hartree = 1. / hartree_to_joule
hartree_to_ev = hartree_to_joule / echarge_si
ev_to_hartree = 1. / hartree_to_ev
kboltz = kboltz_si * joule_to_hartree

autime_to_second = hbar_si / hartree_to_joule
second_to_autime = 1. / autime_to_second
autime_to_fs = 1.e15 * autime_to_second
fs_to_autime = 1. / autime_to_fs
autime_to_ps = 1.e12 * autime_to_second
ps_to_autime = 1. / autime_to_ps

amu_to_au = 1.e-3 / avogadro / emass_si
au_to_amu = 1. / amu_to_au

freq_au_to_inv_cm = (1. / (2. * np.pi * clight_si * 1.e2 * autime_to_second))
inv_cm_to_freq_au = 1. / freq_au_to_inv_cm

def convert_args_to_au(args):
    args.dt *= fs_to_autime
    args.coupling_time *= fs_to_autime
    args.temperature *= kboltz

def find_line_number(enumerate_data, pattern):
    for number, line in enumerate_data:
        if pattern in line:
            return number

# Matrix printout
def mprint(matrix, print_place=sys.stdout):
    for row in matrix:
        for val in row:
            print(fprint_format % val, end='', file = print_place)
        print('', file = print_place)

def xyzprint(xyz, chem_symbols, print_place=sys.stdout):
    xyz_print = xyz * bohr_to_angstrom
    for irow in range(xyz_print.shape[0]):
        print("%5s" % chem_symbols[irow], end='', file=print_place)
        for val in xyz_print[irow, :]:
            print(fprint_format % val, end='', file=print_place)
        print('', file = print_place)

# Write the trajectory to xyz file
def state_centroid_xyzprint(state, print_place=sys.stdout):
    print(len(state.nuclei.mass), file=print_place)
    print("# at time " + str(round(state.time * autime_to_fs, 3)) + " fs", file = print_place)
    xyzprint(state.nuclei.get_centroid_position(), state.nuclei.symbols, print_place)

# Write intermediate information
def inter_print(state, system):
    print (eprint_format % round(state.time * autime_to_fs, 6), end='')
    print ((3 * eprint_format) % \
           (state.kin_energy, state.pot_energy, state.tot_energy),  end='')
    # USER CUSTOMIZED PART STARTS HERE, e.g.
    print ((1 * eprint_format) % \
           (state.nuclei.position[0, 0, 0]), end='')
    print (eprint_format % (state.kin_energy / 1.5 / system.temperature / system.n_beads))

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

def get_state_properties(state, system):
    quantity2 = state.nuclei.get_centroid_position()[0, 0]

    # USER CUSTOMIZED FUNCTION
    return [state.kin_energy, quantity2]
