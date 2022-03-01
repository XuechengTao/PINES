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
meter_to_bohr = 1.0 / bohr_to_meter
bohr_to_angstrom = bohr_to_meter * 1e10
angstrom_to_bohr = 1.0 / bohr_to_angstrom

hartree_to_joule = (afine * clight_si) * (afine * clight_si) * emass_si
joule_to_hartree = 1. / hartree_to_joule
hartree_to_ev = hartree_to_joule / echarge_si
ev_to_hartree = 1. / hartree_to_ev
kboltz = kboltz_si * joule_to_hartree

autime_to_second = hbar_si / hartree_to_joule
second_to_autime = 1.0 / autime_to_second
autime_to_fs = 1.0e15 * autime_to_second
fs_to_autime = 1.0 / autime_to_fs
autime_to_ps = 1.0e12 * autime_to_second
ps_to_autime = 1.0 / autime_to_ps

amu_to_au = 1.0e-3 / avogadro / emass_si
au_to_amu = 1.0 / amu_to_au

freq_au_to_inv_cm = (1.0 / (2 * np.pi * clight_si * 100.0 * autime_to_second))
inv_cm_to_freq_au = 1. / freq_au_to_inv_cm

def convert_args_to_au(args):
    args.dt *= fs_to_autime
    args.coupling_time *= fs_to_autime
    args.temperature *= kboltz

# Matrix printout
def mprint(matrix, print_place = sys.stdout):
    for row in matrix:
        for val in row:
            print(fprint_format % val, end='', file = print_place)
        print('', file = print_place)

# Write intermediate information
def inter_print(state):
    print (eprint_format % round(state.time * autime_to_fs, 6), end='')
    print ((3 * eprint_format) % \
           (state.kin_energy, state.pot_energy, state.tot_energy),  end='')
    print ((2 * eprint_format) % \
           (state.nuclei.position[0, 0, 0], state.nuclei.position[1, 1, 0]))

# Write the trajectory to xyz file
def xyzprint(state, print_place=sys.stdout):
    print(len(state.nuclei.mass), file=print_place)
    print("# at time " + str(round(state.time * autime_to_fs, 3)) + " fs", file = print_place)
    xyz_print = state.nuclei.position * bohr_to_angstrom
    for row in xyz_print:
        print("%5s" % "X", end='', file=print_place)
        for val in row:
            print(fprint_format % val, end='', file=print_place)
        print('', file = print_place)
