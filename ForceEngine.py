import Utils

import numpy as np
import os, shutil, subprocess, re

# '''
# Simple force engines are explicitly coded.
# '''

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

# '''
# Force engine interface to ORCA quantum chemistry program
# '''
def find_orca_energy(enumerate_data, datafile):
    pattern = '# The current total energy in Eh'
    start_number = Utils.find_line_number(enumerate_data, pattern)
    energy = np.loadtxt(datafile, skiprows=start_number+2, max_rows=1, usecols=0)
    return energy

def find_orca_force(enumerate_data, datafile, n_atoms):
    pattern = '# The current gradient in Eh/bohr'
    start_number = Utils.find_line_number(enumerate_data, pattern)
    force = - np.loadtxt(datafile, skiprows=start_number+2, max_rows=3*n_atoms, usecols=0)
    return np.reshape(force, (n_atoms, 3))

def make_orca_force_engine(chem_symbols):
    work_directory = os.getcwd()
    orca_exe = work_directory + '/orca/orca'
    orca_jobinput_source = work_directory + '/examples/orca_energyforce.inp'
    orca_workspace = work_directory + '/orca_tmp'
    os.makedirs(orca_workspace, exist_ok=True)
    orca_jobinput = orca_workspace + '/orca_energyforce.inp'
    shutil.copyfile(orca_jobinput_source, orca_jobinput)
    orca_command = orca_exe + ' ' + orca_jobinput

    def orca_force_engine(position):
        os.chdir(orca_workspace)
        with open(orca_workspace+'/geometry.xyz', 'w') as xyzfile:
            print(len(chem_symbols), '\n', file = xyzfile)
            Utils.xyzprint(position, chem_symbols, print_place = xyzfile)
        output = subprocess.Popen([orca_command], shell=True, \
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]

        orca_datafile = orca_workspace + '/orca_energyforce.engrad'
        with open (orca_datafile, 'r') as data:
            enumerate_data = enumerate(data)
            energy, force = find_orca_energy(enumerate_data, orca_datafile), \
                            find_orca_force (enumerate_data, orca_datafile, len(chem_symbols))

            with open(orca_workspace+'/orca.log', 'a') as logfile:
                output = output.decode('utf-8')
                # for match in re.findall("FINAL SINGLE POINT ENERGY \s* [-+]?(?:\d*\.\d+|\d+)", output, re.S):
                print(output, file=logfile)
                print(" ------ SUMMARIZING ORCA SINGLE POINT CALCULATION -------", file=logfile)
                print("ENERGY", energy, file=logfile)
                Utils.mprint(force, logfile)

            return [energy, force]

    return orca_force_engine
