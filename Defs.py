import numpy as np

"""
System is the class that consists of global system parameters.
"""
class System():

    def __init__(self, temperature, coupling_time):
        self.temperature = temperature
        self.coupling_time = coupling_time

"""
State is the class that consists of information for a snapshot of one dynamics simulation.
"""

class State():

    def __init__(self, nuc_mpv):
        self.time = 0.
        self.kin_energy = 0.
        self.pot_energy = 0.
        self.tot_energy = 0.

        self.nuclei = self.Nuclei(nuc_mpv[0], nuc_mpv[1], nuc_mpv[2])
        self.rdmelectrons = self.RDMElectrons()

    class Nuclei:
        def __init__(self, mass, position, velocity):
            self.mass = mass
            self.position = position
            self.velocity = velocity

    class RDMElectrons:
        def __init__(self):
            self.rdm =  np.eye(2)


