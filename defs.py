import numpy as np

# Set up the calculation environment


fprint_format = "%16.8f"

"""
State is the class that consist of information for a snapshot of one dynamics simulation.
"""


class State():

    def __init__(self, nuc_mpv):
        self.time = 0.
        self.nuclei = self.Nuclei(nuc_mpv[0], nuc_mpv[1], nuc_mpv[2])
        self.rdmelectrons = self.RDMElectrons()

    class Nuclei:
        def __init__(self, mass, position, velocity):
            self.mass = mass
            self.position = position
            self.velocity = velocity

    class RDMElectrons:
        def __init__(self):
            self.rdm = np.eye(2)
