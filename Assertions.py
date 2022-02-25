import Pines, Defs, Utils
import numpy as np
import filecmp

def unittest_mdintegrator():
    args = Pines.parse_args()
    args.nsteps = 150
    args.temperature = 0.
    system = Defs.System(args.temperature, args.coupling_time)
    # MASS, POSITION, VELOCITY in amu, A, A / fs
    state = Defs.State([np.array([1., 2.]) * Utils.amu_to_au, \
                        (np.array([[1., 0., 0.], [0., 1., 0.]])) * Utils.angstrom_to_bohr, \
                        (np.array([[0., 0., 0.], [0., 0., 0.]])) * Utils.angstrom_to_bohr * Utils.autime_to_fs])
    Pines.pines(args, system, state)
    print("\n Assertion 1 - MD integrator: ", filecmp.cmp('trajectory.xyz', 'examples/test_traj1.dat'), "\n")

if __name__ == "__main__":
    
    unittest_mdintegrator()