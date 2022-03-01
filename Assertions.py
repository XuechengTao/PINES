import Pines, Defs, Utils, ForceEngine
import numpy as np
import filecmp

def unittest_mdintegrator():
    args = Pines.parse_args()
    args.n_steps = 150
    args.temperature = 0.
    Utils.convert_args_to_au(args)
    print ("\n Input system setup:", args)

    system = Defs.System(args, ForceEngine.harmonic)
    # MASS, POSITION, VELOCITY in amu, A, A / fs
    state = Defs.State(system, \
                        [np.array([1., 2.]) * Utils.amu_to_au, \
                        (np.array([[1., 0., 0.], [0., 1., 0.]])) * Utils.angstrom_to_bohr, \
                        (np.array([[0., 0., 0.], [0., 0., 0.]])) * Utils.angstrom_to_bohr * Utils.autime_to_fs])
    Pines.pines(system, state)
    print("\n Assertion 1 - MD integrator: ", filecmp.cmp('trajectory.xyz', 'examples/test_traj1.dat'), "\n")

if __name__ == "__main__":
    
    unittest_mdintegrator()