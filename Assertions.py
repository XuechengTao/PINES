import Pines, Defs, Utils, ForceEngine
import numpy as np
import filecmp, copy

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

def unittest_rpmdintegrator():
    args = Pines.parse_args()
    args.dt = 0.05
    args.n_steps = 200
    args.n_beads = 4
    args.temperature = 300.
    args.propagator = 'exactho'

    Utils.convert_args_to_au(args)
    print ("\n Input system setup:", args)

    system = Defs.System(args, ForceEngine.morse)
    # MASS, POSITION, VELOCITY in au input directly in this test
    # HF molecule, 4-bead ring-polymer with r_{H-F} = 0.8A,
    initial_state = Defs.State(system, \
                        [np.array([18.9984, 1.0079]) * Utils.amu_to_au, \
                        (np.array([[[-0.0430166, 0.03959, 1.6445479], [-0.1867609, 0.1718837, 0.576421]], \
                                   [[0.0295738, 0.0184438, 1.4099395], [0.1283976, 0.0800756, -0.4421548]], \
                                   [[0.0317896, 0.0458088, 1.4447306], [0.1380176, 0.1988836, -0.2911054]], \
                                   [[-0.0183468, -0.1038426, 1.5479057], [-0.0796543, -0.4508428, 0.1568392]]]).transpose(1,2,0)), \
                        (np.array([[[-0.0000692, 0.0000715, -0.0003107], [-0.0003003, 0.0003105, -0.0013489]], \
                                  [[-0.0001785, -0.0001158, -0.0000981], [-0.0007748, -0.0005028, -0.0004258]], \
                                  [[0.0000111, -0.0001359, 0.000488], [0.0000482, -0.0005902, 0.0021187]], \
                                  [[0.0002365, 0.0001802, -0.0000792], [0.0010269, 0.0007824, -0.0003439]]]).transpose(1,2,0)) \
                        ])

    state = copy.deepcopy(initial_state)
    final_state = Pines.pines(system, state)
    final_state_position, final_state_velocity = final_state.nuclei.get_centroid_position(), final_state.nuclei.get_centroid_velocity()

    # print(final_state_position[0, 2] - 1.54271804, \
    #       final_state_position[1, 2] + 0.583181911, \
    #       final_state_velocity[0, 2] + 0.000134305886, \
    #       final_state_velocity[1, 2] - 0.00253181108, \
    #       final_state.tot_energy - 0.474321302 / 4.)

    print("\n Assertion 2 - RPMD ExactHO integrator: ", filecmp.cmp('trajectory.xyz', 'examples/test_traj2.dat'), "\n")

    args.propagator = 'cayley'
    print ("\n Input system setup:", args)
    system = Defs.System(args, ForceEngine.morse)
    state = copy.deepcopy(initial_state)
    final_state = Pines.pines(system, state)
    final_state_position, final_state_velocity = final_state.nuclei.get_centroid_position(), final_state.nuclei.get_centroid_velocity()

    # print(final_state_position[0, 2] - 1.54271804, \
    #       final_state_position[1, 2] + 0.5831819104, \
    #       final_state_velocity[0, 2] + 0.000134305886, \
    #       final_state_velocity[1, 2] - 0.00253181108, \
    #       final_state.tot_energy - 0.474321302 / 4.)

    print("\n Assertion 3 - RPMD Cayley integrator: ", filecmp.cmp('trajectory.xyz', 'examples/test_traj3.dat'), "\n")

if __name__ == "__main__":

    unittest_mdintegrator()
    unittest_rpmdintegrator()
