import Pines, Defs, Utils, ForceEngine, MultiState
import numpy as np
import math, filecmp, copy, time

def unittest_mdintegrator():
    args = Pines.parse_args()
    args.n_steps = 150
    args.temperature = 300.
    Utils.convert_args_to_au(args)
    print ("\n  Input system setup:", args, "\n")

    system = Defs.System(args, ForceEngine.harmonic, print_mode="traj_only")
    # MASS, POSITION, VELOCITY in amu, A, A / fs
    state = Defs.State(system, \
                        [np.array([1., 2.]) * Utils.amu_to_au, \
                        (np.array([[1., 0., 0.], [0., 1., 0.]])) * Utils.angstrom_to_bohr, \
                        (np.array([[0., 0., 0.], [0., 0., 0.]])) * Utils.angstrom_to_bohr * Utils.autime_to_fs])

    Pines.pines(system, state)

    assert filecmp.cmp('trajectory.xyz', 'examples/test_traj1.dat'), "Fatal Error; Assertion 1 - MD integrator"

def unittest_rpmdintegrator():
    args = Pines.parse_args()
    args.dt = 0.05
    args.n_steps = 200
    args.n_beads = 4
    args.temperature = 300.
    args.propagator = 'exactho'

    Utils.convert_args_to_au(args)
    print ("\n  Input system setup:", args, "\n")

    system = Defs.System(args, ForceEngine.morse, print_mode="traj_only")
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
    final_state, results_set = Pines.pines(system, state)
    final_state_position, final_state_velocity = final_state.nuclei.get_centroid_position(), final_state.nuclei.get_centroid_velocity()

    # print(final_state_position[0, 2] - 1.54271804, \
    #       final_state_position[1, 2] + 0.583181911, \
    #       final_state_velocity[0, 2] + 0.000134305886, \
    #       final_state_velocity[1, 2] - 0.00253181108, \
    #       final_state.tot_energy - 0.474321302 / 4.)   # from an external code

    assert filecmp.cmp('trajectory.xyz', 'examples/test_traj2.dat'), "Fatal Error; Assertion 2 - RPMD ExactHO integrator"

    args.propagator = 'cayley'
    print ("\n  Input system setup:", args, "\n")
    system = Defs.System(args, ForceEngine.morse, print_mode="traj_only")
    state = copy.deepcopy(initial_state)
    final_state, results_set = Pines.pines(system, state)
    final_state_position, final_state_velocity = final_state.nuclei.get_centroid_position(), final_state.nuclei.get_centroid_velocity()

    # print(final_state_position[0, 2] - 1.54271804, \
    #       final_state_position[1, 2] + 0.5831819104, \
    #       final_state_velocity[0, 2] + 0.000134305886, \
    #       final_state_velocity[1, 2] - 0.00253181108, \
    #       final_state.tot_energy - 0.474321302 / 4.)   # from an external code

    assert filecmp.cmp('trajectory.xyz', 'examples/test_traj3.dat'), "Fatal Error; Assertion 3 - RPMD Cayley integrator"

def unittest_thermostat():
    args = Pines.parse_args()
    args.dt = 0.1
    args.coupling_time = 3.
    args.n_steps = 50  # Regression test
    # args.n_steps = 5000
    args.temperature = 3000.
    args.n_beads = 1
    args.thermostat = "andersen"

    Utils.convert_args_to_au(args)
    print ("\n  Input system setup:", args, "\n")

    system = Defs.System(args, ForceEngine.harmonic, rngseed=10, print_mode="traj_only")
    # MASS, POSITION, VELOCITY in amu, A, A / fs
    state = Defs.State(system, \
                        [np.array([1., 6.]) * Utils.amu_to_au, \
                        (np.array([[.2, 0., 0.], [0., .1, 0.]])) * Utils.angstrom_to_bohr, \
                        (np.array([[0., 0., 0.], [0., 0., 0.]])) * Utils.angstrom_to_bohr * Utils.autime_to_fs])

    final_state, results_set = Pines.pines(system, state)

    q_lst = results_set[:, 1]
    q_sqr_mean = np.mean(np.square(q_lst))
    q_sqr_mean_analytic = system.temperature /  state.nuclei.mass[0] / np.power(3333.3333 * Utils.inv_cm_to_freq_au, 2)
                        # Analytical result <q^2> = 1/\beta m \omega^2
    print(q_sqr_mean, q_sqr_mean_analytic)
    # print(np.abs(q_sqr_mean - q_sqr_mean_analytic) / q_sqr_mean_analytic)
    # np.savetxt("test_file", q_lst.T)

    assert filecmp.cmp('trajectory.xyz', 'examples/test_traj4.dat'), "Fatal Error; Assertion 4 - MD thermostat"

    args.n_beads = 4
    system = Defs.System(args, ForceEngine.harmonic, rngseed=10, print_mode="traj_only")
    # MASS, POSITION, VELOCITY in amu, A, A / fs
    state = Defs.State(system, \
                        [np.array([1., 6.]) * Utils.amu_to_au, \
                        (np.array([[[.15, 0., 0.], [0., .07, 0.]], \
                                   [[.18, 0., 0.], [0., .08, 0.]], \
                                   [[.22, 0., 0.], [0., .12, 0.]], \
                                   [[.25, 0., 0.], [0., .13, 0.]]]).transpose(1,2,0) * Utils.angstrom_to_bohr), \
                        (np.array([[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]], \
                                   [[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]).transpose(1,2,0) \
                                   * Utils.angstrom_to_bohr * Utils.autime_to_fs)])

    final_state, results_set = Pines.pines(system, state)

    q_lst = results_set[:, 1]
    q_sqr_mean = np.mean(np.square(q_lst))
    exp_m_beta_omega = np.exp(- 3333.3333 * Utils.inv_cm_to_freq_au / system.temperature)
    q_sqr_mean_analytic = 0.5 * (1. + exp_m_beta_omega) / (1. - exp_m_beta_omega) \
                          / state.nuclei.mass[0] / (3333.3333 * Utils.inv_cm_to_freq_au)
                        # Analytical result <q^2> = 1/(2 m \omega) * (1 + e^{-\beta \omega}) / (1 - e^{-\beta \omega})

    print(q_sqr_mean, q_sqr_mean_analytic)
    # print(np.abs(q_sqr_mean - q_sqr_mean_analytic) / q_sqr_mean_analytic)
    # np.savetxt("test_file2", q_lst.T)

    assert filecmp.cmp('trajectory.xyz', 'examples/test_traj5.dat'), "Fatal Error; Assertion 5 - RPMD thermostat"

def unittest_orca_interface():
    args = Pines.parse_args()
    args.n_steps = 50
    args.dt = 0.1
    args.temperature = 300.
    args.thermostat = "none"
    Utils.convert_args_to_au(args)
    print ("\n  Input system setup:", args, "\n")

    system = Defs.System(args, "ORCA", print_mode="all")
    state = Defs.State(system, \
                        [np.array([1., 1.]) * Utils.amu_to_au, \
                        (np.array([[0.5, 0., 0.], [0., 0., 0.]])) * Utils.angstrom_to_bohr, \
                        (np.array([[0., 0., 0.], [0., 0., 0.]])) * Utils.angstrom_to_bohr * Utils.autime_to_fs])

    final_state, results_set = Pines.pines(system, state)
    final_state_position, final_state_velocity = final_state.nuclei.get_centroid_position(), final_state.nuclei.get_centroid_velocity()

    # REGRESSION TEST
    assert math.isclose(final_state_position[0, 0],  0.90053211 * Utils.angstrom_to_bohr, abs_tol=1e-8), "Fatal Error; Assertion 11 - ORCA interface"
    assert math.isclose(final_state_position[1, 0], -0.40053211 * Utils.angstrom_to_bohr, abs_tol=1e-8), "Fatal Error; Assertion 11 - ORCA interface"
    # assert filecmp.cmp('trajectory.xyz', 'examples/test_traj11.dat'), "Fatal Error; Assertion 11 - ORCA interface"

def unittest_mean_field_force_engine():
    args = Pines.parse_args()
    args.n_beads = 16
    Utils.convert_args_to_au(args)
    print ("\n  Input system setup:", args, "\n")

    system = Defs.System(args, ForceEngine.harmonic, print_mode="all")
    system.temperature = 1. / 3.25
    test_multistate_potential = np.array(  [[[117.0139794, 0.0077], [0.0077, 0.02262994515]], \
                                            [[108.4693221, 0.0077], [0.0077, 0.3530579445]], \
                                            [[95.53328797, 0.0077], [0.0077, 1.713510074]], \
                                            [[79.92213368, 0.0077], [0.0077, 5.103357455]], \
                                            [[63.60360304, 0.0077], [0.0077, 11.53755435]], \
                                            [[48.46356521, 0.0077], [0.0077, 21.76105313]], \
                                            [[36, 0.0077], [0.0077, 36]], \
                                            [[21.372583, 0.0077], [0.0077, 77.9411255]], \
                                            [[20, 0.0077], [0.0077, 100]], \
                                            [[21.372583, 0.0077], [0.0077, 77.9411255]], \
                                            [[36, 0.0077], [0.0077, 36]], \
                                            [[63.60360304, 0.0077], [0.0077, 11.53755435]], \
                                            [[79.92213368, 0.0077], [0.0077, 5.103357455]], \
                                            [[95.53328797, 0.0077], [0.0077, 1.713510074]], \
                                            [[108.4693221, 0.0077], [0.0077, 0.3530579445]], \
                                            [[117.0139794, 0.0077], [0.0077, 0.02262994515]]]  )
    test_multistate_gradient_tensor_1d = np.array(   [[[[[19.69913495, 0], [0, -0.3008650538]], \
                                                        [[18.81162641, 0], [0, -1.188373585]], \
                                                        [[17.38197779, 0], [0, -2.61802221]], \
                                                        [[15.48187762, 0], [0, -4.518122378]], \
                                                        [[13.20660487, 0], [0, -6.793395131]], \
                                                        [[10.67025121, 0], [0, -9.329748793]], \
                                                        [[8, 0], [0, -12]], \
                                                        [[2.343145751, 0], [0, -17.65685425]], \
                                                        [[0, 0], [0, -20]], \
                                                        [[2.343145751, 0], [0, -17.65685425]], \
                                                        [[8, 0], [0, -12]], \
                                                        [[13.20660487, 0], [0, -6.793395131]], \
                                                        [[15.48187762, 0], [0, -4.518122378]], \
                                                        [[17.38197779, 0], [0, -2.61802221]], \
                                                        [[18.81162641, 0], [0, -1.188373585]], \
                                                        [[19.69913495, 0], [0, -0.3008650538]]]]]  )
    test_multistate_gradient_tensor = np.concatenate((test_multistate_gradient_tensor_1d, \
                                                      np.zeros_like(test_multistate_gradient_tensor_1d), \
                                                      np.zeros_like(test_multistate_gradient_tensor_1d)), axis=1)

    mean_field_energy, mean_field_force = MultiState.mean_field_force_engine(test_multistate_potential, test_multistate_gradient_tensor, system)
    jax_mean_field_energy, jax_mean_field_force = MultiState.jax_mean_field_force_engine(test_multistate_potential, test_multistate_gradient_tensor, system)
    assert np.isclose(mean_field_energy, 15.97344881, atol=1e-8), "Fatal Error; Assertion 12 - JAX auto differentiation"  ## from an external code
    assert np.isclose(jax_mean_field_energy, 15.97344881, atol=1e-8), "Fatal Error; Assertion 12 - JAX auto differentiation"  ## from an external code

    gradient_result = np.array([-0.01880406043, -0.07427334271, -0.1636263797, -0.2823826347, -0.4245483971, \
                                -0.5502163462, -0.07732925642, 0.1389980882, -9.123476373e-08, 0.1384344144, \
                                -0.1210193967, -0.4151299456, -0.2823825183, -0.1636263797, -0.07427334271, \
                                -0.01880406043]) ## from an external code
    assert np.isclose(jax_mean_field_force[0,0,:], gradient_result, atol=1e-12).all(), "Fatal Error; Assertion 12 - JAX auto differentiation"  ## from an external code

if __name__ == "__main__":

    # unittest_mdintegrator()
    # unittest_rpmdintegrator()
    # unittest_thermostat()
    # unittest_orca_interface()

    # TIME TEST ZONE
    start_time = time.time()
    ########################################################
    unittest_mean_field_force_engine()
    ########################################################
    print("--- test finished in %s seconds ---" % (time.time() - start_time))
