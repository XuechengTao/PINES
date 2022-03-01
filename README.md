# PINES
The program aims to provide a simple and readable Python tool to perform path-integral dynamics simulation involving excited states.

<!-- The current features -->
1. Conventional Ring polymer molecular dynamics (RPMD) propagator in NVE/NVT ensemble for ground state simulation
2. The Cayley-modified propagator of the free ring-polymer propagation, such that a larger time step can be used

<!-- To do list -->
1. Adding the test for Cayley NVE trajectory, and make sure that Sqrt_Cayley * Sqrt_Cayley = Cayley
2. Implementing Langevin for ring-polymer internal modes thermostating, and the centroid thermostating 