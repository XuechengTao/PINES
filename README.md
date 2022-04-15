# PINES (Path-Integral iNvolving Excited States)
The program aims to provide a simple and readable Python driver to perform path-integral dynamics simulation involving excited states.

## The current functionalities
1. Conventional **Ring polymer molecular dynamics** (RPMD) propagator in NVE ensemble for ground state simulations
2. **Temperature-conserved** (NVT) simulations from Andersen and Langevin thermostat, also the version with only internal modes thermostating (T-RPMD)
3. The **Cayley-modified propagator** for the free ring-polymer propagation (NVE and BCOCB for NVT), such that larger time steps can be used
4. Interface with **ORCA** quantum chemistry package for the access to **ab initio potential energy surfaces**

## Implementing now
* Add the assertions for the following blocks: Langevin for MD/RPMD, Andersen for MD/RPMD, T-RPMD
* The photons equations of motion, quantum and semiclassical
