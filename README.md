# NUMERICAL-SOLUTION-OF-THE-SCHR-DINGER-EQUATION

This project explores numerical approaches to solving both the Time-Independent and Time-Dependent Schrödinger Equations (TISE and TDSE) in one and two dimensions using finite difference methods and the Crank-Nicolson method. The aim is to simulate quantum mechanical systems, validate numerical results against analytical solutions, and study phenomena such as quantum tunneling, eigenstate behavior, and wave packet evolution under various potentials.

🔬 Features:
1D TISE & TDSE:

Solved for an infinite potential well and validated against analytical solutions.

Introduced perturbations (Half Quantum Simple Harmonic Oscillator) to examine their impact on eigenstates and energy levels.

Simulated quantum tunneling by introducing potential barriers and comparing reflection/transmission coefficients to analytical predictions.

2D TISE & TDSE:

Extended the solver to 2D systems including degenerate eigenstates.

Investigated perturbations and their numerical behavior in higher dimensions.

Simulated a double-slit experiment setup to observe quantum interference patterns.

🛠 Methods:
Finite Difference Method for discretizing spatial derivatives.

Crank-Nicolson Method for stable time evolution in TDSE.

Expansion Method to decompose initial wave packets into eigenstates for TDSE.

Comparison of numerical and analytical solutions through eigenenergy accuracy and wavefunction overlap.

📈 Results:
Eigenenergies in the unperturbed 1D and 2D cases closely matched analytical values.

Crank-Nicolson allowed high-resolution tunneling simulations with excellent accuracy.

Perturbations led to increased deviation in eigenvalues, highlighting the limitations of low-resolution grids and the need for more refined methods in complex systems.
