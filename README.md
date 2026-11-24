# 1D Time-Independent Schrödinger Equation Solver & Convergence Benchmark

A self-contained numerical solver and verification suite for the one-dimensional quantum harmonic oscillator in dimensionless units ($\hbar = m = \omega = 1$).

## Physical Model & Discretization

We solve the stationary Schrödinger eigenvalue equation:

$$\left(-\frac{1}{2}\frac{d^2}{dx^2} + \frac{1}{2}x^2\right)\psi_n(x) = E_n\psi_n(x)$$

on a truncated symmetric domain $[-L, L]$ ($L = 8.0$) with homogeneous Dirichlet boundary conditions $\psi(\pm L) = 0$. Using a second-order central finite-difference stencil on $N$ interior nodes separated by $h = \frac{2L}{N + 1}$, the Hamiltonian reduces to a real symmetric tridiagonal matrix with:

* **Main diagonal:** $d_i = \frac{1}{h^2} + \frac{1}{2}x_i^2$
* **Sub/super-diagonal:** $e_i = -\frac{1}{2h^2}$

The lowest $k$ bound eigenvalues and eigenstates are extracted in $O(N \cdot k)$ operations using `scipy.linalg.eigh_tridiagonal`.

## Analytical Verification Checks

Running `schrodinger_1d_solver.py` executes three automated scientific validation checks:

1. **Eigenvalue Accuracy:** Compares numerical eigenvalues for $n = 0 \dots 5$ ($N = 800$) against exact analytical values $E_n = n + \tfrac{1}{2}$.
2. **Eigenstate Orthonormality:** Verifies that $\int_{-L}^{L} \psi_m(x)\psi_n(x)\,dx = \delta_{mn}$ holds to machine precision ($< 10^{-12}$).
3. **Asymptotic Grid Convergence:** Halves the grid spacing $h$ across $N \in \{200, 400, 800, 1600\}$ and verifies that the ground-state truncation error decays at the theoretical second-order rate $O(h^2)$ ($\log_2(\text{error ratio}) \approx 2.00$).

## Usage

```bash
pip install numpy scipy matplotlib
python schrodinger_1d_solver.py
