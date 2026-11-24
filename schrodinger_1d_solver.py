#!/usr/bin/env python3
"""
1D Time-Independent Schrodinger Equation Solver & Convergence Benchmark.

Solves the dimensionless quantum harmonic oscillator eigenvalue problem
using a second-order central finite-difference discretization and validates
numerical eigenvalues, eigenstate orthonormality, and O(h^2) grid
convergence against exact analytical solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal


@dataclass
class QHOResult:
    """Container for numerical and analytical harmonic oscillator results."""

    x: np.ndarray
    h: float
    numerical_energies: np.ndarray
    analytical_energies: np.ndarray
    abs_errors: np.ndarray
    wavefunctions: np.ndarray


def solve_qho_fd(
    n_grid: int = 800,
    x_max: float = 8.0,
    n_states: int = 6,
) -> QHOResult:
    """
    Solve the 1D quantum harmonic oscillator in dimensionless units (hbar = m = omega = 1).

    Hamiltonian:
        H = -0.5 * d^2/dx^2 + 0.5 * x^2

    Parameters
    ----------
    n_grid : int
        Number of interior grid points.
    x_max : float
        Half-width of the symmetric spatial domain [-x_max, x_max].
    n_states : int
        Number of lowest bound eigenstates to compute.

    Returns
    -------
    QHOResult
        Dataclass containing spatial grid, eigenvalues, error metrics, and
        L2-normalized eigenstates.
    """
    if n_grid <= n_states:
        raise ValueError("n_grid must be strictly greater than n_states.")
    if x_max <= 0.0:
        raise ValueError("x_max must be strictly positive.")

    # Interior grid points with Dirichlet boundary conditions psi(+-x_max) = 0
    h = (2.0 * x_max) / (n_grid + 1)
    x = np.linspace(-x_max + h, x_max - h, n_grid)

    # Dimensionless harmonic potential V(x) = 0.5 * x^2
    potential = 0.5 * x**2

    # Second-order central difference symmetric tridiagonal Hamiltonian
    diagonal = (1.0 / h**2) + potential
    off_diagonal = np.full(n_grid - 1, -0.5 / h**2)

    energies, eigvecs = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        eigvals_only=False,
        select="i",
        select_range=(0, n_states - 1),
    )

    # Normalize eigenstates in continuous L2 norm: integral |psi_n(x)|^2 dx = 1
    wavefunctions = eigvecs / np.sqrt(h)
    for idx in range(n_states):
        peak_idx = int(np.argmax(np.abs(wavefunctions[:, idx])))
        if wavefunctions[peak_idx, idx] < 0.0:
            wavefunctions[:, idx] *= -1.0

    analytical = np.arange(n_states, dtype=float) + 0.5
    abs_errors = np.abs(energies - analytical)

    return QHOResult(
        x=x,
        h=h,
        numerical_energies=energies,
        analytical_energies=analytical,
        abs_errors=abs_errors,
        wavefunctions=wavefunctions,
    )


def verify_convergence(
    grid_sizes: tuple[int, ...] = (200, 400, 800, 1600),
    x_max: float = 8.0,
) -> list[tuple[int, float, float, float]]:
    """
    Measure ground-state eigenvalue error scaling across doubling grid resolutions.
    """
    results: list[tuple[int, float, float, float]] = []
    prev_error: float | None = None

    for n in grid_sizes:
        sol = solve_qho_fd(n_grid=n, x_max=x_max, n_states=1)
        error = float(sol.abs_errors[0])
        order = np.log2(prev_error / error) if prev_error is not None else float("nan")
        results.append((n, sol.h, error, order))
        prev_error = error

    return results


def save_eigenstate_plot(result: QHOResult, output_path: Path) -> bool:
    """
    Render probability densities |psi_n(x)|^2 shifted by energy E_n if matplotlib is available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    mask = np.abs(result.x) <= 4.5
    x_plot = result.x[mask]

    ax.plot(
        x_plot,
        0.5 * x_plot**2,
        color="black",
        linewidth=1.6,
        linestyle="--",
        label=r"Potential $V(x) = \frac{1}{2}x^2$",
    )

    for n, energy in enumerate(result.numerical_energies):
        prob_density = result.wavefunctions[mask, n] ** 2
        ax.axhline(energy, color="gray", linewidth=0.6, alpha=0.6)
        ax.plot(x_plot, energy + 0.85 * prob_density, linewidth=1.5, label=f"n={n} (E={energy:.4f})")

    ax.set_title("1D Quantum Harmonic Oscillator: Eigenstates & Energy Levels")
    ax.set_xlabel("Dimensionless Position $x$")
    ax.set_ylabel("Energy $E_n$ / Shifted Probability Density")
    ax.set_ylim(0.0, float(result.numerical_energies[-1]) + 1.0)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def run_validation_suite() -> None:
    """Execute numerical accuracy, orthonormality, and convergence assertions."""
    solution = solve_qho_fd(n_grid=800, x_max=8.0, n_states=6)

    print("=== 1D Quantum Harmonic Oscillator Eigenvalue Benchmark (N = 800) ===")
    print(f"{'State (n)':<10} {'Numerical E_n':<18} {'Analytical E_n':<18} {'Abs Error':<14}")
    print("-" * 62)
    for n in range(len(solution.numerical_energies)):
        print(
            f"{n:<10d} "
            f"{solution.numerical_energies[n]:<18.10f} "
            f"{solution.analytical_energies[n]:<18.10f} "
            f"{solution.abs_errors[n]:<14.3e}"
        )

    # Check 1: Eigenvalues must agree with E_n = n + 0.5 within 1e-4
    max_energy_err = float(np.max(solution.abs_errors))
    assert max_energy_err < 1e-4, f"Eigenvalue error too large: {max_energy_err}"

    # Check 2: Discrete L2 inner product matrix must equal identity
    overlap = (solution.wavefunctions.T @ solution.wavefunctions) * solution.h
    ortho_err = float(np.max(np.abs(overlap - np.eye(overlap.shape[0]))))
    assert ortho_err < 1e-12, f"Orthonormality violated: max deviation = {ortho_err}"

    print("\n=== Spatial Grid Convergence Verification (Ground State n = 0) ===")
    print(f"{'Grid N':<10} {'Step h':<14} {'Abs Error':<16} {'Empirical Order':<16}")
    print("-" * 58)
    conv_rows = verify_convergence()
    for n, h, err, order in conv_rows:
        order_str = f"{order:.4f}" if not np.isnan(order) else "baseline"
        print(f"{n:<10d} {h:<14.6f} {err:<16.3e} {order_str:<16}")

    # Check 3: Finest refinement step must exhibit second-order O(h^2) convergence
    finest_order = conv_rows[-1][3]
    assert finest_order > 1.99, f"Expected ~2.0 convergence order, got {finest_order:.4f}"

    plot_file = Path(__file__).resolve().parent / "qho_eigenstates.png"
    plot_saved = save_eigenstate_plot(solution, plot_file)

    print("\n=== Validation Summary ===")
    print(f"[PASS] Max eigenvalue error (n=0..5): {max_energy_err:.3e}")
    print(f"[PASS] Max orthonormality error:      {ortho_err:.3e}")
    print(f"[PASS] Asymptotic convergence order:  {finest_order:.4f} (expected 2.0000)")
    if plot_saved:
        print(f"[INFO] Eigenstate figure saved to:    {plot_file.name}")
    else:
        print("[INFO] matplotlib not installed; skipped figure generation.")


if __name__ == "__main__":
    run_validation_suite()
