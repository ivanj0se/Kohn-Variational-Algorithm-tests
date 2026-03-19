import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial
import time

N_L = 10
gamma = 1.0
mu = 1.0
hbar = 1.0


def V(R):
    V0 = 1.0
    a = 0.5
    return V0 / np.cosh(a * R) ** 2


def norm_factor(l, gamma):
    return np.sqrt((2 * gamma) ** (2 * l - 1) / factorial(2 * l - 2))


def norm_factor_numerical(l, gamma):
    norm_sq, _ = quad(lambda R: (R ** (l - 1) * np.exp(-gamma * R)) ** 2, 0, np.inf)
    return 1.0 / np.sqrt(norm_sq)


def u_basis(R, l, gamma, numerical_norm=False):
    F = norm_factor_numerical(l, gamma) if numerical_norm else norm_factor(l, gamma)
    return F * R ** (l - 1) * np.exp(-gamma * R)


def cutoff(R, R_c=0.5):
    return 1.0 - np.exp(-R / R_c)


def u0(R, k):
    v = hbar * k / mu
    return cutoff(R) * np.exp(-1j * k * R) / np.sqrt(v)


def apply_H_minus_E(psi_R, R, E):
    """Second-order finite differences for d^2/dR^2."""
    dR = R[1] - R[0]
    d2_psi = np.zeros_like(psi_R)
    d2_psi[1:-1] = (psi_R[2:] - 2 * psi_R[1:-1] + psi_R[:-2]) / dR ** 2
    d2_psi[0] = d2_psi[1]
    d2_psi[-1] = d2_psi[-2]
    return -hbar ** 2 / (2 * mu) * d2_psi + V(R) * psi_R - E * psi_R


def d2_u_basis_unnorm(R, l, gamma):
    """Analytical d^2/dR^2 of R^(l-1) e^{-gamma R}."""
    term1 = (l - 1) * (l - 2) * R ** (l - 3) if l >= 3 else 0.0
    term2 = -2 * gamma * (l - 1) * R ** (l - 2) if l >= 2 else 0.0
    term3 = gamma ** 2 * R ** (l - 1)
    return (term1 + term2 + term3) * np.exp(-gamma * R)


def H_minus_E_basis_pointwise(R, l, gamma, Energy, F):
    u = F * R ** (l - 1) * np.exp(-gamma * R)
    d2u = F * d2_u_basis_unnorm(R, l, gamma)
    return -hbar ** 2 / (2 * mu) * d2u + (V(R) - Energy) * u


def d2_u0_pointwise(R, k):
    """Analytical d^2/dR^2 of u_0 via product rule on f(R) e^{-ikR}."""
    v = hbar * k / mu
    R_c = 0.5
    f   =  1.0 - np.exp(-R / R_c)
    fp  =  np.exp(-R / R_c) / R_c
    fpp = -np.exp(-R / R_c) / R_c ** 2
    g   = np.exp(-1j * k * R) / np.sqrt(v)
    gp  = -1j * k * g
    gpp = (-1j * k) ** 2 * g
    return fpp * g + 2 * fp * gp + f * gpp


def H_minus_E_u0_pointwise(R, k, Energy):
    v = hbar * k / mu
    u = cutoff(R) * np.exp(-1j * k * R) / np.sqrt(v)
    return -hbar ** 2 / (2 * mu) * d2_u0_pointwise(R, k) + (V(R) - Energy) * u


def build_matrices(Energy, N_L=N_L, gamma=gamma, R_max=30.0, N_grid=5000):
    """
    M    : (N_L-1) x (N_L-1) real symmetric,  <u_{i+2}|H-E|u_{j+2}>
    M0   : (N_L-1) complex vector,             <u_{i+2}|H-E|u_0>
    M00  : complex scalar,                     <u_0|H-E|u_0>
    """
    k = np.sqrt(2 * mu * Energy) / hbar
    R = np.linspace(1e-10, R_max, N_grid)
    dR = R[1] - R[0]
    n_basis = N_L - 1

    phi = np.zeros((n_basis, N_grid))
    for i, l in enumerate(range(2, N_L + 1)):
        phi[i] = u_basis(R, l, gamma)

    u0_grid = u0(R, k)

    H_phi = np.zeros((n_basis, N_grid))
    for i in range(n_basis):
        H_phi[i] = apply_H_minus_E(phi[i], R, Energy)

    H_u0 = apply_H_minus_E(u0_grid, R, Energy)

    M = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(i, n_basis):
            val = np.trapezoid(phi[i] * H_phi[j], dx=dR)
            M[i, j] = val.real
            M[j, i] = val.real

    M0 = np.zeros(n_basis, dtype=complex)
    for i in range(n_basis):
        M0[i] = np.trapezoid(phi[i] * H_u0, dx=dR)

    M00 = np.trapezoid(np.conj(u0_grid) * H_u0, dx=dR)

    return M, M0, M00


def build_matrices_quad(Energy, N_L=N_L, gamma=gamma, R_max=30.0, numerical_norm=True):
    """
    Same as build_matrices but uses Gaussian quadrature with analytical derivatives.
    numerical_norm=True computes F_l by quadrature; False uses the closed form.
    """
    k = np.sqrt(2 * mu * Energy) / hbar
    n_basis = N_L - 1

    F = [
        norm_factor_numerical(l, gamma) if numerical_norm else norm_factor(l, gamma)
        for l in range(2, N_L + 1)
    ]

    def phi_i(R, i):
        l = i + 2
        return F[i] * R ** (l - 1) * np.exp(-gamma * R)

    def H_phi_i(R, i):
        return H_minus_E_basis_pointwise(R, i + 2, gamma, Energy, F[i])

    def H_u0_R(R):
        return H_minus_E_u0_pointwise(R, k, Energy)

    def u0_conj(R):
        v = hbar * k / mu
        return np.conj(cutoff(R) * np.exp(-1j * k * R) / np.sqrt(v))

    def quad_complex(f, a=1e-10, b=R_max):
        re, _ = quad(lambda R: f(R).real, a, b)
        im, _ = quad(lambda R: f(R).imag, a, b)
        return re + 1j * im

    M = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(i, n_basis):
            val, _ = quad(lambda R, i=i, j=j: phi_i(R, i) * H_phi_i(R, j), 1e-10, R_max)
            M[i, j] = val
            M[j, i] = val

    M0 = np.zeros(n_basis, dtype=complex)
    for i in range(n_basis):
        M0[i] = quad_complex(lambda R, i=i: phi_i(R, i) * H_u0_R(R))

    M00 = quad_complex(lambda R: u0_conj(R) * H_u0_R(R))

    return M, M0, M00


if __name__ == "__main__":
    E_test = 0.5

    t0 = time.perf_counter()
    M, M0, M00 = build_matrices(E_test)
    t_trap = time.perf_counter() - t0

    print(f"\n--- trapezoid  (E={E_test}) ---")
    print(f"M {M.shape}:\n{M[:5, :5]}")
    print(f"M0: {M0[:5]}")
    print(f"M00: {M00:.6f}")
    print(f"max|M-M^T| = {np.max(np.abs(M - M.T)):.2e}  |  {t_trap:.3f} s")

    t0 = time.perf_counter()
    M_q, M0_q, M00_q = build_matrices_quad(E_test, numerical_norm=True)
    t_quad = time.perf_counter() - t0

    print(f"\n--- gaussian quadrature ---")
    print(f"M {M_q.shape}:\n{M_q[:5, :5]}")
    print(f"M0: {M0_q[:5]}")
    print(f"M00: {M00_q:.6f}")
    print(f"max|M-M^T| = {np.max(np.abs(M_q - M_q.T)):.2e}  |  {t_quad:.3f} s")

    print(f"\n--- diff ---")
    print(f"max|ΔM|  = {np.max(np.abs(M - M_q)):.2e}")
    print(f"max|ΔM0| = {np.max(np.abs(M0 - M0_q)):.2e}")
    print(f"|ΔM00|   = {abs(M00 - M00_q):.2e}")
    print(f"speedup  = {t_trap / t_quad:.2f}x")
