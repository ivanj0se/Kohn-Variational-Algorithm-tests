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


def u_basis(R, l, gamma):
    F = norm_factor(l, gamma)
    return F * R ** (l - 1) * np.exp(-gamma * R)


def cutoff(R, R_c=0.5):
    return 1.0 - np.exp(-R / R_c)


def u0(R, k):
    v = hbar * k / mu
    return cutoff(R) * np.exp(-1j * k * R) / np.sqrt(v)


def d2_u_basis_unnorm(R, l, gamma):
    term1 = (l - 1) * (l - 2) * R ** (l - 3) if l >= 3 else 0.0
    term2 = -2 * gamma * (l - 1) * R ** (l - 2) if l >= 2 else 0.0
    term3 = gamma ** 2 * R ** (l - 1)
    return (term1 + term2 + term3) * np.exp(-gamma * R)


def H_minus_E_basis(R, l, gamma, Energy):
    F = norm_factor(l, gamma)
    u = F * R ** (l - 1) * np.exp(-gamma * R)
    d2u = F * d2_u_basis_unnorm(R, l, gamma)
    return -hbar ** 2 / (2 * mu) * d2u + (V(R) - Energy) * u


def d2_u0(R, k):
    v = hbar * k / mu
    R_c = 0.5
    f   =  1.0 - np.exp(-R / R_c)
    fp  =  np.exp(-R / R_c) / R_c
    fpp = -np.exp(-R / R_c) / R_c ** 2
    g   = np.exp(-1j * k * R) / np.sqrt(v)
    gp  = -1j * k * g
    gpp = (-1j * k) ** 2 * g
    return fpp * g + 2 * fp * gp + f * gpp


def H_minus_E_u0(R, k, Energy):
    v = hbar * k / mu
    u = cutoff(R) * np.exp(-1j * k * R) / np.sqrt(v)
    return -hbar ** 2 / (2 * mu) * d2_u0(R, k) + (V(R) - Energy) * u


def build_matrices(Energy, N_L=N_L, gamma=gamma, R_max=30.0, N_grid=5000):
    """Trapezoidal integration with analytical derivatives."""
    k = np.sqrt(2 * mu * Energy) / hbar
    R = np.linspace(1e-10, R_max, N_grid)
    dR = R[1] - R[0]
    n = N_L - 1

    # Evaluate basis functions and (H-E) acting on them
    phi = np.zeros((n, N_grid))
    H_phi = np.zeros((n, N_grid))
    for i, l in enumerate(range(2, N_L + 1)):
        phi[i] = u_basis(R, l, gamma)
        H_phi[i] = H_minus_E_basis(R, l, gamma, Energy)

    u0_grid = u0(R, k)
    H_u0 = H_minus_E_u0(R, k, Energy)

    # M matrix
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.trapezoid(phi[i] * H_phi[j], dx=dR)
            M[i, j] = val.real
            M[j, i] = val.real

    # M0 vector
    M0 = np.zeros(n, dtype=complex)
    for i in range(n):
        M0[i] = np.trapezoid(phi[i] * H_u0, dx=dR)

    # M00 scalar
    M00 = np.trapezoid(np.conj(u0_grid) * H_u0, dx=dR)

    return M, M0, M00


def build_matrices_quad(Energy, N_L=N_L, gamma=gamma, R_max=30.0):
    """Gaussian quadrature with analytical derivatives."""
    k = np.sqrt(2 * mu * Energy) / hbar
    n = N_L - 1

    def quad_complex(f, a=1e-10, b=R_max):
        re, _ = quad(lambda R: f(R).real, a, b)
        im, _ = quad(lambda R: f(R).imag, a, b)
        return re + 1j * im

    # M matrix
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val, _ = quad(
                lambda R, i=i, j=j: u_basis(R, i+2, gamma) * H_minus_E_basis(R, j+2, gamma, Energy),
                1e-10, R_max
            )
            M[i, j] = val
            M[j, i] = val

    # M0 vector
    M0 = np.zeros(n, dtype=complex)
    for i in range(n):
        M0[i] = quad_complex(
            lambda R, i=i: u_basis(R, i+2, gamma) * H_minus_E_u0(R, k, Energy)
        )

    # M00 scalar
    M00 = quad_complex(
        lambda R: np.conj(u0(R, k)) * H_minus_E_u0(R, k, Energy)
    )

    return M, M0, M00


if __name__ == "__main__":
    E = 0.5

    t0 = time.perf_counter()
    M, M0, M00 = build_matrices(E)
    dt1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    Mq, M0q, M00q = build_matrices_quad(E)
    dt2 = time.perf_counter() - t0

    print("E =", E)
    print("\ntrap:", dt1, "s")
    print(M[:5,:5])
    print(M0[:5])
    print("M00 =", M00)

    print("\nGQ:", dt2, "s")
    print(Mq[:5,:5])
    print(M0q[:5])
    print("M00 =", M00q)

    print("\ndiffs")
    print("dM  =", np.max(np.abs(M - Mq)))
    print("dM0 =", np.max(np.abs(M0 - M0q)))
    print("dM00=", abs(M00 - M00q))
