import numpy as np
from scipy.integrate import quad
from math import factorial
import time

mu = 1.0
hbar = 1.0


def V(R):
    return -np.exp(-R)

def norm_factor(l, gamma):
    return np.sqrt((2 * gamma) ** (2 * l - 1) / factorial(2 * l - 2))

def u_basis(R, l, gamma):
    F = norm_factor(l, gamma)
    return F * R ** (l - 1) * np.exp(-gamma * R)

def cutoff(R, gamma):
    return 1.0 - np.exp(-gamma * R)

def u0_func(R, k, gamma):
    """Incoming wave: u0 = f(R) exp(-ikR) / sqrt(v)"""
    v = hbar * k / mu
    return cutoff(R, gamma) * np.exp(-1j * k * R) / np.sqrt(v)

def u1_func(R, k, gamma):
    """Outgoing wave: u1 = u0*"""
    return np.conj(u0_func(R, k, gamma))

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

def d2_u0(R, k, gamma):
    v = hbar * k / mu
    f   =  1.0 - np.exp(-gamma * R)
    fp  =  gamma * np.exp(-gamma * R)
    fpp = -gamma**2 * np.exp(-gamma * R)
    g   = np.exp(-1j * k * R) / np.sqrt(v)
    gp  = -1j * k * g
    gpp = (-1j * k) ** 2 * g
    return fpp * g + 2 * fp * gp + f * gpp

def H_minus_E_u0(R, k, gamma, Energy):
    u = u0_func(R, k, gamma)
    d2u = d2_u0(R, k, gamma)
    return -hbar ** 2 / (2 * mu) * d2u + (V(R) - Energy) * u

def build_matrices(Energy, N_L=10, gamma=1.5, R_max=30.0):
    """
    Gaussian quadrature with analytical derivatives.
        M    : (N_L-1) x (N_L-1) real symmetric matrix
        M0   : (N_L-1) complex vector
        M00  : complex scalar  <u0|H-E|u0>
        M10  : complex scalar  <u1|H-E|u0>
    """
    k = np.sqrt(2 * mu * Energy) / hbar
    n = N_L - 1

    def quad_complex(f, a=1e-10, b=R_max):
        re, _ = quad(lambda R: np.real(f(R)), a, b, limit=200)
        im, _ = quad(lambda R: np.imag(f(R)), a, b, limit=200)
        return re + 1j * im

    # M: <u_l | H-E | u_l'>  (real symmetric)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val, _ = quad(
                lambda R, i=i, j=j: u_basis(R, i+2, gamma) * H_minus_E_basis(R, j+2, gamma, Energy),
                1e-10, R_max, limit=200
            )
            M[i, j] = val
            M[j, i] = val

    # M0: <u_l | H-E | u0>  (complex vector)
    M0 = np.zeros(n, dtype=complex)
    for i in range(n):
        M0[i] = quad_complex(
            lambda R, i=i: u_basis(R, i+2, gamma) * H_minus_E_u0(R, k, gamma, Energy)
        )

    # M00: <u0 | H-E | u0>  (complex scalar)
    # Paper convention: u0 is NOT conjugated in bra
    M00 = quad_complex(
        lambda R: u0_func(R, k, gamma) * H_minus_E_u0(R, k, gamma, Energy)
    )

    # M10: <u1 | H-E | u0> = <u0* | H-E | u0>  (complex scalar)
    # Paper convention: u1 is NOT conjugated in bra, and u1 = u0*
    M10 = quad_complex(
        lambda R: u1_func(R, k, gamma) * H_minus_E_u0(R, k, gamma, Energy)
    )

    return M, M0, M00, M10


if __name__ == "__main__":
    E = 0.5
    gamma = 1.5
    N_L = 2

    t0 = time.perf_counter()
    M, M0, M00, M10 = build_matrices(E, N_L=N_L, gamma=gamma)
    dt = time.perf_counter() - t0

    print("E =", E, " time:", dt, "s")
    print("M:", M)
    print("M0:", M0)
    print("M00:", M00)
    print("M10:", M10)
