import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial

N_L = 10          
gamma = 1.0       
mu = 1.0          
hbar = 1.0       


def V(R):
    """Replace with actual potential."""
    V0 = 1.0
    a = 0.5
    return V0 / np.cosh(a * R) ** 2



#u_l(R) for l = 2, ..., N_L

def norm_factor(l, gamma):
    """F_l"""
    return np.sqrt((2 * gamma) ** (2 * l - 1) / factorial(2 * l - 2))


def u_basis(R, l, gamma):
    """u_l(R) = F_l R^(l-1) e^(-gamma R)."""
    F = norm_factor(l, gamma)
    return F * R ** (l - 1) * np.exp(-gamma * R)

#cutoff func f(R)
def cutoff(R, R_c=0.5):
    """
    Smooth cutoff: f(R) -> 0 as R -> 0, f(R) -> 1 as R -> inf.
    """
    return 1.0 - np.exp(-R / R_c)

# Incoming wave u_0(R)
def u0(R, k):
    """u_0(R) = f(R) e^{-ikR} / sqrt(v)."""
    v = hbar * k / mu
    return cutoff(R) * np.exp(-1j * k * R) / np.sqrt(v)

# (H - E) acting on a function, via finite differences for the KE and V(R)

def apply_H_minus_E(psi_R, R, E):
    """
    (H - E) psi on a grid.
    H = -hbar^2/(2 mu) d^2/dR^2 + V(R)
    second-order finite differences for d^2/dR^2.
    """
    dR = R[1] - R[0]
    d2_psi = np.zeros_like(psi_R)
    d2_psi[1:-1] = (psi_R[2:] - 2 * psi_R[1:-1] + psi_R[:-2]) / dR ** 2
    d2_psi[0] = d2_psi[1]
    d2_psi[-1] = d2_psi[-2]

    kinetic = -hbar ** 2 / (2 * mu) * d2_psi
    return kinetic + V(R) * psi_R - E * psi_R

# M, M0, M00

def build_matrices(Energy, N_L=N_L, gamma=gamma, R_max=30.0, N_grid=5000):
    """
        M    : (N_L-1) x (N_L-1) real symmetric matrix,  M[i,j] = <u_{i+2}|H-E|u_{j+2}>
        M0   : (N_L-1) complex vector,                   M0[i]  = <u_{i+2}|H-E|u_0>
        M00  : complex scalar,                            M00    = <u_0|H-E|u_0>
    """
    k = np.sqrt(2 * mu * Energy) / hbar
    R = np.linspace(1e-10, R_max, N_grid)
    dR = R[1] - R[0]

    # basis functions on the grid
    n_basis = N_L - 1  
    phi = np.zeros((n_basis, N_grid))
    for i, l in enumerate(range(2, N_L + 1)):
        phi[i] = u_basis(R, l, gamma)

    # u_0 on the grid
    u0_grid = u0(R, k)

    # (H - E)|u_l> and (H - E)|u_0>
    H_phi = np.zeros((n_basis, N_grid))
    for i in range(n_basis):
        H_phi[i] = apply_H_minus_E(phi[i], R, Energy)

    H_u0 = apply_H_minus_E(u0_grid, R, Energy)

    # M[i,j]
    M = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(i, n_basis):
            val = np.trapezoid(phi[i] * H_phi[j], dx=dR)
            M[i, j] = val.real
            M[j, i] = val.real

    # M0[i]
    M0 = np.zeros(n_basis, dtype=complex)
    for i in range(n_basis):
        M0[i] = np.trapezoid(phi[i] * H_u0, dx=dR)

    # M00
    M00 = np.trapezoid(np.conj(u0_grid) * H_u0, dx=dR)

    return M, M0, M00


# Test
if __name__ == "__main__":
    E_test = 0.5
    M, M0, M00 = build_matrices(E_test)

    print(f"Energy = {E_test}")
    print(f"M  shape: {M.shape}  (real symmetric)")
    print(f"M0 shape: {M0.shape}  (complex vector)")
    print(f"M00:      {M00}  (complex scalar)")
    print(f"\nM (first 5x5 block):\n{M[:5, :5]}")
    print(f"\nM0 (first 5 entries):\n{M0[:5]}")
    print(f"\nSymmetry check max|M - M^T| = {np.max(np.abs(M - M.T)):.2e}")