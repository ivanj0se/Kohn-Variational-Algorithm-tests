# Since M is real symmetric, only Pauli strings with an even number
# of Y factors contribute (the rest have zero trace against M).
# We keep that filter as an optimization but also compute all 4^n
# coefficients and discard the zeros numerically.

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
from week4 import build_matrices

# ----------------------------------------------------------------
# Decompose M into Pauli strings using Qiskit
# ----------------------------------------------------------------
def pauli_decompose(M, tol=1e-10):
    """
    Returns a SparsePauliOp representing M = sum_l c_l P_l.

    Input
    -----
    M   : 2^n x 2^n numpy array
    tol : drop coefficients with |c_l| < tol

    Output
    ------
    op  : SparsePauliOp with fields .paulis (labels) and .coeffs (c_l)
    """
    op = SparsePauliOp.from_operator(Operator(M))
    # Chop small coefficients (numerical noise from roundoff)
    op = op.simplify(atol=tol)
    return op

# ----------------------------------------------------------------
# Quick test: decompose M from week4, verify reconstruction
# ----------------------------------------------------------------
if __name__ == "__main__":
    # N_L = 9 gives an 8x8 M matrix (3 qubits)
    E = 0.5
    gamma = 1.5
    N_L = 9

    M, M0, M00, M10 = build_matrices(E, N_L=N_L, gamma=gamma)
    dim = M.shape[0]
    n_qubits = int(np.log2(dim))
    print(f"M is {dim} x {dim}, n_qubits = {n_qubits}")
    print(f"Total possible Pauli strings: 4^{n_qubits} = {4**n_qubits}")

    # Decompose
    op = pauli_decompose(M)
    print(f"Nonzero terms: {len(op)}")
    print()

    # Print the decomposition
    print("Pauli decomposition of M:")
    for label, c in zip(op.paulis.to_labels(), op.coeffs):
        print(f"  {label}   c = {c.real:+.6f}  (imag {c.imag:+.2e})")

    # Verify: reconstruct M and compare
    M_rec = op.to_matrix()
    err = np.max(np.abs(M - M_rec))
    print()
    print(f"Reconstruction error: max|M - M_rec| = {err:.2e}")