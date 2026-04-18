import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from week4 import build_matrices, V, u_basis, u0_func, u1_func, H_minus_E_basis, H_minus_E_u0

# Implementation with VQLS

'''
prepare four fundamental inputs: matrices P, a gate U, an
ansatz V(θ), and a cost function C.
'''

def ansatz(theta, n_qubits, n_layers):
    """
    Build |x(theta)> = V(theta)|0>.
    Parameter count: n_qubits * (n_layers + 1)
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0

    # First rotation layer
    for q in range(n_qubits):
        qc.ry(theta[idx], q)
        idx += 1

    # Each subsequent layer: CNOT chain then rotations
    for _ in range(n_layers):
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1

    return qc

# ----------------------------------------------------------------
# Prepare |b_k> = U_k |0>
# U_k puts X on each '1' bit of k and I on each '0' bit
# ----------------------------------------------------------------
def prepare_b(k, n_qubits):
    """
    Returns |b_k> as a numpy array of length 2^n_qubits.
    k is the column index of M^{-1} you want to solve for.
    """
    qc = QuantumCircuit(n_qubits)
    bits = format(k, f"0{n_qubits}b")
    for q, bit in enumerate(bits):
        if bit == "1":
            qc.x(q)
    return Statevector.from_instruction(qc).data


# ----------------------------------------------------------------
# Get |x(theta)> as a statevector
# ----------------------------------------------------------------

def get_x(theta, n_qubits, n_layers):
    qc = ansatz(theta, n_qubits, n_layers)
    return Statevector.from_instruction(qc).data

# ----------------------------------------------------------------
# Cost function C(theta) from Eq. (7)
# ----------------------------------------------------------------
def cost(theta, M, b, n_qubits, n_layers):
    """
    Inputs
    ------
    theta    : 1D array of variational parameters
    M        : 2^n x 2^n matrix to invert (from week4 build_matrices)
    b        : 2^n unit vector (right-hand side of M|x> = |b>)
    n_qubits : number of qubits, n = log2(dim M)
    n_layers : ansatz depth

    Output
    ------
    C(theta) : real scalar in [0, 1]
    """
    # |x(theta)> from the ansatz
    x = get_x(theta, n_qubits, n_layers)

    # |Phi> = M|x>
    phi = M @ x

    # Numerator: |<b|Phi>|^2
    numerator = np.abs(np.vdot(b, phi)) ** 2

    # Denominator: <Phi|Phi> = <x|M^dag M|x>
    denominator = np.vdot(phi, phi).real

    return 1.0 - numerator / denominator



# ----------------------------------------------------------------
# Quick test: build M from week4, evaluate cost at random theta
# ----------------------------------------------------------------
if __name__ == "__main__":
    # N_L = 9 gives an 8x8 M matrix (3 qubits)
    E = 0.5
    gamma = 1.5
    N_L = 9

    M, M0, M00, M10 = build_matrices(E, N_L=N_L, gamma=gamma)
    dim = M.shape[0]
    n_qubits = int(np.log2(dim))
    print(f"M is {dim} x {dim}, using {n_qubits} qubits")

    # Solve for column k = 0 of M^{-1}: |b_0> = (1,0,...,0)
    k = 0
    b = prepare_b(k, n_qubits)

    # Ansatz settings
    n_layers = 3
    n_params = n_qubits * (n_layers + 1)

    # Evaluate cost at a random theta
    rng = np.random.default_rng(42)
    theta0 = rng.uniform(0, 2 * np.pi, size=n_params)

    C0 = cost(theta0, M, b, n_qubits, n_layers)
    print(f"Initial theta has {n_params} parameters")
    print(f"C(theta0) = {C0:.6f}")
    