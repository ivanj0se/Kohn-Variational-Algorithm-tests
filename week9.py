#   (a) V(theta) : hardware-efficient ansatz, alternating R_y and CNOT layers
#   (b) U        : prepares |b_k> = U|0> using only X and I gates,
#                  with X on each '1' bit and I on each '0' bit of k

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from week4 import build_matrices
from week8 import pauli_decompose

def build_ansatz(n_qubits, n_layers):
    """
    Returns a parameterized QuantumCircuit V(theta) and the
    ParameterVector theta so the circuit can be bound later.

    Parameter count: n_qubits * (n_layers + 1)
      - One initial R_y layer
      - n_layers subsequent (CNOT chain + R_y layer) blocks

    Inputs
    ------
    n_qubits : number of qubits n = log2(dim M)
    n_layers : ansatz depth (number of entangling layers)

    Outputs
    -------
    qc    : parameterized QuantumCircuit (not yet bound)
    theta : ParameterVector of length n_qubits * (n_layers + 1)
    """
    n_params = n_qubits * (n_layers + 1)
    theta = ParameterVector("θ", n_params)
    qc = QuantumCircuit(n_qubits, name="V(θ)")
    idx = 0

    # First R_y layer
    for q in range(n_qubits):
        qc.ry(theta[idx], q)
        idx += 1

    # Each subsequent layer: CNOT chain then R_y rotations
    for _ in range(n_layers):
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1

    return qc, theta

# U: prepares |b_k> = U|0>

def build_state_prep(k, n_qubits):
    """
    Returns a QuantumCircuit that prepares |b_k>.
    k indexes the column of M^{-1} being solved for.

    Convention: qubit 0 is the most significant bit of k.
    (Qiskit's native ordering is little-endian, but the paper
    reads bits left-to-right, so we follow the paper here.)

    Inputs
    ------
    k        : integer column index, 0 <= k < 2^n_qubits
    n_qubits : number of qubits

    Output
    ------
    qc : QuantumCircuit preparing |b_k> from |0...0>
    """
    assert 0 <= k < 2**n_qubits, f"k={k} out of range for {n_qubits} qubits"
    qc = QuantumCircuit(n_qubits, name=f"U_{k}")
    bits = format(k, f"0{n_qubits}b")   # e.g. k=5, n=3 -> '101'
    for q, bit in enumerate(bits):
        if bit == "1":
            qc.x(q)
    return qc

def build_vqls_circuits(k, n_qubits, n_layers):
    """
    Returns the three ingredients needed to assemble the VQLS
    cost evaluation circuits:

      V_circuit : parameterized ansatz V(theta)
      theta     : ParameterVector for binding
      U_circuit : state prep for |b_k>

    The caller combines these with a Hadamard-test ancilla and
    the Pauli decomposition of M (from week8) to evaluate the
    cost function on hardware.
    """
    V_circuit, theta = build_ansatz(n_qubits, n_layers)
    U_circuit = build_state_prep(k, n_qubits)
    return V_circuit, theta, U_circuit

if __name__ == "__main__":
    from qiskit.quantum_info import Statevector

    # Same setup as weeks 7 and 8: 8x8 M matrix, 3 qubits
    E = 0.5
    gamma = 1.5
    N_L = 9
    M, M0, M00, M10 = build_matrices(E, N_L=N_L, gamma=gamma)
    n_qubits = int(np.log2(M.shape[0]))
    n_layers = 3
    k = 0

    # Build V(theta), U
    V_circuit, theta, U_circuit = build_vqls_circuits(k, n_qubits, n_layers)
    n_params = len(theta)

    print(f"n_qubits = {n_qubits}, n_layers = {n_layers}")
    print(f"n_params = {n_params}")
    print()

    # Show the ansatz structure
    print("Ansatz V(θ):")
    print(V_circuit.draw(output="text"))
    print()

    # Show the state prep for k=0
    print(f"State prep U for k={k}:")
    print(U_circuit.draw(output="text"))
    print()

    # Pauli decomposition of M (from week 8)
    M_pauli = pauli_decompose(M)
    print(f"Pauli decomposition of M: {len(M_pauli)} nonzero terms")
    print()

    # Bind random parameters to V(theta) and confirm it produces
    # a valid statevector (sanity check that the circuit is well-formed)
    rng = np.random.default_rng(42)
    theta_vals = rng.uniform(0, 2 * np.pi, size=n_params)
    V_bound = V_circuit.assign_parameters({theta: theta_vals})
    x_state = Statevector.from_instruction(V_bound).data

    print(f"|x(θ)> norm (should be 1):  {np.linalg.norm(x_state):.6f}")
    print(f"|x(θ)> first few amplitudes:")
    for i, amp in enumerate(x_state[:4]):
        print(f"   x_{i} = {amp.real:+.4f} {amp.imag:+.4f}j")