import numpy as np 

def build_hamiltonian(h11, h12, h21, h22):
    H = np.array([[h11, h12],
                  [h21, h22]], dtype = complex)
    return H

def check_hermitian(H):
    return np.allclose(H, H.conj().T) # build into numpy to check if Hermitian


def diagonalize(H):
    '''
    this func returns eigenvalues in an array (energy states)
    P-matrix made of the e-vectors as cols (q-states)
    D-matrix made of the eigenvalues
    '''
    e_vals, e_vec = np.linalg.eig(H) # build into numpy to get e-vals and e-vecs
    P = e_vec # cols are e-vecs
    D = np.diag(e_vals) # diagonal matrix of e-vals
    return e_vals, P, D

# sanity check

def verify_diagonalization(H, P, D):
    '''
    checking if P^{-1} H P = D
    or equivalently H = P D P^{-1}
    '''
    P_inv =  np.linalg.inv(P)
    H_reconstructed = P @ D @ P_inv
    return np.allclose(H, H_reconstructed) # true or false

def display_results(H, e_vals, P, D):
    print("Hamiltonian H:\n", H)
    print("\nEigenvalues (Energy states):\n", e_vals)
    print("\nP-matrix (Eigenvectors as columns):\n", P)
    print("\nD-matrix (Diagonal matrix of eigenvalues):\n", D)
    
# Example usage

'''
Consider the Hamiltonian: [3,1 ; 1,3] for a vague example
eigenvalues: λ = 3 ± 1  →  E_0 = 2, E_1 = 4
eigenvectors (normalized): |0⟩ = (1, -1)/√2,  |1⟩ = (1, 1)/√2
'''

if __name__ == "__main__":
    H = build_hamiltonian(h11=3, h12=1, h21=1,h22=3)

eigenvalues, P, D = diagonalize(H)
verification = verify_diagonalization(H, P, D)
display_results(H, eigenvalues, P, D)
