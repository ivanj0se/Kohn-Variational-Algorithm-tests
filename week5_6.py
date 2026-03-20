import numpy as np
import matplotlib.pyplot as plt
from week4 import build_matrices


def compute_S(Energy, **kwargs):
    """
    Eqs. (4)-(5):

        B = M00 - M0^T M^{-1} M0
        C = M10 - M0*^T M^{-1} M0
        S = i (B - C^T B*^{-1} C)

    Single-channel: B, C are scalars, so S = i(B - C^2 / B*).
    """
    M, M0, M00, M10 = build_matrices(Energy, **kwargs)

    M_inv = np.linalg.inv(M)

    B = M00 - M0 @ M_inv @ M0
    C = M10 - np.conj(M0) @ M_inv @ M0

    S = 1j * (B - C**2 / np.conj(B))
    return S


if __name__ == "__main__":
    E_test = 0.5
    S_test = compute_S(E_test, N_L=2, gamma=1.5)
    print("Week 5 test")
    print("E =", E_test)
    print("S =", S_test, " |S| =", abs(S_test))

    # V(R) = -exp(-R), gamma = 1.5, mu = 1, N_l = 2
    gamma = 1.5
    N_L = 2

    k_vals = np.linspace(0.01, 1.0, 50)
    E_vals = k_vals**2 / 2.0

    Re_S = np.zeros_like(k_vals)
    Im_S = np.zeros_like(k_vals)

    for i, E in enumerate(E_vals):
        S = compute_S(E, N_L=N_L, gamma=gamma)
        Re_S[i] = S.real
        Im_S[i] = S.imag
        if i % 10 == 0:
            print(f"  k={k_vals[i]:.2f}  E={E:.4f}  S={S:.6f}  |S|={abs(S):.6f}")

    print("\n|S| range:", np.min(np.abs(Re_S + 1j*Im_S)), "to", np.max(np.abs(Re_S + 1j*Im_S)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_vals, Re_S, 'r-', linewidth=2, label='Re(S)')
    ax.plot(k_vals, Im_S, 'b-', linewidth=2, label='Im(S)')
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('S matrix elements', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure3_reproduction.pdf')
    plt.show()
