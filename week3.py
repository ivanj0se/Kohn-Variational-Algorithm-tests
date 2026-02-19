import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def normalization_f(l, gamma):
    """
    F_l s.t. <u_l|u_l> = 1.
    
    Integral of [R^(l-1) e^(-gamma*R)]^2 from 0 to inf
    = integral of R^(2l-2) e^(-2*gamma*R) dR = (2l-2)! / (2*gamma)^(2l-1) -> Using mathematica for this thing
    
    F_l = sqrt((2*gamma)^(2l-1) / (2l-2)!)
    """
    n = 2 * l - 2
    return np.sqrt((2 * gamma) ** (2 * l - 1) / factorial(n))


def basis_func(R, l, gamma):
    """u_l(R) = F_l * R^(l-1) * exp(-gamma * R)."""
    F_l = normalization_f(l, gamma)
    return F_l * R ** (l - 1) * np.exp(-gamma * R)


# --- Parameters ---
N_L = 10        # (l = 2, 3, ..., N_L), amount of basis func 
gamma = 1.0     # decay constant
R = np.linspace(0, 12, 500)

fig, ax = plt.subplots(figsize=(9, 5))

for l in range(2, N_L + 1):
    u = basis_func(R, l, gamma)
    ax.plot(R, u, label=rf"$u_{{{l}}}(R)$")

ax.set_xlabel(r"$R$", fontsize=13)
ax.set_ylabel(r"$u_l(R)$", fontsize=13)
ax.legend(ncol=2, fontsize=9)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlim(0, 12)
plt.show()