import numpy as np 
from derivatives import second_derivative

def potential_sho(x):
    x = np.asarray(x, dtype=float)
    return (1 / 2) * x**2


def local_energy(psi, x, potential, h, order=2):
    x = np.asarray(x, dtype=float)
    wf = psi(x)
    wf_second_deriv = second_derivative(psi, x, h, order)

    return (-1 / 2) * (wf_second_deriv / wf) + potential(x)


def local_energy_exact_eigenstate(x, n):
    x = np.asarray(x, dtype=float)
    return np.full_like(x, n + 0.5)
    