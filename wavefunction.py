import numpy as np
import math

def hermite(n, x):
    """
    Calculates the nth Hermite polynomial for a given array or scalar of x values.
    """
    x= np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    else:
        return 2*x*hermite(n-1, x) - 2*(n-1)*hermite(n-2, x)

def psi_n(x, n):
    """
    Unnormalized wavefunction for the nth energy level of a quantum harmonic oscillator.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-x**2/2) * hermite(n, x)

def rho_exact(x, n):
    """
    Normalised probability density
    """
    x = np.asarray(x, dtype=float)
    norm_constant = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    return norm_constant**2 * psi_n(x, n)**2

def psi_gaussian_trial(x, alpha):
    """
    Non-eigenstate trial function exp(-alpha x ^2)
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-alpha * x**2)

def energy_gaussian_trial_exact(alpha):
    """
    Exact energy of the non-eigenstate trial function exp(-alpha x ^2)
    """
    return alpha / 2 + 1 / (8 * alpha)





