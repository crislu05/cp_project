from hamiltonian import local_energy
from wavefunction import energy_gaussian_trial_exact, hermite, rho_exact, psi_gaussian_trial 
from stats import block_error
from sampling import metropolis, tune_step_size
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad, cumulative_trapezoid
from scipy.stats import chisquare, kstest




def test_hermite(n_max=4):
    x = np.linspace(-3, 3, 101)
    exact ={
        0: lambda x: np.ones_like(x),
        1: lambda x: 2*x,
        2: lambda x: 4*x**2 - 2,
        3: lambda x: 8*x**3 - 12*x,
        4: lambda x: 16*x**4 - 48*x**2 + 12,
    }
    for n in range(n_max + 1):
        np.testing.assert_allclose(hermite(n,x), exact[n](x), rtol=1e-12, atol=1e-12)

    print(f"hermite passed for n = 0 .. {n_max}")


def histogram_test(samples, n, bins):
    samples = np.ravel(samples)
    counts, edges = np.histogram(samples, bins=bins, range=(-5,5))
    centres = 0.5 * (edges[1:] + edges[:-1]) #centre of each bin 
    dx = edges[1] - edges[0]
    N = samples.size 


    dens = counts / (N * dx)
    err = np.sqrt(counts) / (N * dx)

    fig, (ax,axr) = plt.subplots(2, 1, sharex=True, figsize=(6,5), gridspec_kw={"height_ratios": [3,1]})

    xf = np.linspace(-5, 5, 400)
    ax.plot(xf, rho_exact(xf, n), "k-", label=rf"exact $|\psi_{n}|^2$")
    ax.errorbar(centres, dens, yerr=err, fmt="o", ms=3, label="Metropolis")
    ax.set_ylabel("probability density")
    ax.legend()
    expected = rho_exact(centres, n)
    mask = err > 0
    pull = (dens[mask] - expected[mask]) / err[mask]
    axr.plot(centres[mask], pull, "o", ms=3)
    axr.axhline(0, color="k", lw=0.8)
    axr.set_xlabel(r"$x$")
    axr.set_ylabel("pull")
    fig.tight_layout()
    
    return fig, np.mean(pull**2)


def chi_square_test(samples, n, bins):
    samples = np.ravel(samples)
    observed, edges = np.histogram(samples, bins=bins, range=(-5,5))
    N = samples.size

    probs = []
    for a,b in zip(edges[:-1], edges[1:]):
        integrand = lambda x: rho_exact(x, n)
        value, abs_err = quad(integrand, a, b)
        probs.append(value)
    probs = np.array(probs)
    expected = N * probs 

    keep = expected >= 5
    O, E = observed[keep], expected[keep]
    E = E * O.sum() / E.sum()

    chi2, p = chisquare(O, E)
    dof = keep.sum() - 1
    return chi2, dof, p 

def cdf(x, n):
    x_grid = np.linspace(-8, 8, 4001)
    rho = rho_exact(x_grid, n)
    F = cumulative_trapezoid(rho, x_grid, initial=0)
    F /= F[-1]
    return np.interp(x, x_grid, F) #linear interpolation 

def ks_test(samples, n, thin=20):
    s = np.ravel(np.asarray(samples)[::thin]) #reduce correlation 
    D, p = kstest(s, cdf, args=(n,))
    return D, p

def moment_test(samples, n, block_size):
    x = np.asarray(samples)
    x_series = x.mean(axis=1)
    x2_series = (x**2).mean(axis=1)

    x_mean = x_series.mean()
    x2_mean = x2_series.mean()

    x_err = block_error(x_series, block_size)
    x2_err = block_error(x2_series, block_size)

    pull_x = (x_mean - 0.0) / x_err
    pull_x2 = (x2_mean - (n + 0.5)) / x2_err 

    return {
        "x": (x_mean, x_err, pull_x),
        "x2": (x2_mean, x2_err, pull_x2)
    }

def trial_energy_test(alpha_values, potential, block_size, rng):
    E_est, E_err = [], []

    for a in alpha_values:
        psi = lambda x: psi_gaussian_trial(x, a)
        prob = lambda x: psi(x)**2

        step = tune_step_size(prob, 0.5, rng, n_burn=1000)
        samples, acc = metropolis(prob, n_steps=10000, step_size=step, n_burn=1000, n_walkers=500, rng=rng)

        E1 = local_energy(psi, samples, potential, h=1e-3, order=4)
        E1_series = E1.mean(axis=1)
        E_est.append(E1_series.mean())
        E_err.append(block_error(E1_series, block_size))
    
    fig, ax = plt.subplots()
    a_fine = np.linspace(0.15, 1.6, 300)
    ax.plot(a_fine, energy_gaussian_trial_exact(a_fine), "k-", label=r"exact $\alpha/2 + 1/(8\alpha)$")
    ax.errorbar(alpha_values, E_est, yerr=E_err, fmt="o", label="VMC")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\langle H\rangle$")
    ax.legend()
    return fig 



