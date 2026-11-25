"""Run all Section 2.2 validation tests and energy checks."""

import os
import numpy as np
import matplotlib.pyplot as plt

from wavefunction import psi_n
from hamiltonian import local_energy, potential_sho
from sampling import metropolis, tune_step_size, sample_gaussian_direct
from stats import block_error
from validation import (test_hermite, histogram_test, chi_square_test,
                        ks_test, moment_test, trial_energy_test)

# ---------------- settings ----------------
SEED      = 42
N_MAX     = 4
N_STEPS   = 2000      # small for a dry run; increase to ~20000 for final results
N_BURN    = 200
N_WALKERS = 100       # increase to ~500 for final results
THIN      = 20
H_FD      = 1e-3
ORDER     = 4
FIG_DIR   = "figures"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Hermite unit test
    test_hermite(N_MAX)

    # 2. Eigenstates n = 0..N_MAX
    print(f"\n{'n':>2} {'<H>':>10} {'err':>9} {'acc':>5} "
          f"{'chi2/dof':>9} {'p_chi':>6} {'p_KS':>6} {'<x^2>':>8} {'pull':>6}")

    for n in range(N_MAX + 1):
        psi  = lambda x: psi_n(x, n)
        prob = lambda x: psi(x)**2

        step = tune_step_size(prob, 0.5, rng, n_burn=200)
        samples, acc = metropolis(prob, N_STEPS, step, N_BURN, N_WALKERS, rng)
        block = len(samples) // 20

        # sampler validation
        fig, _ = histogram_test(samples[::THIN], n, bins=60)
        fig.savefig(f"{FIG_DIR}/hist_n{n}.png", bbox_inches="tight")
        plt.close(fig)

        chi2, dof, p_chi = chi_square_test(samples[::THIN], n, bins=40)
        D, p_ks = ks_test(samples, n, thin=THIN)
        mom = moment_test(samples, n, block)

        # energy estimate
        El = local_energy(psi, samples, potential_sho, H_FD, ORDER)
        El_series = El.mean(axis=1)
        E, E_err = El_series.mean(), block_error(El_series, block)

        x2, _, pull_x2 = mom["x2"]
        print(f"{n:>2} {E:>10.6f} {E_err:>9.1e} {acc:>5.2f} "
              f"{chi2/dof:>9.2f} {p_chi:>6.3f} {p_ks:>6.3f} {x2:>8.4f} {pull_x2:>6.2f}")

    # 3. Independent benchmark: exact Gaussian sampler, n = 0
    direct = sample_gaussian_direct(50000, rng)
    D, p = ks_test(direct, 0, thin=1)
    print(f"\nDirect sampler KS (n=0): D = {D:.4f}, p = {p:.3f}")

    # 4. Non-eigenstate trial function
    fig = trial_energy_test(np.linspace(0.2, 1.5, 12), potential_sho,
                            block_size=(10000 - 1000) // 20, rng=rng)
    fig.savefig(f"{FIG_DIR}/trial_energy.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()