import numpy as np 


def metropolis(prob, n_steps, step_size, n_burn, n_walkers, rng, x0=0.5):
    if n_steps <= 0 or n_walkers <= 0:
        raise ValueError("n_steps and n_walkers must be positive")
    if not 0 <= n_burn <= n_steps:
        raise ValueError("n_burn must satisfy 0<= n_burn <= n_steps")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    x = np.full(n_walkers, x0, dtype=float)
    samples = np.empty((n_steps, n_walkers))
    n_accepted = 0

    for t in range(n_steps):
        delta = rng.uniform(-step_size, step_size, size=x.shape)
        x_new = x + delta 
        ratio = prob(x_new) / prob(x)
        u = rng.uniform(0.0, 1.0, size=x.shape)
        accept = u < ratio 
        x = np.where(accept, x_new, x) #replace where accept = True, otherwise x remains unchanged 
        samples[t] = x 

        if t >= n_burn:
            n_accepted += accept.sum()
    
    return samples[n_burn:], n_accepted / ((n_steps - n_burn)* n_walkers)


def tune_step_size(prob, x0, rng, n_burn, n_walkers=100, target=0.5, tol=0.05, trial_steps=2000, max_iter=50):
    step_size = 1.0

    for _ in range(max_iter):
        samples, acc_rate = metropolis(prob, trial_steps, step_size, n_burn, n_walkers, rng, x0)
        if abs(acc_rate - target) < tol:
            return step_size
        elif acc_rate > target:
            step_size *= 1.2
        else:
            step_size *= 0.8
        
    raise RuntimeError("tol not reached after max iterations")


def sample_gaussian_direct(n_samples, rng):
    samples = rng.normal(loc=0.0, scale=1/np.sqrt(2), size=n_samples)

    return samples 

    