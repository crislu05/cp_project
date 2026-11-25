import numpy as np 


def mc_estimate(values):
    return np.mean(values), np.std(values, ddof=1)



def autocorrelation(series, max_lag):
    series = series - np.mean(series)
    c0 = np.mean(series * series)

    if c0 == 0:
        raise ValueError("series is constant, autocorrelation undefined")

    acf = np.empty(max_lag + 1)

    for t in range(max_lag + 1):
        acf[t] = np.mean(series[:len(series) - t] * series[t:])
    
    return acf / c0 

def block_error(series, block_size):
    series = np.asarray(series, dtype=float)
    n_blocks = len(series) // block_size
    if n_blocks < 2:
        raise ValueError("need at least 2 blocks")

    trimmed_series = series[:n_blocks*block_size]
    block_mean = trimmed_series.reshape(n_blocks, block_size).mean(axis=1)

    return np.std(block_mean, ddof=1) / np.sqrt(n_blocks)