import numpy as np

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        k += 1
        mu += 1/k * (x[k-1] - mu)
        mean_values.append(mu)
    return mean_values
