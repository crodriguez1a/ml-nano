import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y:list, P:list) -> list:

    results:list = []
    for i,p in enumerate(P):
        results.append(Y[i] * np.log(p) + (1-Y[i]) * np.log(1-p))

    return -(sum(results))
