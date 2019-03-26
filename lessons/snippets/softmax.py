import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L) # [e z1, e z2,... e zN]
    sumExpL = sum(expL) # e z1 + e z2 + ... e zN

    result = []
    for i in expL: # for every class append e zi/ sum of e z1...e zn
        result.append(i/sumExpL)

    return result
