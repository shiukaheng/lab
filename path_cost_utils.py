import numpy as np

def smoothness(array):
    if len(array) < 3:
        return 0
    change = np.diff(array, axis=0)
    unit_vectors_of_change = change / np.linalg.norm(change, axis=1)[:, None]
    dot_products = np.sum(unit_vectors_of_change[1:] * unit_vectors_of_change[:-1], axis=1)
    costs = (dot_products - 1) / -2
    smoothness = np.mean(costs ** 2)
    return smoothness