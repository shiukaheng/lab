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

def get_window_centers(window_size, stride, length_of_path):
    if length_of_path < window_size:
        raise ValueError("Path length must be greater than window size")
    all_indices = np.arange(length_of_path) # [0, 1, 2, ..., length_of_path - 1]
    legal_indices = all_indices[window_size//2:-window_size//2] # [window_size//2, ..., length_of_path - 1 - window_size//2]
    sampled_indices = legal_indices[::stride] # [window_size//2, window_size//2 + stride, ..., length_of_path - 1 - window_size//2]
    return sampled_indices

def get_windows(window_size, stride, length_of_path):
    sampled_indices = get_window_centers(window_size, stride, length_of_path)
    windows = []
    for index in sampled_indices:
        windows.append(np.arange(index - window_size//2, index + window_size//2 + 1))
    return np.array(windows, dtype=int), sampled_indices

def calc_local_cost(path, window_size, stride, cost_function):
    windows, window_centers = get_windows(window_size, stride, len(path))
    local_costs = []
    for window in windows:
        local_costs.append(cost_function(path[window]))
    interp_func = interp1d(window_centers, local_costs, kind='cubic', fill_value='extrapolate')
    return interp_func(np.arange(len(path)))