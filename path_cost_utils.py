import numpy as np

def mean_curvature(array):
    if len(array) < 3:
        return 0
    # First order difference
    derivative = np.diff(array, axis=0)
    # Second order difference
    derivative2 = np.diff(derivative, axis=0)
    # Magnitude of second order difference
    derivative2_mag = np.linalg.norm(derivative2, axis=1)
    # Mean curvature
    curvature = np.mean(derivative2_mag)
    return curvature

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