import numpy as np
from path_cost_utils import *
from scipy.interpolate import interp1d

def path_cost(weight):
    def decorator(func):
        func._is_path_cost_function = True
        func._weight = weight
        return func
    return decorator

def cube_path_cost(weight):
    def decorator(func):
        func._is_cube_path_cost_function = True
        func._weight = weight
        return func
    return decorator

def pose_path_cost(weight):
    def decorator(func):
        func._is_pose_path_cost_function = True
        func._weight = weight
        return func
    return decorator

def cube_cost(weight):
    def decorator(func):
        func._is_cube_cost_function = True
        func._weight = weight
        return func
    return decorator

def pose_cost(weight):
    def decorator(func):
        func._is_pose_cost_function = True
        func._weight = weight
        return func
    return decorator

def access_dict_safe(dictionary, key):
    if key in dictionary:
        return dictionary[key]
    else:
        return None

class PathCostMeta(type):
    def __new__(cls, name, bases, dct):
        path_cost_functions = []
        cube_path_cost_functions = []
        pose_path_cost_functions = []
        cube_cost_functions = []
        pose_cost_functions = []
        for attr_name, attr_value in dct.items():
            if getattr(attr_value, '_is_path_cost_function', False):
                path_cost_functions.append((attr_value, getattr(attr_value, '_weight')))
            if getattr(attr_value, '_is_cube_path_cost_function', False):
                cube_path_cost_functions.append((attr_value, getattr(attr_value, '_weight')))
            if getattr(attr_value, '_is_pose_path_cost_function', False):
                pose_path_cost_functions.append((attr_value, getattr(attr_value, '_weight')))
            if getattr(attr_value, '_is_cube_cost_function', False):
                cube_cost_functions.append((attr_value, getattr(attr_value, '_weight')))
            if getattr(attr_value, '_is_pose_cost_function', False):
                pose_cost_functions.append((attr_value, getattr(attr_value, '_weight')))
        dct['path_cost_functions'] = path_cost_functions
        dct['cube_path_cost_functions'] = cube_path_cost_functions
        dct['pose_path_cost_functions'] = pose_path_cost_functions
        dct['cube_cost_functions'] = cube_cost_functions
        dct['pose_cost_functions'] = pose_cost_functions
        return super().__new__(cls, name, bases, dct)

class BasePathCostEvaluator(metaclass=PathCostMeta):
    def __init__(self, robot, cube):
        self.robot = robot
        self.cube = cube

    def _compute_cost(self, path, frame_costs_cache=None, frame_indices=None):
        consumed_path = [tuple(frame) for frame in path]
        if frame_costs_cache is None:
            frame_costs_cache = {}
        total_cost = 0
        for func, weight in self.path_cost_functions:
            total_cost += func(self, consumed_path) * weight
        cube_path = np.array(list(self.extract_cube_path(path)))
        for func, weight in self.cube_path_cost_functions:
            total_cost += func(self, cube_path) * weight
        pose_path = np.array(list(self.extract_pose_path(path)))
        for func, weight in self.pose_path_cost_functions:
            total_cost += func(self, pose_path) * weight
        total_frame_costs = 0
        if frame_indices is None:
            frame_indices = range(len(path))
        for ((cube, pose), index) in zip(path, frame_indices):
            cached = access_dict_safe(frame_costs_cache, index)
            if cached is not None:
                total_frame_costs += cached
            else:
                frame_cost = 0
                for func, weight in self.cube_cost_functions:
                    frame_cost += func(self, cube) * weight
                for func, weight in self.pose_cost_functions:
                    frame_cost += func(self, pose) * weight
                total_frame_costs += frame_cost
                frame_costs_cache[index] = frame_cost
        total_cost += total_frame_costs / len(path)
        return total_cost, frame_costs_cache
    
    def compute_cost(self, path):
        return self._compute_cost(np.array(path, dtype=object))[0]
    
    def compute_local_cost(self, path, window_size=5, stride=1):
        # Get list of indices
        window_centers, local_costs = self.compute_local_costs_non_interpolated(path, window_size, stride)
        interp_func = interp1d(window_centers, local_costs, kind='linear', fill_value='extrapolate')
        return interp_func(np.arange(len(path)))

    def compute_local_costs_non_interpolated(self, path, window_size, stride):
        windows, window_centers = get_windows(window_size, stride, len(path))
        frame_cost_cache = {}
        local_costs = []
        path_numpy = np.array(path, dtype=object)
        for window in windows:
            cost, frame_cost_cache = self._compute_cost(path_numpy[window], frame_cost_cache, window)
            local_costs.append(cost)
        return window_centers,local_costs

    def extract_cube_path(self, path):
        return np.array(path)[:, 0]
    
    def extract_pose_path(self, path):
        return np.array(path)[:, 1]