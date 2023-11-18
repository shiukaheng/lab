import numpy as np

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

    def compute(self, path):
        total_cost = 0
        for func, weight in self.path_cost_functions:
            total_cost += func(self, path) * weight
        cube_path = self.extract_cube_path(path)
        for func, weight in self.cube_path_cost_functions:
            total_cost += func(self, cube_path) * weight
        pose_path = self.extract_pose_path(path)
        for func, weight in self.pose_path_cost_functions:
            total_cost += func(self, pose_path) * weight
        frame_costs = 0
        for (cube, pose) in path:
            for func, weight in self.cube_cost_functions:
                frame_costs += func(self, cube) * weight
            for func, weight in self.pose_cost_functions:
                frame_costs += func(self, pose) * weight
        total_cost += frame_costs / len(path)
        return total_cost
    
    @path_cost(weight=1)
    def cube_path_length(self, path):
        cube_positions = self.extract_cube_path(path)
        return np.linalg.norm(np.diff(cube_positions, axis=0), axis=1).sum()

    def extract_cube_path(self, path):
        return np.array(path)[:, 0]
    
    def extract_pose_path(self, path):
        return np.array(path)[:, 1]