import numpy as np
from base_path_cost_evaluator import *
from inverse_geometry_utils import distanceToObstacle
from path_cost_utils import smoothness


class PathCostEvaluator(BasePathCostEvaluator):

    # Cost on distance of cube path
    @cube_path_cost(weight=1)
    def cube_path_length(self, cube_path):
        return np.linalg.norm(np.diff(cube_path, axis=0), axis=1).sum()
    
    # Cost on distance from obstacles
    @pose_cost(weight=1)
    def obstacle_distance(self, pose):
        return distanceToObstacle(self.robot, pose) ** 2
    
    # Local direction change cost on pose
    @pose_path_cost(weight=1)
    def pose_smoothness(self, path):
        return smoothness(path)
    
    # Local direction change cost on cube
    @cube_path_cost(weight=1)
    def cube_smoothness(self, path):
        return smoothness(path)