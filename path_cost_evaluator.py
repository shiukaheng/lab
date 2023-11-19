import numpy as np
from base_path_cost_evaluator import *
from inverse_geometry_utils import distanceToObstacle
from path_cost_utils import *


class PathCostEvaluator(BasePathCostEvaluator):

    # Cost on distance of cube path
    @cube_path_cost(weight=5)
    def cube_path_length(self, cube_path):
        return np.linalg.norm(np.diff(cube_path, axis=0), axis=1).sum()
    
    # Cost on distance from obstacles
    # @pose_cost(weight=50)
    # def obstacle_distance(self, pose):
    #     return -distanceToObstacle(self.robot, pose)
    
    # # Curvature penalty
    # @pose_path_cost(weight=1)
    # def pose_smoothness(self, path):
    #     return mean_curvature(path)
    
    # # Local direction change cost on cube
    # @cube_path_cost(weight=100)
    # def cube_smoothness(self, path):
    #     return mean_curvature(path)