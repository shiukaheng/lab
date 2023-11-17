import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

class TrajectoryOptimizer:
    def __init__(self, robot, cube, viz, evaluation_points=100):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.evaluation_points = evaluation_points
        self.traj_cp_shape = None
        self.reference_grip_transform = []
        self.meshcat_path = "/trajectory_optimizer"

    def cost(self, opt_params): # Trajectory control points will be in the shape of q * frames (flattened)
        return 0
    
    def trajectory_to_opt_params(self, trajectory):
        # Flatten the trajectory control points
        return trajectory.reshape(-1)

    def opt_params_to_trajectory(self, opt_params):
        # Unflatten the trajectory control points
        return opt_params.reshape(self.traj_cp_shape)
    
    def opt_callback(self, opt_params):
        pass
    
    def optimize(self, traj_cp):
        try:
            self.reference_grip_transform = self.get_grip_transform(traj_cp[0])
            self.traj_cp_shape = traj_cp.shape
            optimized_traj_cp = fmin_bfgs(self.cost, traj_cp, callback=self.opt_callback)
            # optimized_traj_cp, _, _ = fmin_l_bfgs_b(self.cost, to_optimize, callback=self.opt_callback, epsilon=0.01, approx_grad=True)
            optimized = optimized_traj_cp.reshape(self.traj_cp_shape)
            return merge_arrays(optimized, unoptimized, [1, 2])
        except Exception as e:
            # Clear the meshcat path
            self.viz[self.meshcat_path].delete()
            raise e