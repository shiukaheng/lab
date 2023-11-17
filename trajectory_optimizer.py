from dataclasses import dataclass

import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
import time

from config import LEFT_HAND, RIGHT_HAND
from inverse_geometry_utils import distanceToObstacle
import meshcat.geometry as g

def split_array(arr, exclude_cols):
    # Splitting the array into columns that need optimization and those that don't
    need_opt = np.delete(arr, exclude_cols, axis=1)
    no_opt = arr[:, exclude_cols]
    return need_opt, no_opt

def merge_arrays(opt_arr, no_opt_arr, exclude_cols):
    # Initialize an array of the same shape as the original
    merged = np.empty((opt_arr.shape[0], opt_arr.shape[1] + no_opt_arr.shape[1]), dtype=opt_arr.dtype)

    # Insert the non-optimized columns at their original positions
    for col in exclude_cols:
        merged[:, col] = no_opt_arr[:, exclude_cols.index(col)]

    # Insert the optimized columns into the remaining positions
    opt_idx = 0
    for col in range(merged.shape[1]):
        if col not in exclude_cols:
            merged[:, col] = opt_arr[:, opt_idx]
            opt_idx += 1

    return merged
class TrajectoryBezier:

    def __init__(self, robot, cube, trajectory_control_points):
        self.robot = robot
        self.cube = cube
        self.q_pos = Bezier(trajectory_control_points, t_max=1)
        self.q_vel = self.q_pos.derivative(1)
        self.q_acc = self.q_pos.derivative(2)

    def hand_frames(self, t):
        # Apply q to the robot
        q = self.q_pos(t)
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        # Retrieve oMl and oMr
        oMl = self.robot.data.oMf[self.robot.model.getFrameId(LEFT_HAND)]
        oMr = self.robot.data.oMf[self.robot.model.getFrameId(RIGHT_HAND)]
        # Calculate spatial velocities
        # pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        # vq = self.q_vel(t)
        # leftJacobians = pin.getFrameJacobian(self.robot.model, self.robot.data, self.robot.model.getFrameId(LEFT_HAND), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # rightJacobians = pin.getFrameJacobian(self.robot.model, self.robot.data, self.robot.model.getFrameId(RIGHT_HAND), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # vl = leftJacobians @ vq
        # vr = rightJacobians @ vq
        return oMl, oMr

class TrajectoryOptimizer:
    def __init__(self, robot, cube, viz, evaluation_points=100):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.evaluation_points = evaluation_points
        self.trajectory_control_points_shape = None
        self.reference_grip_transform = []
        self.meshcat_path = "/trajectory_optimizer"

    def cost(self, trajectory_control_points): # Trajectory control points will be in the shape of q * frames (flattened)
        # Now we need to unflatten the trajectory control points into a list of q's
        b = self.get_optimization_terms(trajectory_control_points)
        # Create the samples
        samples = np.linspace(0, 1, self.evaluation_points)
        # Calculate the cost
        cost = 0
        for t in samples:
            pq = b.q_pos(t)
            
            # We want to penalize any deviation from the original grip transform
            # transform = self.get_grip_transform(pq)
            # cost += np.linalg.norm(transform.homogeneous - self.reference_grip_transform.homogeneous) * 100

            # We also want to penalize being too close to the obstacle
            dist = distanceToObstacle(self.robot, pq, computeFrameForwardKinematics=False)
            cost += max(0, 0.5 - dist) * 10
        cost /= self.evaluation_points
        return cost
    
    def opt_callback(self, qarray):
        print(self.cost(qarray))
        cube_path = self.get_cube_path(qarray)
        self.viz[self.meshcat_path].set_object(g.Line(g.PointsGeometry(np.array(cube_path).transpose()), g.MeshBasicMaterial(color=0x0000ff)))
        time.sleep(0.1)

    def get_optimization_terms(self, trajectory_control_points):
        trajectory_control_points = self.optimization_form_to_full(trajectory_control_points)
        b = TrajectoryBezier(self.robot, self.cube, trajectory_control_points)
        return b

    def optimization_form_to_full(self, trajectory_control_points):
        trajectory_control_points = trajectory_control_points.reshape(self.trajectory_control_points_shape)
        # Insert dummy values for the non-optimized columns (0)
        dummy = np.zeros((trajectory_control_points.shape[0], 2))
        trajectory_control_points = merge_arrays(trajectory_control_points, dummy, [1, 2])
        return trajectory_control_points
    
    def optimize(self, trajectory_control_points):
        try:
            self.num_control_points = trajectory_control_points.shape[0]
            self.reference_grip_transform = self.get_grip_transform(trajectory_control_points[0])
            to_optimize, unoptimized = split_array(trajectory_control_points, [1, 2])
            self.trajectory_control_points_shape = to_optimize.shape
            optimized_trajectory_control_points = fmin_bfgs(self.cost, to_optimize, callback=self.opt_callback)
            # optimized_trajectory_control_points, _, _ = fmin_l_bfgs_b(self.cost, to_optimize, callback=self.opt_callback, epsilon=0.01, approx_grad=True)
            optimized = optimized_trajectory_control_points.reshape(self.trajectory_control_points_shape)
            return merge_arrays(optimized, unoptimized, [1, 2])
        except Exception as e:
            # Clear the meshcat path
            self.viz[self.meshcat_path].delete()
            raise e
        
    def get_grip_transform(self, q):
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        oMl = self.robot.data.oMf[self.robot.model.getFrameId(LEFT_HAND)]
        oMr = self.robot.data.oMf[self.robot.model.getFrameId(RIGHT_HAND)]
        return oMl.inverse() * oMr
    
    def get_cube_path(self, trajectory_control_points):
        frames = self.optimization_form_to_full(trajectory_control_points)
        cube_frames = []
        for f in frames:
            pin.framesForwardKinematics(self.robot.model, self.robot.data, f)
            oMl = self.robot.data.oMf[self.robot.model.getFrameId(LEFT_HAND)]
            oMr = self.robot.data.oMf[self.robot.model.getFrameId(RIGHT_HAND)]
            # Get average of the two translations
            translation = (oMl.translation + oMr.translation) / 2
            cube_frames.append(translation)
        return cube_frames