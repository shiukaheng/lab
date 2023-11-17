from dataclasses import dataclass

import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs

from config import LEFT_HAND, RIGHT_HAND

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
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        vq = self.q_vel(t)
        leftJacobians = pin.getFrameJacobian(self.robot.model, self.robot.data, self.robot.model.getFrameId(LEFT_HAND), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        rightJacobians = pin.getFrameJacobian(self.robot.model, self.robot.data, self.robot.model.getFrameId(RIGHT_HAND), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        vl = leftJacobians @ vq
        vr = rightJacobians @ vq
        return oMl, oMr, vl, vr

class TrajectoryOptimizer:
    def __init__(self, robot, cube, viz, evaluation_points=100):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.evaluation_points = evaluation_points
        self.trajectory_control_points_shape = None # Will be set later

    def cost(self, trajectory_control_points): # Trajectory control points will be in the shape of q * frames (flattened)
        # Now we need to unflatten the trajectory control points into a list of q's
        b = self.get_optimization_terms(trajectory_control_points)
        # Create the samples
        samples = np.linspace(0, 1, self.evaluation_points)
        # Calculate the cost
        cost = 0
        for t in samples:
            oMl, oMr, vl, vr = b.hand_frames(t)
            pq = b.q_pos(t)
            vq = b.q_vel(t)
            aq = b.q_acc(t)
        return 0

    def get_optimization_terms(self, trajectory_control_points):
        trajectory_control_points = trajectory_control_points.reshape(self.trajectory_control_points_shape)
        b = TrajectoryBezier(self.robot, self.cube, trajectory_control_points)
        return b
    
    def optimize(self, trajectory_control_points):
        self.trajectory_control_points_shape = trajectory_control_points.shape
        optimized_trajectory_control_points = fmin_bfgs(self.cost, trajectory_control_points)
        return optimized_trajectory_control_points.reshape(self.trajectory_control_points_shape)
        