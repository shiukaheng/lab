import time
import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
import meshcat.geometry as g
from inverse_geometry_utils import distanceToObstacle

from config import LEFT_HAND, RIGHT_HAND

class TrajectoryBezier:
    def __init__(self, robot, cube, traj_cp):
        self.robot = robot
        self.cube = cube
        self.q_pos = Bezier(traj_cp, t_max=1)
        self.q_vel = self.q_pos.derivative(1)
        self.q_acc = self.q_pos.derivative(2)

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
        trajectory = self.opt_params_to_trajectory(opt_params)
        cube_trajectory = self.get_cube_path(trajectory)
        b = Bezier(trajectory, t_max=1)
        # Get the trajectory evaluation points
        samples = np.linspace(0, 1, self.evaluation_points)
        cost = 0
        for sample in samples:
            # Get the current frame
            q = b(sample)
            # # Get the current grip transform
            current_grip_transform = self.get_grip_transform(q)
            # Get the difference between the current grip transform and the reference grip transform
            cost += np.linalg.norm(current_grip_transform.homogeneous - self.reference_grip_transform.homogeneous)
            # Obstacle cost
            cost += max(0.05-distanceToObstacle(self.robot, q, computeFrameForwardKinematics=False), 0)
        return cost / self.evaluation_points
    
    def trajectory_to_opt_params(self, trajectory):
        # Flatten the trajectory control points
        return trajectory.reshape(-1)

    def opt_params_to_trajectory(self, opt_params):
        # Unflatten the trajectory control points
        return opt_params.reshape(self.traj_cp_shape)
    
    def opt_callback(self, opt_params):
        print(self.cost(opt_params))
        traj = self.opt_params_to_trajectory(opt_params)
        self.plot_cube_path(traj)
        time.sleep(0.5)
    
    def optimize(self, traj_cp):
        try:
            self.reference_grip_transform = self.get_grip_transform(traj_cp[0])
            self.traj_cp_shape = traj_cp.shape
            opt_params = self.trajectory_to_opt_params(traj_cp)
            opt_params = fmin_bfgs(self.cost, opt_params, callback=self.opt_callback, maxiter=100)
            opt_traj = self.opt_params_to_trajectory(opt_params)
            self.plot_cube_path(opt_traj)
            time.sleep(5)
            self.viz[self.meshcat_path].delete()
            return opt_traj
        except Exception as e:
            # Clear the meshcat path
            self.viz[self.meshcat_path].delete()
            raise e
        
    def get_grip_transform(self, q):
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        oMl = self.robot.data.oMf[self.robot.model.getFrameId(LEFT_HAND)]
        oMr = self.robot.data.oMf[self.robot.model.getFrameId(RIGHT_HAND)]
        return oMl.inverse() * oMr
    
    def get_cube_path(self, traj_cp):
        cube_frames = []
        for f in traj_cp:
            pin.framesForwardKinematics(self.robot.model, self.robot.data, f)
            oMl = self.robot.data.oMf[self.robot.model.getFrameId(LEFT_HAND)]
            oMr = self.robot.data.oMf[self.robot.model.getFrameId(RIGHT_HAND)]
            # Get average of the two translations
            translation = (oMl.translation + oMr.translation) / 2
            cube_frames.append(translation)
        return cube_frames
    
    def plot_cube_path(self, traj_cp, bezier_samples=100):
        cube_path = self.get_cube_path(traj_cp)
        # self.viz[self.meshcat_path].set_object(g.Line(g.PointsGeometry(np.array(cube_path).transpose()), g.MeshBasicMaterial(color=0x0000ff)))
        bezier_path = Bezier(cube_path, t_max=1)
        samples = np.linspace(0, 1, bezier_samples)
        self.viz[self.meshcat_path].set_object(g.Line(g.PointsGeometry(np.array([bezier_path(t) for t in samples]).transpose()), g.MeshBasicMaterial(color=0x0000ff)))