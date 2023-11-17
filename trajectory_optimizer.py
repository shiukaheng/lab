import time
import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
import meshcat.geometry as g

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
        # trajectory = self.opt_params_to_trajectory(opt_params)
        return np.linalg.norm(opt_params)
    
    def trajectory_to_opt_params(self, trajectory):
        # Flatten the trajectory control points
        return trajectory.reshape(-1)

    def opt_params_to_trajectory(self, opt_params):
        # Unflatten the trajectory control points
        return opt_params.reshape(self.traj_cp_shape)
    
    def opt_callback(self, opt_params):
        print(self.cost(opt_params))
        traj = self.opt_params_to_trajectory(opt_params)
        print(traj.shape)
        self.plot_cube_path(traj)
    
        time.sleep(0.5)
    
    def optimize(self, traj_cp):
        try:
            self.reference_grip_transform = self.get_grip_transform(traj_cp[0])
            self.traj_cp_shape = traj_cp.shape
            optimized_params = fmin_bfgs(self.cost, traj_cp, callback=self.opt_callback)
            return self.opt_params_to_trajectory(optimized_params)
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
        self.viz[self.meshcat_path].set_object(g.Line(g.PointsGeometry(np.array(cube_path).transpose()), g.MeshBasicMaterial(color=0x0000ff)))
        # bezier_path = Bezier(cube_path, t_max=1)
        # samples = np.linspace(0, 1, bezier_samples)
        # self.viz[self.meshcat_path].set_object(g.Line(g.PointsGeometry(np.array([bezier_path(t) for t in samples]).transpose()), g.MeshBasicMaterial(color=0x0000ff)))