import time
import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b, fmin_slsqp
import meshcat.geometry as g
from inverse_geometry_utils import distanceToObstacle, selfCollisionDistance

from config import LEFT_HAND, RIGHT_HAND
from tools import jointlimitscost

class TrajectoryBezier:
    def __init__(self, robot, cube, traj_cp):
        self.robot = robot
        self.cube = cube
        self.q_pos = Bezier(traj_cp, t_max=1)
        self.q_vel = self.q_pos.derivative(1)
        self.q_acc = self.q_pos.derivative(2)

class TrajectoryOptimizer:
    def __init__(self, robot, cube, viz, evaluation_points=100, iters=15):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.evaluation_points = evaluation_points
        self.opt_og_shape = None
        self.reference_grip_transform = []
        self.begin = None
        self.end = None
        self.starting_cube_path = None
        self.iters = iters
        self.current_iter = 0

    def cost(self, opt_params): # Trajectory control points will be in the shape of q * frames (flattened)
        trajectory = self.opt_params_to_trajectory(opt_params)
        cube_trajectory = self.get_cube_path(trajectory)
        b = Bezier(trajectory, t_max=1)
        # Get the trajectory evaluation points
        samples = np.linspace(0, 1, self.evaluation_points)
        costs = []
        for sample in samples:
            # Get the current frame
            sample_cost = 0
            q = b(sample)
            # Get the current grip transform
            sample_cost += self.illegal_grip_penalty(q) ** 2 * 150
            # Obstacle cost
            sample_cost += self.obstacle_distance_penalty(q) * 70
            # Joint limits cost
            sample_cost += jointlimitscost(self.robot, q) * 1000
            # Add the cost to the list
            costs.append(sample_cost)
        costs = np.array(costs)
        # MSE of costs array
        final_cost = np.mean(costs)
        final_cost += self.cube_path_distance(cube_trajectory, self.starting_cube_path) * 0.5
        return final_cost

    def obstacle_distance_penalty(self, q, dist_offset=0.2, self_collision_dist_offset=0.2
                                  ):
        raw_distance = distanceToObstacle(self.robot, q, computeFrameForwardKinematics=False)
        raw_self_collision_distance = selfCollisionDistance(self.robot, q, computeFrameForwardKinematics=False, computeGeometryPlacements=False)
        return max(-raw_distance + dist_offset, -raw_self_collision_distance + self_collision_dist_offset, 0)

    def illegal_grip_penalty(self, q):
        current_grip_transform = self.calc_grip_transform(q)
        legal_grip_cost = np.linalg.norm(current_grip_transform.homogeneous - self.reference_grip_transform.homogeneous)
        return legal_grip_cost
    
    def init_trajectory_to_opt_params_converter(self, traj_cp):
        self.reference_grip_transform = self.calc_grip_transform(traj_cp[0])
        self.begin = traj_cp[0]
        self.end = traj_cp[-1]
        # Remove the first and last control points
        traj_cp = traj_cp[1:-1]
        # Remove the first and second columns out of the 15
        traj_cp = traj_cp[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        # Save the shape of the trajectory unflattened = points
        self.opt_og_shape = traj_cp.shape
    
    def trajectory_to_opt_params(self, trajectory):
        # Remove the first and last control points
        trajectory = trajectory[1:-1]
        trajectory = trajectory[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        # Flatten the trajectory control points
        return trajectory.reshape(-1)

    def opt_params_to_trajectory(self, opt_params):
        # Unflatten the trajectory control points
        unflattened = opt_params.reshape(self.opt_og_shape)
        # Add back the two columns that were removed
        unflattened = np.insert(unflattened, [1, 1], 0, axis=1)
        # Add back the first and last control points
        return np.concatenate((np.array([self.begin]), unflattened, np.array([self.end])), axis=0)
    
    def opt_callback(self, opt_params):
        # print(self.cost(opt_params))
        self.current_iter += 1
        print(f"✨ Trajectory optimization: Iteration {self.current_iter}/{self.iters}, cost: {self.cost(opt_params)}       ", end="\r")
        traj = self.opt_params_to_trajectory(opt_params)
        self.plot_cube_path(traj)
    
    def optimize(self, traj_cp):
        try:
            assert len(traj_cp) > 2
            self.current_iter = 0
            self.init_trajectory_to_opt_params_converter(traj_cp)
            self.starting_cube_path = self.get_cube_path(traj_cp)
            # Plot original trajectory
            self.plot_cube_path(traj_cp, path_name="original_path", color=0xcccccc)
            opt_params = self.trajectory_to_opt_params(traj_cp)
            # opt_params = fmin_bfgs(self.cost, opt_params, callback=self.opt_callback, maxiter=30)
            print(f"🚀 Trajectory optimization: Initializing...", end="\r")
            opt_params = fmin_slsqp(self.cost, opt_params, iter=self.iters, callback=self.opt_callback)
            print(f"✅ Trajectory optimization: Done! Final cost: {self.cost(opt_params)}          ")
            opt_traj = self.opt_params_to_trajectory(opt_params)
            self.plot_cube_path(opt_traj)
            time.sleep(1)
            self.viz["/trajectory_optimizer"].delete()
            return opt_traj
        except Exception as e:
            # Clear the meshcat path
            self.viz["/trajectory_optimizer"].delete()
            raise e
        
    def calc_grip_transform(self, q):
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
    
    def plot_cube_path(self, traj_cp, bezier_samples=100, path_name="cube_path", color=0x0000ff):
        cube_path = self.get_cube_path(traj_cp)
        # self.viz[self.meshcat_path].set_object(g.Line(g.PointsGeometry(np.array(cube_path).transpose()), g.MeshBasicMaterial(color=0x0000ff)))
        bezier_path = Bezier(cube_path, t_max=1)
        samples = np.linspace(0, 1, bezier_samples)
        self.viz[f"/trajectory_optimizer/{path_name}"].set_object(g.Line(g.PointsGeometry(np.array([bezier_path(t) for t in samples]).transpose()), g.MeshBasicMaterial(color=color)))

    def cube_path_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))