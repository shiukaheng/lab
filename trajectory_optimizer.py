import time
import numpy as np
from bezier import Bezier
import pinocchio as pin
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b, fmin_slsqp, minimize
import meshcat.geometry as g
from inverse_geometry_utils import distanceToObstacle, selfCollisionDistance
import meshcat.transformations as tf

from config import LEFT_HAND, RIGHT_HAND
from tools import jointlimitscost

class TrajectoryBezier:
    def __init__(self, robot, cube, traj_cp):
        self.robot = robot
        self.cube = cube
        self.q_pos = Bezier(traj_cp, t_max=1)
        self.q_vel = self.q_pos.derivative(1)
        self.q_acc = self.q_pos.derivative(2)
        self.control_point_paths = []

class TrajectoryOptimizer:
    def __init__(self, robot, cube, viz, evaluation_points=100, max_iters=20):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.evaluation_points = evaluation_points
        self.opt_og_shape = None
        self.reference_grip_transform = []
        self.meshcat_path = "/trajectory_optimizer"
        self.begin = None
        self.end = None
        self.starting_cube_path = None
        self.max_iters = max_iters
        self.iter = 0
        self.best_cost = np.inf
        self.original_cost = np.inf
        self.opt_bounds = None
        self.vel_bezier = None

    def cost(self, opt_params, offset_penalty=True): # Trajectory control points will be in the shape of q * frames (flattened)
        trajectory = self.opt_params_to_trajectory(opt_params)
        cube_trajectory = self.get_cube_path(trajectory)
        b = Bezier(trajectory, t_max=1)
        # vel_b = b.derivative(1)
        # Get the trajectory evaluation points
        samples = np.linspace(0, 1, self.evaluation_points)
        costs = []
        for sample in samples:
            # Get the current frame
            q = b(sample)
            # vq = vel_b(sample) 
            vq = None
            sample_cost = self.calc_sample_cost(q, vq, sample)
            # Add the cost to the list
            costs.append(sample_cost)
        costs = np.array(costs)
        # MSE of costs array
        final_cost = np.mean(costs)
        if offset_penalty:
            # Add a penalty for the distance between the cube path and the starting cube path
            final_cost += self.cube_path_distance(cube_trajectory, self.starting_cube_path) * 5
        return final_cost

    def calc_sample_cost(self, q, vq=None, sample=None):
        sample_cost = 0
        # Get the current grip transform
        sample_cost += self.illegal_grip_penalty(q) ** 2 * 150
        # Obstacle cost
        # sample_cost += self.obstacle_distance_penalty(q) * 50
        # Joint limits cost
        sample_cost += jointlimitscost(self.robot, q) * 1000
        if vq is not None and self.vel_bezier is not None and sample is not None:
            # Velocity deviation cost
            sample_cost += np.linalg.norm(vq - self.vel_bezier(sample)) * 100
        return sample_cost

    def obstacle_distance_penalty(self, q, dist_offset=0.4, self_collision_dist_offset=0.2
                                  ):
        raw_distance = distanceToObstacle(self.robot, q, computeFrameForwardKinematics=False)
        raw_self_collision_distance = selfCollisionDistance(self.robot, q, computeFrameForwardKinematics=False, computeGeometryPlacements=False)
        return max(-raw_distance + dist_offset, -raw_self_collision_distance * 2 + self_collision_dist_offset, 0)

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
        self.iter += 1
        traj = self.opt_params_to_trajectory(opt_params)
        cost = self.cost(opt_params, offset_penalty=False)
        if cost < self.best_cost:
            self.best_cost = cost
        try:
            print(f"✨ Trajectory optimization: {int(self.iter / self.max_iters * 100)}%, improvement: {int((self.original_cost - self.best_cost) / self.original_cost * 100)}%     ", end="\r")
        except:
            pass
        cube_traj = self.get_cube_path(traj)
        self.clear_exploration_points()
        for q, cq in zip(traj, cube_traj):
            self.plot_exploration_point(cq)
        self.plot_cube_path(traj)
    
    def optimize(self, traj_cp):
        try:
            assert len(traj_cp) > 2
            opt_params = self.initialize(traj_cp)
            opt_params = fmin_slsqp(self.cost, opt_params, callback=self.opt_callback, bounds=self.opt_bounds, epsilon=1e-8, iprint=0)
            # opt_params = minimize(self.cost, opt_params, method="dogleg", callback=self.opt_callback, bounds=self.opt_bounds, options={"maxiter": self.max_iters, "disp": True}).x
            print(f"✅ Trajectory optimization: Done! Cost reduced by: {int((self.original_cost - self.best_cost) / self.original_cost * 100)}%            ")
            
            opt_traj = self.opt_params_to_trajectory(opt_params)
            self.plot_cube_path(opt_traj)
            time.sleep(1)
            self.clear_plots()
            return opt_traj
        except Exception as e:
            # Clear the meshcat path
            self.clear_plots()
            raise e

    def initialize(self, traj_cp):
        self.init_trajectory_to_opt_params_converter(traj_cp)
        self.starting_cube_path = self.get_cube_path(traj_cp)
        self.original_cost = self.cost(self.trajectory_to_opt_params(traj_cp), offset_penalty=False)  
        self.best_cost = self.original_cost
        opt_params = self.trajectory_to_opt_params(traj_cp)
        self.plot_cube_path(traj_cp, path="/original_bezier", color=0xcccccc)
        self.vel_bezier = Bezier(traj_cp, t_max=1).derivative(1)
        
        # Calculate bounds for the optimization
        low_q = self.robot.model.lowerPositionLimit
        high_q = self.robot.model.upperPositionLimit
        # Get number of frames
        n_frames = np.array(traj_cp).shape[0]
        low_traj = np.tile(low_q, (n_frames, 1))
        high_traj = np.tile(high_q, (n_frames, 1))
        low_opt_params = self.trajectory_to_opt_params(low_traj)
        high_opt_params = self.trajectory_to_opt_params(high_traj)
        self.opt_bounds = list(zip(low_opt_params, high_opt_params))
        return opt_params
        
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
    
    def plot_cube_path(self, traj_cp, bezier_samples=100, path=None, color=0x0000ff):
        if path is None:
            path = self.meshcat_path
        cube_path = self.get_cube_path(traj_cp)
        bezier_path = Bezier(cube_path, t_max=1)
        samples = np.linspace(0, 1, bezier_samples)
        self.viz[path].set_object(g.Line(g.PointsGeometry(np.array([bezier_path(t) for t in samples]).transpose()), g.MeshBasicMaterial(color=color)))

    def cube_path_distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def plot_exploration_point(self, point):
        path = f"/traj_exploration_point/{np.random.random()}"
        self.viz[path].set_object(g.Sphere(0.01), g.MeshBasicMaterial(color=0xffff00))
        self.viz[path].set_transform(tf.translation_matrix(point))

    def clear_exploration_points(self):
        self.viz[f"/traj_exploration_point"].delete()

    def clear_plots(self):
        self.viz[self.meshcat_path].delete()
        self.viz[f"/original_bezier"].delete()
        self.clear_exploration_points()