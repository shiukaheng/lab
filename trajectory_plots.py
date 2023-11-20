import numpy as np
from base_path_cost_evaluator import *
import pinocchio as pin
from config import LEFT_HAND, RIGHT_HAND
from inverse_geometry_utils import distanceToObstacle
from tools import setupwithmeshcat
import numpy as np

def to_cube_path(robot, q_trajectory):
    cube_path = []
    for q in q_trajectory:
        pin.framesForwardKinematics(robot.model, robot.data, q)
        # Compute left and right end effector frames
        left_end_effector_placement = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        right_end_effector_placement = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        # Compute cube placement as the average of the left and right end effector translations
        cube_placement = (left_end_effector_placement.translation + right_end_effector_placement.translation) / 2
        cube_path.append(cube_placement)
    return np.array(cube_path)

def trajectory_analysis(q0, q_trajectory, dt): # dt is time between frames
    robot, cube, viz = setupwithmeshcat()
    # Calculate cube trajectory
    cube_trajectory = to_cube_path(robot, q_trajectory)
    # Calculate the distance the cube has traveled from frame to frame
    cube_distance = np.cumsum(np.linalg.norm(np.diff(cube_trajectory, axis=0), axis=1))
    # Calculate the per frame velocity of the cube
    cube_velocity = np.diff(cube_trajectory, axis=0) / dt
    # Calculate the per frame acceleration of the cube
    cube_acceleration = np.diff(cube_velocity, axis=0) / dt
    # Calculate the per frame velocity per joint
    joint_velocity = np.diff(q_trajectory, axis=0) / dt
    # Calculate the per frame acceleration per joint
    joint_acceleration = np.diff(joint_velocity, axis=0) / dt
    # Calculate distance to obstacle per frame
    obstacle_distances = []
    for q in q_trajectory:
        obstacle_distances.append(distanceToObstacle(robot, q))
    obstacle_distances = np.array(obstacle_distances)
    # Calculate the cube grasping residual per frame
    original = relative_transform(robot, q0)
    cube_grasping_residual = []
    for q in q_trajectory:
        current = relative_transform(robot, q)
        cube_grasping_residual.append(np.linalg.norm(current.homogeneous - original.homogeneous))
    cube_grasping_residual = np.array(cube_grasping_residual)
    # return cube_trajectory, cube_distance, cube_velocity, cube_acceleration, joint_velocity, joint_acceleration, obstacle_distances, cube_grasping_residual
    return {
        'cube_trajectory': cube_trajectory,
        'cube_distance': cube_distance,
        'cube_velocity': cube_velocity,
        'cube_acceleration': cube_acceleration,
        'joint_velocity': joint_velocity,
        'joint_acceleration': joint_acceleration,
        'obstacle_distances': obstacle_distances,
        'cube_grasping_residual': cube_grasping_residual
    }

def relative_transform(robot, q0):
    pin.framesForwardKinematics(robot.model, robot.data, q0)
    # Compute left and right end effector frames
    left_end_effector_placement = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    right_end_effector_placement = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
    relative_transform = left_end_effector_placement.inverse() * right_end_effector_placement
    return relative_transform

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from control_utils import create_linear_trajectory, create_naive_bezier_trajectory, create_naive_bezier_trajectory_b, create_optimized_bezier_trajectory  
from inverse_geometry import computeqgrasppose
from path import computepathwithcubepos

def create_path_comparison(robot, cube, viz=None, total_time=3, ramp_time=0.5, n_bezier_control_points=10, n_bezier_cost_samples=50, evaluation_samples=1000):
      
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepathwithcubepos(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube, viz)
    cube_waypoints, pose_waypoints = zip(*path)

    # Linear trajectory
    pql_eval = create_linear_trajectory(pose_waypoints, cube_waypoints, total_time=total_time, ramp_time=ramp_time, n_samples=evaluation_samples)

    # Create an array of parameters to iterate through
    t = np.linspace(0, total_time, evaluation_samples)

    # Create the naive trajectory
    pqn, _, _ = create_naive_bezier_trajectory(pose_waypoints, cube_waypoints, total_time, ramp_time, n_samples=1000)

    pqn_eval = np.array([pqn(ti) for ti in t])

    pqnd, _, _ = create_naive_bezier_trajectory_b(pose_waypoints,
                                               total_time,
                                               ramp_time,
                                               n_bezier_control_points)
    
    pqnd_eval = np.array([pqnd(ti) for ti in t])

    # Create the optimized trajectory
    pqo, _, _ = create_optimized_bezier_trajectory(robot, cube, viz, pose_waypoints,
                                               total_time,
                                               ramp_time,
                                               n_bezier_control_points,
                                               n_bezier_cost_samples)

    # Evaluate pq at each time step
    pqo_eval = np.array([pqo(ti) for ti in t])

    dt = total_time / evaluation_samples

    return {
        "linear": trajectory_analysis(q0, pql_eval, dt),
        "naive_bezier": trajectory_analysis(q0, pqn_eval, dt),
        "naive_bezier_downsampled": trajectory_analysis(q0, pqnd_eval, dt),
        "optimized_bezier": trajectory_analysis(q0, pqo_eval, dt)
    }

def create_path_comparison_b(pose_waypoints, cube_waypoints, pqn, pq, total_time=3, ramp_time=0.5, evaluation_samples=1000):

    # Linear trajectory
    pql_eval = create_linear_trajectory(pose_waypoints, cube_waypoints, total_time=total_time, ramp_time=ramp_time, n_samples=evaluation_samples)

    # Create an array of parameters to iterate through
    t = np.linspace(0, total_time, evaluation_samples)

    pqn_eval = np.array([pqn(ti) for ti in t])

    # Evaluate pq at each time step
    pq_eval = np.array([pq(ti) for ti in t])

    dt = total_time / evaluation_samples

    q0 = pose_waypoints[0]

    return {
        "linear": trajectory_analysis(q0, pql_eval, dt),
        "naive_bezier": trajectory_analysis(q0, pqn_eval, dt),
        "optimized_bezier": trajectory_analysis(q0, pq_eval, dt)
    }

def normalize(arr, axis=None):
    """
    Normalize a NumPy array to the range [0, 1] along the specified axis.

    Parameters:
    - arr: NumPy array to be normalized.
    - axis: Axis along which to normalize. Default is None (normalize the entire array).

    Returns:
    - Normalized NumPy array.
    """
    if axis is None:
        return (arr - arr.min()) / (arr.max() - arr.min())
    elif axis == 0:
        return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))
    elif axis == 1:
        return (arr - arr.min(axis=1, keepdims=True)) / (arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True))
    else:
        raise ValueError("Invalid axis. Use None for entire array, 0 for columns, or 1 for rows.")