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
        pin.forwardKinematics(robot.model, robot.data, q)
        # Compute left and right end effector frames
        left_end_effector_placement = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        right_end_effector_placement = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        # Compute cube placement as the average of the left and right end effector translations
        cube_placement = (left_end_effector_placement.translation + right_end_effector_placement.translation) / 2
        cube_path.append(cube_placement)
    return np.array(cube_path)

def trajectory_analysis(q0, q_trajectory):
    robot, cube, viz = setupwithmeshcat()
    # Calculate cube trajectory
    cube_trajectory = to_cube_path(robot, q_trajectory)
    # Calculate the distance the cube has traveled from frame to frame
    cube_distance = np.linalg.norm(np.diff(cube_trajectory, axis=0), axis=1)
    # Calculate the per frame velocity of the cube
    cube_velocity = np.diff(cube_trajectory, axis=0)
    # Calculate the per frame acceleration of the cube
    cube_acceleration = np.diff(cube_velocity, axis=0)
    # Calculate the per frame velocity per joint
    joint_velocity = np.diff(q_trajectory, axis=0)
    # Calculate the per frame acceleration per joint
    joint_acceleration = np.diff(joint_velocity, axis=0)
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
    return cube_distance, cube_velocity, cube_acceleration, joint_velocity, joint_acceleration, obstacle_distances, cube_grasping_residual

def relative_transform(robot, q0):
    pin.forwardKinematics(robot.model, robot.data, q0)
    # Compute left and right end effector frames
    left_end_effector_placement = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    right_end_effector_placement = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
    relative_transform = left_end_effector_placement.inverse() * right_end_effector_placement
    return relative_transform