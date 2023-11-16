#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
from bezier import Bezier

import numpy as np
import scipy.interpolate
from config import LEFT_HAND, RIGHT_HAND

from pid_controller import PIDController
from setup_pybullet import Simulation
from tools import getcubeplacement, setupwithmeshcat
import pinocchio as pin
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 5000.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim: Simulation, robot, trajs, tcurrent, cube):
    pos_traj, vel_traj, acc_traj = trajs

    qr, vqr, aqr = pos_traj(tcurrent), vel_traj(tcurrent), acc_traj(tcurrent)
    q, vq = sim.getpybulletstate()
    M = pin.crba(robot.model, robot.data, q)
    h = pin.nle(robot.model, robot.data, q, vq)

    # Update the robot's geometry placements based on the current configuration q
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    left_end_effector_jacobian = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId('LARM_EFF'))
    right_end_effector_jacobian = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId('RARM_EFF'))
    
    gripping_force = 150

    left_force_6d = np.array([0, -1, 0, 0, 0, -0.03])
    right_force_6d = np.array([0, -1, 0, 0, 0, 0.03])

    # Probably we are having the wrong frame. The force vector should be in the end effector frame, not the world frame.
    gripping_tau = left_end_effector_jacobian.T @ left_force_6d * gripping_force + right_end_effector_jacobian.T @ right_force_6d * gripping_force

    aqd = aqr - Kp * (q - qr) - Kv * (vq - vqr)
    tau = M @ aqd + h + gripping_tau
    sim.step(tau)


def create_linear_velocity_profile(total_duration, ramp_time, n_samples):
    # Time for each sample
    dt = total_duration / n_samples
    # Create time array
    time = np.linspace(0, total_duration, n_samples)
    # Ramp up profile
    ramp_up = np.minimum(time / ramp_time, 1)
    # Ramp down profile
    ramp_down = np.minimum((total_duration - time) / ramp_time, 1)
    # Constant velocity profile
    constant_velocity = np.ones(n_samples)
    # Combine the three profiles
    velocity_profile = np.minimum(np.minimum(ramp_up, ramp_down), constant_velocity)

    return velocity_profile

def resample_path(waypoints, cube_waypoints, velocity_profile, total_time, sampling_rate):
    if len(waypoints) != len(cube_waypoints):
        raise ValueError("waypoints and cube_waypoints must be of identical length")

    # Calculate cumulative distance along cube_waypoints
    distances = np.cumsum(np.sqrt(np.sum(np.diff(cube_waypoints, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Insert 0 at the start

    # Normalize distances to [0, 1]
    normalized_distances = distances / distances[-1]

    # Normalize the velocity profile
    t_norm = np.linspace(0, 1, len(velocity_profile))
    t_total = np.linspace(0, total_time, int(total_time * sampling_rate))

    # Interpolate and integrate velocity profile to get position
    velocity_interp = scipy.interpolate.interp1d(t_norm, velocity_profile, kind='linear', fill_value="extrapolate")
    position_profile = np.cumsum(velocity_interp(np.linspace(0, 1, len(t_total))) / sampling_rate)

    # Normalize the position profile to [0, 1]
    position_profile = position_profile / position_profile[-1]

    # Interpolate joint space waypoints according to the normalized position profile
    joint_trajectory = []
    for dim in range(np.array(waypoints).shape[1]):
        waypoint_interp = scipy.interpolate.interp1d(normalized_distances, np.array(waypoints)[:, dim], kind='linear')
        joint_trajectory.append(waypoint_interp(position_profile))

    return np.array(joint_trajectory).T

def create_bezier_trajectory(trajectory, t_max):
    # Create Bezier curve from trajectory points
    bezier_curve = Bezier(trajectory, t_max=t_max)

    # Compute first and second derivatives (velocity and acceleration)
    velocity_curve = bezier_curve.derivative(1)
    acceleration_curve = bezier_curve.derivative(2)

    return bezier_curve, velocity_curve, acceleration_curve

def create_trajectory(waypoints, cube_waypoints, total_time, ramp_time, sampling_rate):
    # Create velocity profile
    velocity_profile = create_linear_velocity_profile(total_time, ramp_time, n_samples=1000)

    # Resample path according to velocity profile
    pose_trajectory = resample_path(waypoints, cube_waypoints, velocity_profile, 1, 1000) # Forcing it to have 1000 samples, because higher samples cause numerical issues with Bezier curve
    # cube_trajectory = resample_path(cube_waypoints, cube_waypoints, velocity_profile, 1, 1000) # Forcing it to have 1000 samples, because higher samples cause numerical issues with Bezier curve
    # Create Bezier curve from trajectory points
    pose_bezier_curve, pose_velocity_curve, pose_acceleration_curve = create_bezier_trajectory(pose_trajectory, t_max=total_time) # Relying on Bezier t_max to stretch the trajectory to the desired duration
    return pose_bezier_curve, pose_velocity_curve, pose_acceleration_curve

if __name__ == "__main__":
        
    from tools import setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    # robot, sim, cube, viz = setupwithpybulletandmeshcat()
    robot, cube, viz = setupwithmeshcat()
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepathwithcubepos, displaypath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepathwithcubepos(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube, viz)
    cube_waypoints, pose_waypoints = zip(*path)

    robot, sim, cube, viz = setupwithpybulletandmeshcat()

    tcur = 0.
    total_time = 3.

    # Create a trajectory
    trajs = create_trajectory(pose_waypoints, cube_waypoints, total_time=total_time, ramp_time=0.5, sampling_rate=int(1/DT))

    sim.setqsim(q0)

    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT