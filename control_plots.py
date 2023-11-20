#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import time
import numpy as np
from setup_pybullet import Simulation
from tools import getcubeplacement, setupwithmeshcat
import pinocchio as pin

from control_utils import *
from trajectory_optimizer import TrajectoryOptimizer
from trajectory_plots import trajectory_analysis
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 6000.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim: Simulation, robot, trajs, tcurrent):
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

if __name__ == "__main__":
        
    from tools import setupwithpybulletandmeshcat, rununtil
    from config import DT

    robot, cube, viz = setupwithmeshcat()
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepathwithcubepos, displaypath

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)

    tcur = 0.
    total_time = 3.
    
    trajs = create_optimized_bezier_trajectory()
    
    # Sample the trajectory
    bezier_samples = np.linspace(0, total_time, 1000)
    q = []
    for t in bezier_samples:
        q.append(trajs[0](t))
    q = np.array(q)

    trajectory_analysis(q0, q)