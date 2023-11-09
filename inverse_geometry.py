#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated, jointlimitscost
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET


from tools import setcubeplacement

# Import BFGS
from scipy.optimize import fmin_slsqp, fmin_bfgs, fmin
import time
from pinocchio.utils import rotate

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''

    # robot should be of type RobotWrapper
    # qcurrent should be a numpy array of size robot.nq
    # cube should be of type RobotWrapper
    # cubetarget is a SE3 object
    # I suspect the viz is taken optionally to update the visuals in the meshcat using `updatevisuals(viz, robot, cube, q)`

    setcubeplacement(robot, cube, cubetarget)
    pin.updateFramePlacements(cube.model, cube.data)

    def cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q) # Update the robot data
        # Lets get the position of the LEFT_HAND and RIGHT_HAND
        oMl = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        oMr = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        oMhl = getcubeplacement(cube, LEFT_HOOK)
        oMhr = getcubeplacement(cube, RIGHT_HOOK)
        dist = norm(oMl.translation - oMhl.translation) + norm(oMr.translation - oMhr.translation) + norm(oMl.rotation - oMhl.rotation) + norm(oMr.rotation - oMhr.rotation)
        
        joints = jointlimitscost(robot, q)
        cost = dist + joints
        return cost

    def callback(q):
        updatevisuals(viz, robot, cube, q)

    def constraint_ineq(q):
        # We want to avoid going beyond the joint limits
        # First we project the configuration to the joint limits
        qproj = projecttojointlimits(robot, q)
        # Then we compute the difference between the two configurations
        dq = q - qproj
        # We want to avoid collisions
        # We compute the distance to the obstacle

        return dq * 10000

    # Now we optimize the cost function
    # res = fmin_slsqp(cost, qcurrent, callback=callback, acc=1e-6, iter=100, f_ieqcons=constraint_ineq, iprint=0)
    res = fmin_bfgs(cost, qcurrent, callback=callback, disp=False)
    # q = res.x # The result of the optimization
    
    evaluate_pose(robot, res)

    return res, False

def generate_cube_pos():
    # X in [-1, 1]
    # Y in [-1, 1]
    # Z in [0.5, 1]
    x = np.random.uniform(-0.5, 0.5)
    y = np.random.uniform(-0.5, 0.5)
    z = np.random.uniform(0.9, 1.1)
    return pin.SE3(rotate('z', 0.),np.array([x, y, z]))

def evaluate_pose(robot, q):
    # Check if the robot is in collision or respecting the joint limits
    collision_violated = False
    joint_limits_violated = False
    if collision(robot, q):
        collision_violated = True
    if jointlimitsviolated(robot, q):
        joint_limit_violated = True
    print(f"Collision: {collision_violated}, Joint Limits: {joint_limits_violated}, Cost: {jointlimitscost(robot, q)}")
    
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    # q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz) # Pose the robot in a collision free configuration grasping the cube at the initial position
    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz) # Pose the robot in a collision free configuration grasping the cube at the target position
    
    while True:
        cubetarget = generate_cube_pos()
        q0,successinit = computeqgrasppose(robot, q, cube, cubetarget, viz)
        # time.sleep(1)

    updatevisuals(viz, robot, cube, q0)
    
    
    
