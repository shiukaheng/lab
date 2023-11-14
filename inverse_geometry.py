#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated, jointlimitscost, distanceToObstacle, get_colliding_pairs
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from tools import setupwithmeshcat
from setup_meshcat import updatevisuals
from tools import setcubeplacement

# Import BFGS
from inverse_geometry_utils import *
from scipy.optimize import fmin_bfgs
import time
from pinocchio.utils import rotate

qinitial = np.array([0, # Base
                         0, 0, # Head
                        0, 0, 0, 0, -1.57, -1.57, # Left arm
                        0, 0, 0, 0, -1.57, 1.57,]) # Right arm

'''
Ideas:
- Modified such that initial pose always is not in collision, because collision metric is not differentiable, and min distance is not differentiable once you are in collision
- Used a multi-tiered approach with early termination. It starts with a relaxed optimization only account for grasping positions, then a full optimization accounting for collisions and joint limits, and finally a stochastic refinement to try to get out of local minima
- Reduced search space by setting head joints to constant
'''

def remove_waist(q): # Remove first joint
    return q[1:]

def add_waist(q, waist=0): # Add first joint
    return np.insert(q, 0, waist)

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''

    setcubeplacement(robot, cube, cubetarget)
    pin.updateFramePlacements(cube.model, cube.data)

    def simple_cost(q):
        q = to_full(q)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        return effector_distance_cost(robot, cube)

    def callback(q):
        q = to_full(q)
        if viz is not None:
            updatevisuals(viz, robot, cube, q)
            time.sleep(0.05)

    def inequality_constraint(q):
        # To satisfy the inequality constraint, we need to return a value >= 0
        distance_to_obstacle = distanceToObstacle(robot, to_full(q))
        return np.array([distance_to_obstacle+0.05])
    
    approx = fmin_bfgs(simple_cost, to_compact(qcurrent), disp=False, epsilon=EPSILON)
    s, iss = success(robot, cube, to_full(approx))
    if s:
        return to_full(approx), True

    approx = fmin_slsqp(simple_cost, approx, f_ieqcons=inequality_constraint, disp=False, bounds=list(zip(to_compact(robot.model.lowerPositionLimit), to_compact(robot.model.upperPositionLimit))), epsilon=EPSILON)
    s, iss = success(robot, cube, to_full(approx))
    if s:
        return to_full(approx), True
    
    return to_full(approx), False

def test_position(robot, cube, viz, q0, pose, name=None):
    print("=====================================")
    if name is not None:
        print(f"Testing {name}")
    else:
        print("Testing pose")
    q, success = computeqgrasppose(robot, q0, cube, pose, viz)
    print("=====================================")
    print()
    return q, success

def original_tests(robot, cube, viz, interactive=False):
    q = robot.q0.copy()
    test_position(robot, cube, viz, q, CUBE_PLACEMENT, "initial pose")
    test_position(robot, cube, viz, q, CUBE_PLACEMENT_TARGET, "target pose")
            
if __name__ == "__main__":
    robot, cube, viz = setupwithmeshcat()
    original_tests(robot, cube, viz, interactive=True)