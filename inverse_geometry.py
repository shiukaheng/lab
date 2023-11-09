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
from scipy.optimize import fmin_slsqp, fmin_bfgs, fmin
import time
from pinocchio.utils import rotate

def effector_distance_cost(robot, cube):
    # Calculate the distance cost from the effectors to the cube hooks
    oMl = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]  # Assuming 'left_hand' is the correct frame name
    oMr = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]  # Assuming 'right_hand' is the correct frame name
    oMhl = getcubeplacement(cube, LEFT_HOOK)  # Assuming 'left_hook' is the correct frame name
    oMhr = getcubeplacement(cube, RIGHT_HOOK)  # Assuming 'right_hook' is the correct frame name
    dist_l = np.linalg.norm(oMl.translation - oMhl.translation) ** 2 + np.linalg.norm(oMl.rotation - oMhl.rotation) ** 2
    dist_r = np.linalg.norm(oMr.translation - oMhr.translation) ** 2 + np.linalg.norm(oMr.rotation - oMhr.rotation) ** 2
    return dist_l ** 2 + dist_r ** 2

def distanceToObstacle(robot, q):
    '''Return the shortest distance between robot and the obstacle. '''

    geomidobs = robot.collision_model.getGeometryId('obstaclebase_0')
    geomidtable = robot.collision_model.getGeometryId('baseLink_0')

    pairs = [i for i, pair in enumerate(robot.collision_model.collisionPairs) if pair.second == geomidobs or pair.second == geomidtable]

    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)

    dists = [pin.computeDistance(robot.collision_model, robot.collision_data, idx).min_distance for idx in pairs]      
    
    return np.mean(dists)

def weirdPostureCost(robot, q):
    # L2 of the difference between the current posture and the initial posture
    return np.linalg.norm(q - robot.q0) ** 2

def selfCollisionDistance(robot, q):
    '''Return the shortest distance between robot and the obstacle. '''
    geomidobs = robot.collision_model.getGeometryId('obstaclebase_0')
    geomidtable = robot.collision_model.getGeometryId('baseLink_0')
    pairs = [i for i, pair in enumerate(robot.collision_model.collisionPairs) if not (pair.second == geomidobs or pair.second == geomidtable)]
    # print([(robot.collision_model.geometryObjects[pair.first].name, robot.collision_model.geometryObjects[pair.second].name) for pair in robot.collision_model.collisionPairs])
    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
    dists = [pin.computeDistance(robot.collision_model, robot.collision_data, idx).min_distance for idx in pairs]      
    # print(dists)
    return np.mean(dists)

def forcefield(distance, threshold=0.1, multiplier=10):
    return max((-distance + threshold), 0) * multiplier

def success(robot, cube, q):
    # No collisions, no joint limits violated, and the cube is grasped (cost < 0.1)
    return not collision(robot, q) and not jointlimitsviolated(robot, q) and effector_distance_cost(robot, cube) < 0.05

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
        dist = effector_distance_cost(robot, cube)
        self_collision = selfCollisionDistance(robot, q)
        joints = jointlimitscost(robot, q)
        dist_to_obstacle = distanceToObstacle(robot, q)
        weird_posture = weirdPostureCost(robot, q)
        # return dist * 50 + forcefield(dist_to_obstacle, 0.01, 1) + joints + weird_posture * 0.01
        return dist

    def callback(q):
        if viz is not None:
            updatevisuals(viz, robot, cube, q)

    def constraint_ineq(q):
        # We want to avoid going beyond the joint limits
        # First we project the configuration to the joint limits
        joint_cost = -jointlimitscost(robot, q)
        distance_cost = distanceToObstacle(robot, q) - 0.05

        return np.array([joint_cost, distance_cost])
        # return np.array([0.])

    # Now we optimize the cost function
    # res = fmin_slsqp(cost, qcurrent, callback=callback, acc=1e-6, iter=100, f_ieqcons=constraint_ineq, iprint=0)

    # Create zero array called "noise" shape of qcurrent
    noise = np.zeros(qcurrent.shape)
    res = None
    for i in range(10):
        res = fmin_bfgs(cost, qcurrent + noise, callback=callback, disp=False)
        if success(robot, cube, res):
            break
        # Add noise to qcurrent with mean 0 and std 1
        noise = np.random.normal(0, 0.2, qcurrent.shape)
        print(f"Failed to find a solution, trying again ({i+1}/10)")
    
    col, joints, _, _ = evaluate_pose(robot, res, cube, printEval=False)
    return res, not (col or joints)

def generate_cube_pos():
    x = np.random.uniform(0.3, 0.5)
    y = np.random.uniform(-0.5, 0.5)
    z = np.random.uniform(0.9, 1.1)
    return pin.SE3(rotate('z', 0.),np.array([x, y, z]))

def evaluate_pose(robot, q, cube, printEval=True):
    # Check if the robot is in collision or respecting the joint limits
    collision_violated = False
    joint_limits_violated = False
    if collision(robot, q):
        collision_violated = True
    if jointlimitsviolated(robot, q):
        joint_limits_violated = True
    jlc = jointlimitscost(robot, q)
    edc = effector_distance_cost(robot, cube)
    if printEval:
        print(f"Collision: {collision_violated}, Joint Limits: {joint_limits_violated}, Joints Cost: {joints_limits_cost}, Effector Distance Cost: {effector_distance_cost}")
    return collision_violated, joint_limits_violated, jlc, edc

def test_implementation(iters=50, seed=42, interactive=False):
    robot, cube, viz = setupwithmeshcat()
    if not interactive:
        viz = None
    # Seed the random number generator
    np.random.seed(seed)
    # Create initial configuration
    q = robot.q0.copy()
    tests = []
    print(f"Running {iters} tests")
    for i in range(iters):
        cubetarget = generate_cube_pos()
        print(f"Test {i+1}/{iters}: {cubetarget.translation}")
        res, success = computeqgrasppose(robot, q, cube, cubetarget, viz)
        col, joints, jlc, edc = evaluate_pose(robot, res, cube, printEval=False)
        colliding_pairs = get_colliding_pairs(robot, res)
        tests.append((cubetarget, res, col, joints, colliding_pairs, jlc, edc))
        if viz is not None:
            time.sleep(1)
        
    print()
    # Calculate the statistics: percentage of collision, percentage of joint limits violated
    col = sum([test[2] for test in tests])
    joints = sum([test[3] for test in tests])
    print(f"Collision: {col/iters*100}%, Joint Limits: {joints/iters*100}%")
    return tests
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()

    test_implementation(interactive=True)
    
# if __name__ == "__main__":
#     from tools import setupwithmeshcat
#     from setup_meshcat import updatevisuals
#     robot, cube, viz = setupwithmeshcat()
    
#     q = robot.q0.copy()
    
#     q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz) # Pose the robot in a collision free configuration grasping the cube at the initial position
#     qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz) # Pose the robot in a collision free configuration grasping the cube at the target position
    
#     while True:
#         cubetarget = generate_cube_pos()
#         q0,successinit = computeqgrasppose(robot, q, cube, cubetarget, viz)
#         time.sleep(1)