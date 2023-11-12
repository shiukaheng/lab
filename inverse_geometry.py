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

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''

    setcubeplacement(robot, cube, cubetarget)
    pin.updateFramePlacements(cube.model, cube.data)

    def initial_cost(q):
        q = to_full(q)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        return effector_distance_cost(robot, cube)

    def refinement_cost(q):
        q = to_full(q)
        pin.framesForwardKinematics(robot.model, robot.data, q) # Update the robot data
        dist = effector_distance_cost(robot, cube)
        self_collision = selfCollisionDistance(robot, q)
        joints = jointlimitscost(robot, q)
        # joints = 0
        dist_to_obstacle = distanceToObstacle(robot, q)
        return dist * 20 + forcefield(dist_to_obstacle, 0.2, 50) + joints - self_collision * 0.1

    def callback(q):
        q = to_full(q)
        if viz is not None:
            updatevisuals(viz, robot, cube, q)

    # First we make a relaxed optimization to get close to the solution
    # qcurrent = qinitial
    qcurrent = to_compact(qcurrent)

    print("💭 Relaxed optimization")
    approx = fmin_bfgs(initial_cost, qcurrent, callback=callback, disp=False)
    # We check if the relaxed optimization is a success
    s, iss = success(robot, cube, to_full(approx))
    if s:
        print("✅ Relaxed optimization success")
        return to_full(approx), True
    else:
        print("🪄 Relaxed optimization failed, performing full optimization")

    print("💭 Full optimization")
    approx = fmin_bfgs(refinement_cost, qcurrent, callback=callback, disp=False)
    s, iss = success(robot, cube, to_full(approx))
    if s:
        print("✅ Full optimization success")
        return to_full(approx), True
    else:
        print("🪄 Full optimization failed, attempting stochastic refinement")


    # TODO: Optimize from initial relaxed solution

    old_cost = initial_cost(approx)
    refined_draft = approx
    refined_success = False
    for i in range(10):
        print(f"🪄 Refinement {i+1}/10: {iss}")
        noise = np.random.normal(0, 0.5, qcurrent.shape)
        # for j in range(100):
        #     if not collision(robot, to_full(refined_draft + noise)):
        #         break
        #     noise += np.random.normal(0, 0.05, qcurrent.shape)
        new_refined = fmin_bfgs(refinement_cost, refined_draft + noise, callback=callback, disp=False)
        new_cost = refinement_cost(new_refined)
        s, iss = success(robot, cube, to_full(new_refined))
        if s:
            print(f"🪄 Refinement {i+1}/10: {i}")
            print("✅ Refinement success")
            refined_success = True
            break
        if new_cost < old_cost:
            # print("Refinement failed, but better than before")
            refined_draft = new_refined
            old_cost = new_cost

    if refined_success:
        return to_full(new_refined), True
    else:
        print("❌ Refinement failed")
        return to_full(refined_draft), False

def generate_cube_pos(x=None, y=None, z=None):
    x = x if x is not None else np.random.uniform(0.4, 0.5)
    y = y if y is not None else np.random.uniform(-0.4, 0.4)
    z = z if z is not None else np.random.uniform(0.9, 1.1)
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

def random_tests(robot, cube, viz, iters=50, seed=42, interactive=False):
    """
    Generate random tests and evaluate the results. Random tests are not guaranteed to have solutions but we can
    still evaluate relative performance.
    """

    q = robot.q0.copy()
    if not interactive:
        viz = None

    # Seed random number generator
    np.random.seed(seed)
    
    # Run tests
    test_results = []
    print(f"Running {iters} tests")
    for i in range(iters):
        cubetarget = generate_cube_pos()
        print("=====================================")
        print(f"Test {i+1}/{iters}: {cubetarget.translation}")
        res, success = computeqgrasppose(robot, q, cube, cubetarget, viz)
        print("=====================================")
        print()
        col, joints, jlc, edc = evaluate_pose(robot, res, cube, printEval=False)
        colliding_pairs = get_colliding_pairs(robot, res)
        test_results.append((cubetarget, res, col, joints, colliding_pairs, jlc, edc))
        if viz is not None:
            time.sleep(1)
        
    print()
    # Calculate the statistics: percentage of collision, percentage of joint limits violated
    col = sum([test[2] for test in test_results])
    joints = sum([test[3] for test in test_results])
    print(f"Collision: {col/iters*100}%, Joint Limits: {joints/iters*100}%")
    return test_results

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
    # test_position(robot, cube, viz, q, generate_cube_pos(0.5, 0.15, 0.93), "target pose modified")
    # test_position(robot, cube, viz, q, generate_cube_pos(0.5, 0.11, 0.93), "target pose modified 2")
            
if __name__ == "__main__":
    robot, cube, viz = setupwithmeshcat()
    original_tests(robot, cube, viz, interactive=True)
    # random_tests(robot, cube, viz, interactive=True)