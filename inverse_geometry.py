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

def effector_distance_cost(robot, cube, inflationRadius=0.005):
    # Calculate the distance cost from the effectors to the cube hooks
    oMl = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]  # Assuming 'left_hand' is the correct frame name
    oMr = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]  # Assuming 'right_hand' is the correct frame name
    oMhl = getcubeplacement(cube, LEFT_HOOK)  # Assuming 'left_hook' is the correct frame name
    oMhr = getcubeplacement(cube, RIGHT_HOOK)  # Assuming 'right_hook' is the correct frame name
    oMcube = getcubeplacement(cube)

    # Calculate the direction vectors from the cube to the hooks
    direction_l = oMhl.translation - oMcube.translation
    direction_r = oMhr.translation - oMcube.translation

    # Normalize the direction vectors
    norm_l = np.linalg.norm(direction_l)
    norm_r = np.linalg.norm(direction_r)
    direction_l = direction_l / norm_l if norm_l > 0 else direction_l
    direction_r = direction_r / norm_r if norm_r > 0 else direction_r

    # Apply the inflation radius to shift the hooks' positions outwards
    oMhl_inflated_translation = oMhl.translation + direction_l * inflationRadius
    oMhr_inflated_translation = oMhr.translation + direction_r * inflationRadius

    # Calculate the squared distances including the inflation
    dist_l = np.linalg.norm(oMl.translation - oMhl_inflated_translation) ** 2 + np.linalg.norm(oMl.rotation - oMhl.rotation) ** 2
    dist_r = np.linalg.norm(oMr.translation - oMhr_inflated_translation) ** 2 + np.linalg.norm(oMr.rotation - oMhr.rotation) ** 2

    # Return the sum of the squared distances
    return dist_l + dist_r

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
    collision_ok = not collision(robot, q)
    joint_limits_ok = not jointlimitsviolated(robot, q)
    effector_distance_ok = effector_distance_cost(robot, cube) < 0.05
    issue = ""
    if not (collision_ok and joint_limits_ok and effector_distance_ok):
        issue = f"Collision: {'✅' if collision_ok else '❌'}, Joint Limits: {'✅' if joint_limits_ok else '❌'}, Effector Distance Cost: {'✅' if effector_distance_ok else '❌'}, Colliding Pairs: {get_colliding_pairs(robot, q)}"

    return collision_ok and joint_limits_ok and effector_distance_ok, issue

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''

    # robot should be of type RobotWrapper
    # qcurrent should be a numpy array of size robot.nq
    # cube should be of type RobotWrapper
    # cubetarget is a SE3 object
    # I suspect the viz is taken optionally to update the visuals in the meshcat using `updatevisuals(viz, robot, cube, q)`

    setcubeplacement(robot, cube, cubetarget)
    pin.updateFramePlacements(cube.model, cube.data)

    def initial_cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q)
        return effector_distance_cost(robot, cube)

    def refinement_cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q) # Update the robot data
        dist = effector_distance_cost(robot, cube)
        self_collision = selfCollisionDistance(robot, q)
        joints = jointlimitscost(robot, q)
        dist_to_obstacle = distanceToObstacle(robot, q)
        weird_posture = weirdPostureCost(robot, q)
        return dist * 50 + forcefield(dist_to_obstacle, 0.01, 1) + joints + weird_posture * 0.01 + forcefield(self_collision, 0.01, 1)

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

    # First we make a relaxed optimization to get close to the solution
    print("💭 Relaxed optimization")
    approx = fmin_bfgs(initial_cost, qcurrent, callback=callback, disp=False)
    # We check if the relaxed optimization is a success
    s, iss = success(robot, cube, approx)
    if s:
        print("✅ Relaxed optimization success")
        return approx, True
    else:
        print("🪄 Relaxed optimization failed, refining solution")

    old_cost = initial_cost(approx)
    refined_draft = approx
    refined_success = False
    for i in range(10):
        print(f"🪄 Refinement {i+1}/10: {iss}")
        noise = np.random.normal(0, 0.5, qcurrent.shape)
        new_refined = fmin_bfgs(refinement_cost, refined_draft + noise, callback=callback, disp=False)
        new_cost = refinement_cost(new_refined)
        s, iss = success(robot, cube, new_refined)
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
        return new_refined, True
    else:
        print("❌ Refinement failed")
        return refined_draft, False

def generate_cube_pos():
    x = np.random.uniform(0.3, 0.5)
    y = np.random.uniform(-0.4, 0.4)
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
        print("=====================================")
        print(f"Test {i+1}/{iters}: {cubetarget.translation}")
        res, success = computeqgrasppose(robot, q, cube, cubetarget, viz)
        print("=====================================")
        print()
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