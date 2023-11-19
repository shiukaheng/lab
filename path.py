#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

from numpy.linalg import pinv

import time

from rrt_star_ig import *
from cache_results import cache_results

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations

def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal, robot, cube, viz):
    path = RRTStarIG(
        robot, cube, viz,
        initial=cubeplacementq0.translation,
        goal=cubeplacementqgoal.translation,
        q_init=qinit,
        collision_samples=3,
        max_neighbours=10,
        step_size=0.05,
    ).solve(max_iterations=100, post_goal_iterations=100, shortcut_iterations=50)
    return [path[1] for path in path]

# @cache_results
def computepathwithcubepos(qinit,qgoal,cubeplacementq0, cubeplacementqgoal, robot, cube, viz):
    path = RRTStarIG(
        robot, cube, viz,
        initial=cubeplacementq0.translation,
        goal=cubeplacementqgoal.translation,
        q_init=qinit,
        collision_samples=5,
    ).solve(max_iterations=100, post_goal_iterations=0, shortcut_iterations=50)
    return path

def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    # Seed the random number generator
    # np.random.seed(42)
    
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube, viz)
    displaypath(robot,path,dt=0.2,viz=viz) #you ll probably want to lower dt
    
