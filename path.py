#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

from typing import Optional
import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time

from rtt_star_ig import *

#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal, robot, cube, viz):
    path = RTTStarImpl(
        robot, cube, viz,
        initial=cubeplacementq0.translation,
        goal=cubeplacementqgoal.translation,
        q_init=qinit,
    ).solve(post_goal_iterations=0)
    frames = []
    for f in path:
        # print(f.interpolated_frames)
        if len(f.interpolated_frames) > 0:
            frames.extend([x[1] for x in f.interpolated_frames])
        frames.append(f.q)
    return frames


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    # Seed the random number generator
    # np.random.seed(55)
    
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube, viz)
    
    displaypath(robot,path,dt=0.2,viz=viz) #you ll probably want to lower dt
    
