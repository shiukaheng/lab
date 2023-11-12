#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time

from rrt_star import *
from inverse_geometry_utils import *

class RTTStarImpl(RTTStar):
    # Make init pass all arguments to super
    def __init__(self,
                    robot,
                    cube,
                    viz,
                    dimensions: int=13,
                    initial: Optional[np.ndarray] = None,
                    step_size: float = 0.1,
                    neighbor_radius: float = 0.3,
                    lower_bounds: Optional[np.ndarray] = None,
                    upper_bounds: Optional[np.ndarray] = None,
                    collision_samples: int = 10,
                    goal: Optional[np.ndarray] = None,
                    goal_radius: float = 0.05,
                    bias: float = 0.05,
                    ):
            super().__init__(dimensions, initial, step_size, neighbor_radius, lower_bounds, upper_bounds, collision_samples, goal, goal_radius, bias)
            self.robot = robot
            self.cube = cube
            self.viz = viz
    def check_point_collision(self, point: np.ndarray) -> bool:
        q = to_full(point)
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
        return pin.computeCollisions(robot.collision_model,robot.collision_data,False)
        # Attempt inverse geometry to see if we can get collision free config
        # Optimization: Use the last point as a starting point
        # Optimization 2: Instantly terminate if we are in collision
        # Issue: Given we are incrementally optimizing from the last pose, how do we guarantee we can smoothly transition to the goal pose?


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal, robot, cube, viz):
    path = RTTStarImpl(
        robot, cube, viz,
        initial=to_compact(qinit),
        goal=to_compact(qgoal),
        lower_bounds=to_compact(robot.model.lowerPositionLimit),
        upper_bounds=to_compact(robot.model.upperPositionLimit),
    ).solve()
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube, viz)
    path = [to_full(q) for q in path]
    
    displaypath(robot,path,dt=0.1,viz=viz) #you ll probably want to lower dt
    
