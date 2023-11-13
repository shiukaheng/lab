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

from rrt_star import *

class RTTStarNodeImpl(RTTNode):
    def __init__(self, 
                 point: np.ndarray, 
                 left: Optional['RTTNode'] = None, 
                 right: Optional['RTTNode'] = None,
                 ):
        super().__init__(point, left, right)
        self.q = None


class RTTStarImpl(RTTStar):
    # Make init pass all arguments to super
    def __init__(self,
                    robot,
                    cube,
                    viz,
                    dimensions: int=3,
                    initial: Optional[np.ndarray] = None,
                    step_size: float = 0.05,
                    neighbor_radius: float = 0.2,
                    lower_bounds: Optional[np.ndarray] = np.array([0.4, -0.4, 0.93]),
                    upper_bounds: Optional[np.ndarray] = np.array([0.5, 0.4, 1.5]),
                    collision_samples: int = 10,
                    goal: Optional[np.ndarray] = None,
                    goal_radius: float = 0.05,
                    bias: float = 0.05,
                    ):
            super().__init__(dimensions, initial, step_size, neighbor_radius, lower_bounds, upper_bounds, collision_samples, goal, goal_radius, bias)
            self.robot = robot
            self.cube = cube
            self.viz = viz

    def sample(self) -> np.ndarray:
        # Lets sample a random point in the search space. If we have a goal, we sample that with a certain probability.
        if (self.goal is not None) and (np.random.uniform() < self.bias):
            # Lets sample the goal
            new_point = self.goal
            # print("Sampling goal:", new_point)
        else:
            new_point = np.random.uniform(self.lower_bounds, self.upper_bounds)
        node = self.nearest_neighbor(new_point)
        difference = new_point - node.point
        magnitude = np.linalg.norm(difference)
        if magnitude == 0:
            new_point = node.point
        else:
            direction = difference / magnitude
            new_point_max = node.point + direction * min(self.step_size, magnitude)
            _, new_point = self.check_edge_collision(node.point, new_point_max) # 

        if new_point is None:
            return self.sample() # If we cant find a point, lets try again
        # Now that we have a new point, lets insert it into the kd-tree
        new_node = self.insert(new_point)
        # Search for neighbors within a certain radius and see if we can find a better parent
        neighbors = self.query_spheroid(new_point, self.neighbor_radius)
        # Lets filter away all neighbors that are not reachable from the new point
        neighbors_in_reach = [n for n in neighbors if self.check_edge_collision(n.point, new_point)[0] == False and n != new_node]
        neighbors_in_reach = [(n, self.get_path(n)[1]) for n in neighbors_in_reach]
        # Now, lets find the best parent
        best_cost = np.inf
        best_parent = None
        for neighbor, neighbor_cost in neighbors_in_reach:
            cost = neighbor_cost + np.linalg.norm(np.array(new_point) - np.array(neighbor.point))
            if cost < best_cost:
                best_cost = cost
                best_parent = neighbor
        if best_parent is None:
            raise RuntimeError("No valid parent found, increase neighbor radius. Should implement a better way to handle this. Cant delete node in KDTree, so we have to handle other way")
        # Now that we have a parent, lets update the node
        new_node.parent = best_parent
        # Now, lets see if we can help any of the neighbors reduce their cost by making them point to the new node
        # Lets first get our own cost to the root
        _, current_cost = self.get_path(new_node)
        for neighbor, neighbor_cost in [n for n in neighbors_in_reach if n[0] != best_parent]:
            # Lets see if we can get a better cost by going through the new node
            new_cost = current_cost + np.linalg.norm(new_point - neighbor.point)
            if new_cost < neighbor_cost:
                # We can get a better cost, lets update the neighbor
                neighbor.parent = new_node
        # Did we reach the goal?
        if self.goal is not None:
            goal_reached = np.linalg.norm(new_point - self.goal) < self.goal_radius
            best_goal = self.goal_node is None or current_cost < self.get_path(self.goal_node)[1]
            if goal_reached and best_goal:
                self.goal_node = new_node
        return new_node
    
    def check_point_collision(self, point: np.ndarray) -> bool:
        neighbour = self.nearest_neighbor(point)
        pass


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal, robot, cube, viz):
    path = RTTStarImpl(
        robot, cube, viz,
        initial=cubeplacementq0.translation,
        goal=cubeplacementqgoal.translation
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
    
    # displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
