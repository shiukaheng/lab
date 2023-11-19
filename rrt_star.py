import numpy as np
from cache_results import cache_results
from kd_tree import KDTree, KDTreeNode
from typing import Optional, List
import time

'''
N-dimensional RRT* implementation
'''

class RRTStarNode(KDTreeNode):
    def __init__(self, 
                 point: np.ndarray, 
                 left: Optional['RRTStarNode'] = None, 
                 right: Optional['RRTStarNode'] = None
                 ):
        super().__init__(point, left, right)
        self.parent = None
        self.is_goal = False
    def print_path(self):
        node = self
        while node is not None:
            print(node.point)
            node = node.parent

class RRTStar(KDTree):
    def __init__(self, 
                 dimensions: int,
                 initial: Optional[np.ndarray] = None,
                 step_size: float = 0.5,
                 neighbor_radius: float = 2.0,
                 lower_bounds: Optional[np.ndarray] = None,
                 upper_bounds: Optional[np.ndarray] = None,
                 collision_samples: int = 10,
                 goal: Optional[np.ndarray] = None,
                 bias: float = 0.05,
                 goal_seeking_radius: Optional[float] = 1.0,
                 node_class = RRTStarNode,
                 ):
        super().__init__(dimensions, node_class)
        self.initial_node = self.insert(initial)
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.collision_samples = collision_samples
        self.goal = goal
        self.bias = bias
        self.goal_node = None
        if goal is not None:
            self.goal_node = node_class(goal) # We don't insert so it doesn't get added to the kd-tree
            self.goal_node.is_goal = True
        self.goal_seeking_radius = goal_seeking_radius

    def sample(self) -> np.ndarray:
        raise Exception("Removed to prevent accidental use")

    # Old strategy: If we are within the goal radius, we see if we are the best path to the goal, and if so we connect to the goal
    # Question: How is the q of goal calculated?

    def get_goal_cost(self):
        raise Exception("Removed to prevent accidental use")
        
    def get_hypothetical_cost(self, node, new_point):
        raise Exception("Removed to prevent accidental use")
    
    def get_node_cost(self, node):
        raise Exception("Removed to prevent accidental use")
    
    def handle_goal(self, new_clamped_point, new_node, current_cost):
        if self.goal is None:
            return
        if self.goal_seeking_radius is not None:
            # Check if we are within the goal direct radius, if so we attempt to connect directly to the goal
            distance_to_goal = self.distance_to_goal(new_node.point)
            goal_direct_reached = distance_to_goal < self.goal_seeking_radius
            if goal_direct_reached:
                # We are within the goal direct radius, lets check if we can connect directly to the goal
                # goal_direct_collision, goal_direct_point = self.check_edge_collision(new_node.point, self.goal)
                result = self.check_edge_collision(new_node.point, self.goal)
                goal_direct_collision = result[0]
                goal_direct_point = result[1]
                if not goal_direct_collision:
                    # We can connect directly to the goal! Does another path already exist?
                    if self.goal_node.parent is None:
                        # No path exists, lets connect directly to the goal
                        self.goal_node.parent = new_node
                        return
                    # Otherwise, lets compare the existing path to the new path
                    existing_path_cost = self.get_path(self.goal_node)[1]
                    new_path_cost = current_cost + np.linalg.norm(new_clamped_point - self.goal)
                    if new_path_cost < existing_path_cost:
                        # We have a better path, lets connect directly to the goal
                        self.goal_node.parent = new_node
                        return

    def distance_to_goal(self, new_clamped_point):
        return np.linalg.norm(new_clamped_point - self.goal)# Update the goal node

    def sample_random_point(self):
        if (self.goal is not None) and (np.random.uniform() < self.bias):
            return self.goal, True
        else:
            new_point = np.random.uniform(self.lower_bounds, self.upper_bounds)
            return new_point, False
    
    def get_path(self, node: RRTStarNode) -> (List[RRTStarNode], float):
        path = []
        cost = 0.0
        while node is not None:
            path.append(node)
            if node.parent is not None:
                cost += np.linalg.norm(node.point - node.parent.point)
            node = node.parent
        return path, cost

    def check_point_collision(self, point: np.ndarray) -> bool:
        # Extend this to check for collisions
        return False
        
    def check_edge_collision(self, start: np.ndarray, end: np.ndarray) -> (bool, Optional[np.ndarray]):
        # Sample along the edge and check for collisions using linspace, from start to end, return last non-collision sample
        samples = np.linspace(start, end, self.collision_samples)
        for (i, sample) in enumerate(samples):
            if i == 0:
                # Theoretically, we should check the start point, but we assume it was checked before
                continue
            if self.check_point_collision(sample):
                if i == 1:
                    return True, None
                else:
                    return True, samples[i-1]
        return False, end

    def goal_found(self):
        # Case where no goal is set. It will be free exploration, so always return false.
        if self.goal_node is None:
            return False
        # Case where goal is set, but no path was found
        if self.goal_node.parent is None:
            return False
        else: # Case where goal is set and path was found
            return True