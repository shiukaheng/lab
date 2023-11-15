import numpy as np
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
        # Lets sample a random point in the search space. If we have a goal, we sample that with a certain probability.
        new_point, biased = self.sample_random_point()

        node = self.nearest_neighbor(new_point)
        difference = new_point - node.point
        magnitude = np.linalg.norm(difference)
        reached_goal = False
        if magnitude == 0:
            new_clamped_point = node.point
        else:
            direction = difference / magnitude
            new_point_max = node.point + direction * min(self.step_size, magnitude)
            is_unclamped = magnitude <= self.step_size
            has_collision, new_clamped_point = self.check_edge_collision(node.point, new_point_max)
            if new_clamped_point is None:
                return self.sample()
            if (not has_collision) and is_unclamped and biased:
                reached_goal = True

        if reached_goal:
            # See if this is the best path to the goal
            old_cost = self.get_goal_cost()
            # See the new cost
            new_cost = self.get_hypothetical_cost(node)
            if new_cost < old_cost:
                # We have a better path to the goal, lets update it
                self.goal_node.parent = node
                return self.goal_node
        else:
            # Now that we have a new point, lets insert it into the kd-tree
            new_node = self.insert(new_clamped_point)
            # Search for neighbors within a certain radius and see if we can find a better parent
            neighbors = self.query_spheroid(new_clamped_point, self.neighbor_radius)
            # Lets filter away all neighbors that are not reachable from the new point
            neighbors_in_reach = [n for n in neighbors if self.check_edge_collision(n.point, new_clamped_point)[0] == False and n != new_node]
            neighbors_in_reach = [(n, self.get_path(n)[1]) for n in neighbors_in_reach]
            # Now, lets find the best parent
            best_cost = np.inf
            best_parent = None
            for neighbor, neighbor_cost in neighbors_in_reach:
                cost = neighbor_cost + np.linalg.norm(np.array(new_clamped_point) - np.array(neighbor.point))
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
                new_cost = current_cost + np.linalg.norm(new_clamped_point - neighbor.point)
                if new_cost < neighbor_cost:
                    # We can get a better cost, lets update the neighbor
                    neighbor.parent = new_node
            # Did we reach the goal?
            self.handle_goal(new_clamped_point, new_node, current_cost)
            return new_node

    # Old strategy: If we are within the goal radius, we see if we are the best path to the goal, and if so we connect to the goal
    # Question: How is the q of goal calculated?

    def get_goal_cost(self):
        if self.goal_node is None:
            return np.inf
        elif self.goal_node.parent is None:
            return np.inf
        else:
            return self.get_path(self.goal_node)[1]
        
    def get_hypothetical_cost(self, node):
        if self.goal_node is None:
            return np.inf
        else:
            return np.linalg.norm(node.point - self.goal_node.point) + self.get_path(node)[1]

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
    
    def solve(self, max_iterations: int = 5000, post_goal_iterations: int = 1000, verbose=True) -> Optional[List[RRTStarNode]]:
        # Sample until we find a path to the goal
        if verbose:
            print("🦾 Starting RTT*")
        iterations = 0
        while not self.goal_found() and iterations < max_iterations:
            self.sample()
            iterations += 1
            if verbose:
                print(f"🔍 Exploring search space: Iteration {iterations}/{max_iterations}", end="\r", flush=True)
        if not self.goal_found():
            if verbose:
                print("❌ Exploring search space: No path found!", flush=True)
        else:
            print("✅ Exploring search space: Path found!          ", flush=True)
            # Now, lets sample some more to see if we can find a better path
            for i in range(post_goal_iterations):
                self.sample()
                if verbose:
                    print(f"✨ Refining path: Iteration {i}/{post_goal_iterations}", end="\r", flush=True)
            if verbose:
                if post_goal_iterations > 0:
                    print("✅ Refining path: Done!          ", flush=True)
            
        # Now, lets get the path
        path, _ = self.get_path(self.goal_node)
        # Reverse the path list
        path.reverse()
        return path

    def goal_found(self):
        # Case where no goal is set. It will be free exploration, so always return false.
        if self.goal_node is None:
            return False
        # Case where goal is set, but no path was found
        if self.goal_node.parent is None:
            return False
        else: # Case where goal is set and path was found
            return True