import numpy as np
from kd_tree import KDTree, KDTreeNode
from typing import Optional, List
import time

'''
N-dimensional RRT* implementation
'''

class RTTNode(KDTreeNode):
    def __init__(self, 
                 point: np.ndarray, 
                 left: Optional['RTTNode'] = None, 
                 right: Optional['RTTNode'] = None,
                 ):
        super().__init__(point, left, right)
        self.parent = None
    def print_path(self):
        node = self
        while node is not None:
            print(node.point)
            node = node.parent

class RTTStar(KDTree):
    def __init__(self, 
                 dimensions: int,
                 initial: Optional[np.ndarray] = None,
                 step_size: float = 1.0,
                 neighbor_radius: float = 5.0,
                 lower_bounds: Optional[np.ndarray] = None,
                 upper_bounds: Optional[np.ndarray] = None,
                 collision_samples: int = 10,
                 goal: Optional[np.ndarray] = None,
                 goal_radius: float = 0.5,
                 bias: float = 0.05,
                 ):
        super().__init__(dimensions, RTTNode)
        self.insert(initial)
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.collision_samples = collision_samples
        self.goal = goal
        self.goal_radius = goal_radius
        self.bias = bias
        self.goal_node = None

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
    
    def get_path(self, node: RTTNode) -> (List[RTTNode], float):
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
                return True, samples[i-1]
        return False, end
    
    def solve(self, max_iterations: int = 5000, post_goal_iterations: int = 1000, verbose=True) -> Optional[List[RTTNode]]:
        # Sample until we find a path to the goal
        if verbose:
            print("=====================================")
            print("🦾 Starting RTT*")
        iterations = 0
        while self.goal_node is None and iterations < max_iterations:
            self.sample()
            iterations += 1
            if verbose:
                print(f"🔍 Exploring search space: {int((iterations+1)/max_iterations*100)}%", end="\r", flush=True)
        if self.goal_node is None:
            if verbose:
                print("❌ Exploring search space: No path found!", flush=True)
        else:
            print("✅ Exploring search space: Path found!", flush=True)
            # Now, lets sample some more to see if we can find a better path
            for i in range(post_goal_iterations):
                self.sample()
                if verbose:
                    print(f"✨ Refining path: {int((i+1)/post_goal_iterations*100)}%", end="\r", flush=True)
            if verbose:
                print("✅ Refining path: Done!", flush=True)
        print("=====================================")
            
        # Now, lets get the path
        path, _ = self.get_path(self.goal_node)
        return [x.point for x in path]