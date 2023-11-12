import numpy as np
from kd_tree import KDTree, KDTreeNode
from typing import Optional, List
import time

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
                 starting_point: Optional[np.ndarray] = None,
                 step_size: float = 1.0,
                 neighbor_radius: float = 5.0,
                 lower_bounds: Optional[np.ndarray] = None,
                 upper_bounds: Optional[np.ndarray] = None,
                 collision_samples: int = 10,
                 ):
        super().__init__(dimensions, RTTNode)
        self.goal = None
        self.insert(starting_point)
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.collision_samples = collision_samples

    def sample(self) -> np.ndarray:
        # Lets sample a random point in the search space
        # print("Sampling random point")
        new_point = np.random.uniform(self.lower_bounds, self.upper_bounds)
        node = self.nearest_neighbor(new_point)
        difference = new_point - node.point
        magnitude = np.linalg.norm(difference)
        direction = difference / magnitude
        new_point_max = node.point + direction * min(self.step_size, magnitude)
        collides, new_point = self.check_edge_collision(node.point, new_point_max)
        if new_point is None:
            return self.sample() # If we cant find a point, lets try again
        # Now that we have a new point, lets insert it into the kd-tree
        # print("Inserting new point to kd-tree")
        new_node = self.insert(new_point)
        # print(new_node)
        # Search for neighbors within a certain radius and see if we can find a better parent
        # print("Searching for neighbors")
        neighbors = self.query_spheroid(new_point, self.neighbor_radius)
        # Lets filter away all neighbors that are not reachable from the new point
        # print("Filtering away neighbors that are not reachable")
        neighbors_in_reach = [n for n in neighbors if self.check_edge_collision(n.point, new_point)[0] == False and n != new_node]
        neighbors_in_reach = [(n, self.get_path(n)[1]) for n in neighbors_in_reach]
        # Now, lets find the best parent
        # print("Finding best parent")
        best_cost = np.inf
        best_parent = None
        for neighbor, neighbor_cost in neighbors_in_reach:
            cost = neighbor_cost + np.linalg.norm(new_point - neighbor.point)
            if cost < best_cost:
                best_cost = cost
                best_parent = neighbor
        if best_parent is None:
            raise RuntimeError("No valid parent found, increase neighbor radius. Should implement a better way to handle this. Cant delete node in KDTree, so we have to handle other way")
        # Now that we have a parent, lets update the node
        new_node.parent = best_parent
        # print(new_node, new_node.parent)
        # Now, lets see if we can help any of the neighbors reduce their cost by making them point to the new node
        # Lets first get our own cost to the root
        # print("Reducing cost of neighbors")
        _, current_cost = self.get_path(new_node)
        for neighbor, neighbor_cost in [n for n in neighbors_in_reach if n[0] != best_parent]:
            # Lets see if we can get a better cost by going through the new node
            new_cost = current_cost + np.linalg.norm(new_point - neighbor.point)
            if new_cost < neighbor_cost:
                # We can get a better cost, lets update the neighbor
                neighbor.parent = new_node
                # print(neighbor, neighbor.parent)
        return new_node
    
    def get_path(self, node: RTTNode) -> (List[RTTNode], float):
        # print("Getting path")
        path = []
        cost = 0.0
        while node is not None:
            path.append(node)
            # print(path)
            if node.parent is not None:
                cost += np.linalg.norm(node.point - node.parent.point)
            node = node.parent
            # time.sleep(0.05)
        return path, cost

    def check_point_collision(self, point: np.ndarray) -> bool:
        # print("Checking point collision")
        # This is a dummy collision checker that always returns false
        return False
    
    def check_edge_collision(self, start: np.ndarray, end: np.ndarray) -> (bool, Optional[np.ndarray]):
        # print("Checking edge collision")
        # Sample along the edge and check for collisions using linspace, from start to end, return last non-collision sample
        samples = np.linspace(start, end, self.collision_samples).tolist()
        samples = [None] + samples
        for (i, sample) in enumerate(samples):
            if i == 0:
                continue
            if self.check_point_collision(sample):
                return True, samples[i-1]
        return False, end

