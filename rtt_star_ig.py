
from typing import List, Optional
import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time
from inverse_geometry import computeqgrasppose
from inverse_geometry_utils import generate_cube_pos, to_full

# Goal 1: Check collision using inverse geometry
# Goal 2: Provide better initial guess for inverse geometry
# Goal 3: Output interpolated parts of the path too

from rrt_star import *
from rrt_star import RTTNode
import meshcat.geometry as g

class RTTStarNodeImpl(RTTNode):
    def __init__(self, 
                 point: np.ndarray, 
                 left: Optional['RTTNode'] = None, 
                 right: Optional['RTTNode'] = None,
                 ):
        super().__init__(point, left, right)
        self.q = None
        self.interpolated_frames = []


class RTTStarImpl(RTTStar):
    # Make init pass all arguments to super
    def __init__(self,
                    robot,
                    cube,
                    viz,
                    dimensions: int=3,
                    initial: Optional[np.ndarray] = None,
                    step_size: float = 0.2,
                    neighbor_radius: float = 0.4,
                    lower_bounds: Optional[np.ndarray] = np.array([0.4, -0.4, 0.93]),
                    upper_bounds: Optional[np.ndarray] = np.array([0.5, 0.4, 1.5]),
                    collision_samples: int = 5,
                    goal: Optional[np.ndarray] = None,
                    bias: float = 0.1,
                    goal_seeking_radius: Optional[float] = 1.0,
                    q_init: Optional[np.ndarray] = None,
                    ):
            super().__init__(
                dimensions=dimensions,
                initial=initial,
                step_size=step_size,
                neighbor_radius=neighbor_radius,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                collision_samples=collision_samples,
                goal=goal,
                bias=bias,
                goal_seeking_radius=goal_seeking_radius,
                node_class=RTTStarNodeImpl,
            )
            self.robot = robot
            self.cube = cube
            self.viz = viz
            self.q_init = q_init if q_init is not None else robot.q0.copy()
            self.initial_node.q = self.q_init
            self.meshcat_paths = []

    def plot_segment(self, start: RTTStarNodeImpl, end: RTTStarNodeImpl):
        # Plot a segment between start and end, and add the meshcat path to the segment array
        path = f"/path_{len(self.meshcat_paths)}"
        # print(f"Start: {start.point}, End: {end.point}")
        self.viz[path].set_object(g.Line(g.PointsGeometry(np.array([start.point, end.point]).transpose()), g.MeshBasicMaterial(color=0xff0000)))
        self.meshcat_paths.append(path)

    def clear_paths(self):
        for path in self.meshcat_paths:
            self.viz[path].delete()
        self.meshcat_paths = []

    def sample(self) -> np.ndarray:
        # Lets find the nearest neighbor to target
        # self.print_nn()

        # Sample a new point
        new_point = self.sample_random_point()
        new_point_clamped, new_q = self.clamp_sampled_point(new_point)

        # If we don't have any valid point, lets try again with a new sample
        if new_point_clamped is None:
            return self.sample()
        
        # Now that we have a new point, lets insert it into the kd-tree
        new_node = self.insert_with_solved_q(new_point_clamped, new_q)

        # Search for neighbors within a certain radius and see if we can find a better parent
        neighbors_in_reach = self.find_reachable_neighbours(new_node)

        # Now, lets find the best parent
        best_parent = self.find_best_parent(new_point_clamped, neighbors_in_reach)
        
        # Now that we have a parent, lets update the node
        self.link_nodes(new_node, best_parent)
        
        # Now, lets see if we can help any of the neighbors reduce their cost by making them point to the new node
        # Lets first get our own cost to the root
        current_cost = self.reduce_neighbour_cost(new_node, neighbors_in_reach, best_parent)
        
        # Did we reach the goal?
        self.handle_goal(new_point_clamped, new_node, current_cost)
        return new_node

    def reduce_neighbour_cost(self, new_node, neighbors_in_reach, best_parent):
        _, current_cost = self.get_path(new_node)
        for neighbor, neighbor_cost in [n for n in neighbors_in_reach if n[0] != best_parent]:
            # Lets see if we can get a better cost by going through the new node
            new_cost = current_cost + np.linalg.norm(new_node.point - neighbor.point)
            if new_cost < neighbor_cost:
                # We can get a better cost, lets update the neighbor
                neighbor.parent = new_node
        return current_cost

    def link_nodes(self, child, parent):
        child.parent = parent
        self.plot_segment(parent, child)

    def find_best_parent(self, new_point_clamped, neighbors_in_reach):
        best_cost = np.inf
        best_parent = None
        for neighbor, neighbor_cost in neighbors_in_reach:
            cost = neighbor_cost + np.linalg.norm(np.array(new_point_clamped) - np.array(neighbor.point))
            if cost < best_cost:
                best_cost = cost
                best_parent = neighbor

        if best_parent is None:
            raise RuntimeError("No valid parent found, increase neighbor radius. Should implement a better way to handle this. Cant delete node in KDTree, so we have to handle other way")
        return best_parent

    def find_reachable_neighbours(self, new_node):
        neighbors = self.query_spheroid(new_node.point, self.neighbor_radius)

        # Lets filter away all neighbors that are not reachable from the new point
        neighbors_in_reach = [n for n in neighbors if self.check_edge_collision(n.point, new_node.point, n.q)[0] == False and n != new_node]
        neighbors_in_reach = [(n, self.get_path(n)[1]) for n in neighbors_in_reach]
        return neighbors_in_reach

    def insert_with_solved_q(self, new_point_clamped, new_q):
        new_node = self.insert(new_point_clamped)
        new_node.q = new_q
        return new_node

    def print_nn(self):
        nn = self.nearest_neighbor(self.goal)
        if nn is not None:
            distance = np.linalg.norm(self.goal - nn.point)
            print("Distance to goal:", distance)

    def clamp_sampled_point(self, new_point):
        nearest_node = self.nearest_neighbor(new_point)
        difference = new_point - nearest_node.point
        magnitude = np.linalg.norm(difference)

        # Lets extend to the maximum step size or until we hit an obstacle
        if magnitude == 0: # For cases where we are already at the nearest node
            new_point_clamped = nearest_node.point
            new_q = None
        else:
            direction = difference / magnitude
            new_point_max = nearest_node.point + direction * min(self.step_size, magnitude)
            _, new_point_clamped, new_q = self.check_edge_collision(nearest_node.point, new_point_max)
        return new_point_clamped,new_q
    
    def check_point_collision(self, point: np.ndarray, start_q: Optional[np.ndarray]=None) -> bool:
        if start_q is None:
            start_q = self.q_init
        q, success = computeqgrasppose(self.robot, start_q, self.cube, generate_cube_pos(*point), self.viz)
        return not success, q
    
    def check_edge_collision(self, start: np.ndarray, end: np.ndarray, start_q: Optional[np.ndarray]=None) -> (bool, Optional[np.ndarray], Optional[np.ndarray]): # (collision, best end, best end's q)
        # Sample along the edge and check for collisions using linspace, from start to end, return last non-collision sample
        samples = np.linspace(start, end, self.collision_samples)
        for (i, sample) in enumerate(samples):
            if i == 0:
                # Theoretically, we should check the start point, but we assume it was checked before
                continue
            collision, start_q = self.check_point_collision(sample, start_q)
            if collision:
                if i == 1:
                    return True, None, None
                else:
                    return True, samples[i-1], start_q
        return False, end, start_q
    
    def solve(self, max_iterations: int = 500, post_goal_iterations: int = 100, verbose=True) -> List[RTTNode] | None:
        try:
            r = super().solve(max_iterations, post_goal_iterations, verbose)
            self.clear_paths()
            return r
        except KeyboardInterrupt:
            self.clear_paths()
            raise KeyboardInterrupt()
        
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
                end_q = result[2]

                if not goal_direct_collision:
                    # We can connect directly to the goal! Does another path already exist?
                    if (self.goal_node.parent is None):
                        # No path exists, lets connect directly to the goal
                        self.connect_to_end(new_node, end_q)
                        return
                    # Otherwise, lets compare the existing path to the new path
                    existing_path_cost = self.get_path(self.goal_node)[1]
                    new_path_cost = current_cost + np.linalg.norm(new_clamped_point - self.goal)
                    if new_path_cost < existing_path_cost:
                        # We have a better path, lets connect directly to the goal
                        self.connect_to_end(new_node, end_q)
                        return

    def connect_to_end(self, new_node, end_q):
        self.goal_node.parent = new_node
        self.goal_node.q = end_q