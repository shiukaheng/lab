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
# Goal 3: Output interpolated_frames parts of the path too

from rrt_star import *
from rrt_star import RRTStarNode
import meshcat.geometry as g
class RRTStarIGNode(RRTStarNode):
    def __init__(self, 
                 point: np.ndarray, 
                 left: Optional['RRTStarNode'] = None, 
                 right: Optional['RRTStarNode'] = None,
                 ):
        super().__init__(point, left, right)
        self.q = None
        self.interpolated_frames = []


class RRTStarIG(RRTStar):
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
                    shortcut_tolerance: float = 0.05,
                    max_neighbours: Optional[int] = 5,
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
                node_class=RRTStarIGNode,
            )
            self.robot = robot
            self.cube = cube
            self.viz = viz
            self.q_init = q_init if q_init is not None else robot.q0.copy()
            self.initial_node.q = self.q_init
            self.meshcat_paths = []
            self.meshcast_best_paths = []
            self.shortcut_tolerance = shortcut_tolerance
            self.max_neighbours = max_neighbours

    def sample_random_point(self):
        if (self.goal is not None) and (np.random.uniform() < self.bias):
            return self.goal, True
        # elif self.goal_node is not None and self.goal_node.parent is not None: # Informed RRT*
        else:
            new_point = np.random.uniform(self.lower_bounds, self.upper_bounds)
            return new_point, False

    def plot_node_segment(self, start_node: RRTStarIGNode, end_node: RRTStarIGNode):
        # Plot a segment between start and end, and add the meshcat path to the segment array
        if self.viz is None:
            return
        path = f"/path_{hash(start_node)}_{hash(end_node)}"
        # print(f"Start: {start.point}, End: {end.point}")
        self.plot_segment(start_node.point, end_node.point, path)
        self.meshcat_paths.append(path)

    def plot_segment(self, start, end, path, color=0xff0000):
        self.viz[path].set_object(g.Line(g.PointsGeometry(np.array([start, end]).transpose()), g.MeshBasicMaterial(color=color, wireframeLinewidth=5)))

    def plot_expanded_path(self, expanded_path):
        if self.viz is None:
            return
        # Clear existing best path
        for path in self.meshcast_best_paths:
            self.viz[path].delete()
        expanded_cube_path = [p[0] for p in expanded_path]
        self.viz["/expanded_path"].set_object(g.Line(g.PointsGeometry(np.array(expanded_cube_path).transpose()), g.MeshBasicMaterial(color=0x00ff00, wireframeLinewidth=5)))
        self.meshcast_best_paths.append("/expanded_path")

    def plot_best_goal_path(self):
        path, cost = self.get_path(self.goal_node)
        self.plot_expanded_path(path)

    def clear_paths(self):
        for path in self.meshcat_paths:
            self.viz[path].delete()
        for path in self.meshcast_best_paths:
            self.viz[path].delete()
        self.meshcat_paths = []
        self.meshcast_best_paths = []

    def sample(self) -> np.ndarray:
        # Lets find the nearest neighbor to target
        # self.print_nn()

        # Sample a new point
        new_point, biased = self.sample_random_point()
        nearest_node, new_point_clamped, new_q, reached_goal, interpolated = self.clamp_sampled_point(new_point, biased)

        # If we don't have any valid point, lets try again with a new sample
        if new_point_clamped is None:
            return self.sample()
        
        if reached_goal and self.get_hypothetical_cost(nearest_node) < self.get_goal_cost():
            self.goal_node.parent = nearest_node
            self.goal_node.q = new_q
            self.goal_node.interpolated_frames = interpolated
            return self.goal_node
        
        # Now that we have a new point, lets insert it into the kd-tree
        new_node = self.insert_with_solved_q(new_point_clamped, new_q)

        # Search for neighbors within a certain radius and see if we can find a better parent
        neighbors_in_reach, best_parent, best_neighbor_collision_return = self.find_reachable_neighbours(new_node)
        
        # Now that we have a parent, lets update the node
        self.link_nodes(new_node, best_parent, best_neighbor_collision_return)
        
        # Now, lets see if we can help any of the neighbors reduce their cost by making them point to the new node
        # Lets first get our own cost to the root
        current_cost = self.reduce_neighbour_cost(new_node, neighbors_in_reach, best_parent)
        
        # Did we reach the goal?
        self.handle_goal(new_point_clamped, new_node, current_cost)
        return new_node

    def reduce_neighbour_cost(self, new_node, neighbors_in_reach, best_parent):
        # print("New node:", new_node.point)
        # print("Neighbors in reach:", neighbors_in_reach)
        # print("Best parent:", best_parent)
        _, current_cost = self.get_path(new_node)
        for neighbor, neighbor_details in [n for n in neighbors_in_reach if n[0] != best_parent]:
            # Lets see if we can get a better cost by going through the new node
            new_cost = current_cost + np.linalg.norm(new_node.point - neighbor.point)
            old_cost = self.get_path(neighbor)[1]
            if new_cost < old_cost:
                # We can get a better cost, lets update the neighbor
                neighbor.parent = new_node
                neighbor.q = neighbor_details[2]
                neighbor.interpolated_frames = neighbor_details[3]
        return current_cost

    def link_nodes(self, child, parent, collision_return):
        child.parent = parent
        child.q = collision_return[2]
        child.interpolated_frames = collision_return[3]
        self.plot_node_segment(parent, child)

    def find_reachable_neighbours(self, new_node):
        neighbors = self.query_spheroid(new_node.point, self.neighbor_radius)

        # Lets filter away all neighbors that are not reachable from the new point
        neighbors_expanded = [(n, self.check_edge_collision(n.point, new_node.point, n.q)) for n in neighbors if n != new_node]
        neighbors_in_reach = [n for n in neighbors_expanded if n[1][0] == False]
        if self.max_neighbours is not None and len(neighbors_in_reach) > self.max_neighbours:
            # Sort the neighbors by distance to the new node and only keep the closest ones
            neighbors_in_reach = sorted(neighbors_in_reach, key=lambda n: np.linalg.norm(n[0].point - new_node.point))[:self.max_neighbours]

        best_cost = np.inf
        best_parent = None
        best_neighbor_collision_return = None
        for neighbor, neighbor_collision_return in neighbors_in_reach:
            cost = self.get_hypothetical_cost(neighbor)
            if cost < best_cost:
                best_cost = cost
                best_parent = neighbor
                best_neighbor_collision_return = neighbor_collision_return

        if best_parent is None:
            raise RuntimeError("No valid parent found, increase neighbor radius. Should implement a better way to handle this. Cant delete node in KDTree, so we have to handle other way")
        return neighbors_in_reach, best_parent, best_neighbor_collision_return

    def insert_with_solved_q(self, new_point_clamped, new_q):
        new_node = self.insert(new_point_clamped)
        new_node.q = new_q
        return new_node

    def print_nn(self):
        nn = self.nearest_neighbor(self.goal)
        if nn is not None:
            distance = np.linalg.norm(self.goal - nn.point)
            print("Distance to goal:", distance)

    def clamp_sampled_point(self, new_point, biased):
        nearest_node = self.nearest_neighbor(new_point)
        difference = new_point - nearest_node.point
        magnitude = np.linalg.norm(difference)
        reached_goal = False
        # Lets extend to the maximum step size or until we hit an obstacle
        if magnitude == 0: # For cases where we are already at the nearest node
            new_point_clamped = nearest_node.point
            new_q = None
            interpolated = []
        else:
            direction = difference / magnitude
            new_point_max = nearest_node.point + direction * min(self.step_size, magnitude)
            is_unclamped = magnitude <= self.step_size
            has_collision, new_point_clamped, new_q, interpolated = self.check_edge_collision(nearest_node.point, new_point_max, nearest_node.q)
            if (not has_collision) and is_unclamped and biased:
                reached_goal = True
        return nearest_node,new_point_clamped,new_q,reached_goal,interpolated
    
    def check_point_collision(self, point: np.ndarray, start_q: Optional[np.ndarray]=None) -> bool:
        if start_q is None:
            start_q = self.q_init
        q, success = computeqgrasppose(self.robot, start_q, self.cube, generate_cube_pos(*point), self.viz)
        return not success, q
    
    def check_edge_collision(self, start: np.ndarray, end: np.ndarray, start_q: Optional[np.ndarray]=None) -> (bool, Optional[np.ndarray], Optional[np.ndarray], List[np.ndarray]): # (collision, best end, best end's q)
        # Sample along the edge and check for collisions using linspace, from start to end, return last non-collision sample
        distance = np.linalg.norm(end - start)
        samples = np.linspace(start, end, int(self.collision_samples * distance / self.step_size))
        interpolated = [] # (point, q)
        for (i, sample) in enumerate(samples):
            if i == 0:
                # Theoretically, we should check the start point, but we assume it was checked before
                continue
            collision, start_q = self.check_point_collision(sample, start_q)
            if i != len(samples) - 1:
                interpolated.append((sample, start_q))
            if collision:
                if i == 1:
                    return True, None, None, interpolated
                else:
                    return True, samples[i-1], start_q, interpolated
        return False, end, start_q, interpolated
    
    def solve(self, max_iterations: int = 500, post_goal_iterations: int = 100, shortcut_iterations: int = 500, verbose=True) -> List[RRTStarNode] | None:
        try:
            if verbose:
                print("=====================================")
            path = super().solve(max_iterations, post_goal_iterations, verbose)
            self.clear_paths()
            if path is None:
                return None
            shortcut_optimized = self.path_shortcut(path, shortcut_iterations, verbose)
            self.clear_paths()
            if verbose:
                print("=====================================")
            return shortcut_optimized
        except KeyboardInterrupt:
            self.clear_paths()
            raise KeyboardInterrupt()
        
    def expand_path(self, path):
        # We take the path and expand it into a flat list of (point, q) pairs, including the interpolated frames
        expanded_path = []
        for node in path:
            if len(node.interpolated_frames) > 0:
                expanded_path.extend(node.interpolated_frames)
            expanded_path.append((node.point, node.q))
        return expanded_path
    
    def path_shortcut_once(self, expanded_path):
        # Select first random point
        first_index = np.random.randint(0, len(expanded_path) - 1)
        # Select the offset from the first point [1, len - 1]
        second_index_offset = np.random.randint(1, len(expanded_path) - 1)
        # Select the second point
        second_index = (first_index + second_index_offset) % len(expanded_path)
        # Sort the indices
        first_index, second_index = sorted([first_index, second_index])
        # Lets retrieve the points
        first_point, first_q = expanded_path[first_index]
        second_point, second_q = expanded_path[second_index]
        # Lets check if we can connect the points
        has_collision, best_point, best_q, interpolated = self.check_edge_collision(first_point, second_point, first_q)
        if has_collision or second_q is None:
            return expanded_path
        # Lets check if the ending q is (almost) the same as the starting q, if not, we discard the shortcut
        if np.linalg.norm(best_q - second_q) > self.shortcut_tolerance:
            return expanded_path
        # We can connect the points! Lets cut the original path into three parts (before, discarded, after)
        before = expanded_path[:first_index+1]
        discarded = expanded_path[first_index:second_index]
        after = expanded_path[second_index:]
        # Now we can replace the discarded part with the interpolated path
        return before + interpolated + after
    
    def expanded_path_length(self, expanded_path):
        return sum([np.linalg.norm(p[0] - q[0]) for p, q in zip(expanded_path[:-1], expanded_path[1:])])
    
    def path_shortcut(self, path, iterations=100, verbose=True):
        expanded_path = self.expand_path(path)
        original_length = self.expanded_path_length(expanded_path)
        for i in range(iterations):
            expanded_path = self.path_shortcut_once(expanded_path)
            print(f"✨ Local path optimization: {int((i+1)/iterations*100)}%, length reduced by {int((original_length-self.expanded_path_length(expanded_path))/original_length*100)}%", end="\r")
            self.plot_expanded_path(expanded_path)
            if len(expanded_path) <= 2:
                print()
                print("✨ Path reached minimum length, stopping") # Should not happen
                break
        print(f"✅ Local path optimization: Done! Path length reduced by: {int((original_length-self.expanded_path_length(expanded_path))/original_length*100)}%            ")
        return expanded_path
        
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
                interpolated = result[3]

                if not goal_direct_collision:
                    # We can connect directly to the goal! Does another path already exist?
                    if (self.goal_node.parent is None):
                        # No path exists, lets connect directly to the goal
                        self.connect_to_end(new_node, end_q, interpolated)
                        return
                    # Otherwise, lets compare the existing path to the new path
                    existing_path_cost = self.get_path(self.goal_node)[1]
                    new_path_cost = current_cost + np.linalg.norm(new_clamped_point - self.goal)
                    if new_path_cost < existing_path_cost:
                        # We have a better path, lets connect directly to the goal
                        self.connect_to_end(new_node, end_q, interpolated)
                        return

    def connect_to_end(self, new_node, end_q, interpolated):
        self.goal_node.parent = new_node
        self.goal_node.q = end_q
        self.goal_node.interpolated_frames = interpolated