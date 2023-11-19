from typing import List, Optional
import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time
from inverse_geometry import computeqgrasppose
from inverse_geometry_utils import generate_cube_pos, to_full
import meshcat.transformations as tf

# Goal 1: Check collision using inverse geometry
# Goal 2: Provide better initial guess for inverse geometry
# Goal 3: Output interpolated_frames parts of the path too

def segment_dir(points, tolerance=0.01):
    if len(points) < 2:
        return [points]  # If there's only one point, return it as a single segment.

    segments = []
    current_segment = [points[0]]
    last_dir = None

    for i in range(1, len(points)):
        # Calculate slope
        diff =  points[i][0] - points[i-1][0]
        current_dir = diff / np.linalg.norm(diff)

        # Check if direction has changed
        if last_dir is not None and current_dir.dot(last_dir) < (1 - tolerance):
            # print("Direction changed from {} to {}".format(last_dir, current_dir))
            segments.append(current_segment)
            current_segment = [points[i-1]]

        # print("Adding point {} to segment {}".format(points[i], len(segments)))

        current_segment.append(points[i])
        last_dir = current_dir

    segments.append(current_segment)  # Append the last segment
    # Remove the first element of each segment if its not the first segment
    segments = [segment[1:] if i > 0 else segment for i, segment in enumerate(segments)]
    return segments

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
        path = f"/rrt/path_{hash(start_node)}_{hash(end_node)}"
        # print(f"Start: {start.point}, End: {end.point}")
        self.plot_graph_segment(start_node.point, end_node.point, path)

    def plot_graph_segment(self, start, end, path, color=0xff0000):
        self.viz[path].set_object(g.Line(g.PointsGeometry(np.array([start, end]).transpose()), g.MeshBasicMaterial(color=color, wireframeLinewidth=5)))

    def plot_explore_segment(self, start, end, color=0xff00ff):
        self.viz["/explore"].set_object(g.Line(g.PointsGeometry(np.array([start, end]).transpose()), g.MeshBasicMaterial(color=color, wireframeLinewidth=5)))

    def clear_explore_segment(self):
        self.viz["/explore"].delete()

    def plot_expanded_path(self, expanded_path, pathname="path", color=0x00ff00):
        if self.viz is None:
            return
        expanded_cube_path = [p[0] for p in expanded_path]
        self.viz[f"/rrt/{pathname}"].set_object(g.Line(g.PointsGeometry(np.array(expanded_cube_path).transpose()), g.MeshBasicMaterial(color=0x00ff00, wireframeLinewidth=5)))

    def plot_best_goal_path(self):
        path, cost = self.get_path(self.goal_node)
        self.plot_expanded_path(path)

    def plot_exploration_point(self, point):
        self.viz["/exploration_point"].set_object(g.Sphere(0.01), g.MeshBasicMaterial(color=0x00ffff))
        self.viz["/exploration_point"].set_transform(tf.translation_matrix(point))

    def clear_exploration_point(self):
        self.viz["/exploration_point"].delete()

    def clear_paths(self):
        self.viz["/rrt"].delete()
        self.clear_explore_segment()
        self.clear_exploration_point()

    def get_goal_cost(self):
        if self.goal_node is None:
            return np.inf
        elif self.goal_node.parent is None:
            return np.inf
        else:
            return self.get_node_cost(self.goal_node)
        
    def get_hypothetical_cost(self, node, new_point, new_q):
        return self.get_node_cost(node) + np.linalg.norm(node.point - new_point)
    
    def get_node_cost(self, node):
        return self.get_path(node)[1]

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

        # Search for neighbors within a certain radius and see if we can find a better parent
        neighbors_in_reach, best_parent, best_neighbor_collision_return = self.analyse_neighbors(new_point_clamped)

        if best_parent is None:
            # We did not find any neighbors within reach, lets try again with a new sample
            return self.sample()

        # Now that we have a new point, lets insert it into the kd-tree
        new_node = self.insert_with_solved_q(new_point_clamped, new_q)
        
        # Now that we have a parent, lets update the node
        self.link_nodes(new_node, best_parent, best_neighbor_collision_return)
        
        # Now, lets see if we can help any of the neighbors reduce their cost by making them point to the new node
        # Lets first get our own cost to the root
        self.reduce_neighbour_cost(new_node, neighbors_in_reach, best_parent)
        
        # Did we reach the goal?
        self.handle_goal(new_node)
        return new_node

    def reduce_neighbour_cost(self, new_node, neighbors_in_reach, best_parent):
        # COST CHECKING
        for neighbor, neighbor_details in [n for n in neighbors_in_reach if n[0] != best_parent]:
            # Lets see if we can get a better cost by going through the new node
            # new_cost = current_cost + np.linalg.norm(new_node.point - neighbor.point)
            new_cost = self.get_hypothetical_cost(new_node, neighbor.point, neighbor_details[2])
            # old_cost = self.get_path(neighbor)[1]
            old_cost = self.get_node_cost(neighbor)
            if new_cost < old_cost:
                # We can get a better cost, lets update the neighbor
                neighbor.parent = new_node
                neighbor.q = neighbor_details[2]
                neighbor.interpolated_frames = neighbor_details[3]

    def link_nodes(self, child, parent, collision_return):
        child.parent = parent
        child.q = collision_return[2]
        child.interpolated_frames = collision_return[3]
        self.plot_node_segment(parent, child)

    def analyse_neighbors(self, new_point):
        neighbors = self.query_spheroid(new_point, self.neighbor_radius)

        # Lets filter away all neighbors that are not reachable from the new point
        neighbors_expanded = [(n, self.check_edge_collision(n.point, new_point, n.q)) for n in neighbors]
        neighbors_in_reach = [n for n in neighbors_expanded if n[1][0] == False]
        if self.max_neighbours is not None and len(neighbors_in_reach) > self.max_neighbours:
            # Sort the neighbors by distance to the new node and only keep the closest ones
            neighbors_in_reach = sorted(neighbors_in_reach, key=lambda n: np.linalg.norm(n[0].point - new_point))[:self.max_neighbours]

        # COST CHECKING
        best_cost = np.inf
        best_parent = None
        best_neighbor_collision_return = None
        for neighbor, neighbor_collision_return in neighbors_in_reach:
            cost = self.get_hypothetical_cost(neighbor, new_point, neighbor_collision_return[2])
            if cost < best_cost:
                best_cost = cost
                best_parent = neighbor
                best_neighbor_collision_return = neighbor_collision_return

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
        self.plot_exploration_point(point)
        if start_q is None:
            start_q = self.q_init
        q, success = computeqgrasppose(self.robot, start_q, self.cube, generate_cube_pos(*point), self.viz)
        return not success, q
    
    def check_edge_collision(self, start: np.ndarray, end: np.ndarray, start_q: Optional[np.ndarray]=None) -> (bool, Optional[np.ndarray], Optional[np.ndarray], List[np.ndarray]): # (collision, best end, best end's q)
        self.plot_explore_segment(start, end)
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
    
    def sample_loop(self, max_iterations: int = 5000, post_goal_iterations: int = 1000, verbose=True) -> Optional[List[RRTStarNode]]:
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
            # Lets get the path length
            _, original_path_cost = self.get_path(self.goal_node)
            # Now, lets sample some more to see if we can find a better path
            for i in range(post_goal_iterations):
                self.sample()
                if verbose:
                    # Lets calculate the new path length
                    _, new_path_cost = self.get_path(self.goal_node)
                    print(f"✨ Global path optimization: {int((i+1)/post_goal_iterations*100)}%, length reduced by {int((original_path_cost - new_path_cost)/original_path_cost*100)}%     ", end="\r", flush=True)
            if verbose:
                if post_goal_iterations > 0:
                    print(f"✅ Global path optimization: Done! Path length reduced by {int((original_path_cost - new_path_cost)/original_path_cost*100)}%       ", flush=True)
            
        # Now, lets get the path
        path, _ = self.get_path(self.goal_node)
        # Reverse the path list
        path.reverse()
        return path
    
    # @cache_results
    def solve(self, max_iterations: int = 500, post_goal_iterations: int = 100, shortcut_iterations: int = 500, verbose=True) -> List[RRTStarNode] | None:
        try:
            path = self.sample_loop(max_iterations, post_goal_iterations, verbose)
            self.clear_paths()
            if path is None:
                return None
            shortcut_optimized = self.path_shortcut(path, shortcut_iterations, verbose)
            self.clear_paths()
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
        segments = segment_dir(expanded_path)
        if len(segments) <= 1:
            return expanded_path
        segment_weights = [len(s) for s in segments]
        # Select first random segment
        # first_segment = np.random.randint(0, len(segments) - 1)
        first_segment = np.random.choice(len(segments), p=segment_weights/np.sum(segment_weights))
        # Set the selected first segment weight to 0
        segment_weights[first_segment] = 0
        # Select the second random segment
        second_segment = np.random.choice(len(segments), p=segment_weights/np.sum(segment_weights))
        # Select an index for the first segment
        if len(segments[first_segment]) >= 2:
            first_segment_index = np.random.randint(0, len(segments[first_segment]) - 1)
        else:
            first_segment_index = 0
        # Select an index for the second segment
        if len(segments[second_segment]) >= 2:
            second_segment_index = np.random.randint(0, len(segments[second_segment]) - 1)
        else:
            second_segment_index = 0
        # Calculate the actual indices
        first_index = sum([len(s) for s in segments[:first_segment]]) + first_segment_index
        second_index = sum([len(s) for s in segments[:second_segment]]) + second_segment_index
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
        
    def handle_goal(self, new_node):
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

                    # COST CHECKING
                    # existing_path_cost = self.get_path(self.goal_node)[1]
                    existing_path_cost = self.get_node_cost(self.goal_node)
                    # new_path_cost = current_cost + np.linalg.norm(new_clamped_point - self.goal)
                    new_path_cost = self.get_hypothetical_cost(new_node, self.goal, end_q)
                    if new_path_cost < existing_path_cost:
                        # We have a better path, lets connect directly to the goal
                        self.connect_to_end(new_node, end_q, interpolated)
                        return

    def connect_to_end(self, new_node, end_q, interpolated):
        self.goal_node.parent = new_node
        self.goal_node.q = end_q
        self.goal_node.interpolated_frames = interpolated