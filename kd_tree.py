from typing import Optional, List
import numpy as np

# class KDTreeNode:
#     def __init__(self, point: np.ndarray, left, right):
#         self.point = point

# class KDTree:
#     def __init__(self, dimensions: int, node_class = KDTreeNode):
#         self.root = None
#         self.dimensions = dimensions
#         self.node_class = node_class
#         self.nodes = []

#     def insert(self, point: np.ndarray) -> KDTreeNode:
#         self.nodes.append(self.node_class(point))
#         return self.nodes[-1]

#     def nearest_neighbor(self, query_point: np.ndarray) -> Optional[KDTreeNode]:
#         closest_point = None
#         closest_distance = float('inf')
#         for node in self.nodes:
#             dist = np.linalg.norm(node.point - query_point)
#             if dist < closest_distance:
#                 closest_distance = dist
#                 closest_point = node
#         return closest_point
    
#     def query_spheroid(self, center: np.ndarray, radius: float) -> List[KDTreeNode]:
#         return [node for node in self.nodes if np.linalg.norm(node.point - center) <= radius]

class KDTreeNode:
    def __init__(self, point: np.ndarray, left: Optional['KDTreeNode'] = None, right: Optional['KDTreeNode'] = None):
        self.point = point
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, dimensions: int, node_class = KDTreeNode):
        self.root = None
        self.dimensions = dimensions
        self.node_class = node_class
        self.nodes = []

    def insert(self, point: np.ndarray) -> KDTreeNode:
        def _insert_rec(node, point, depth):

            if node is None:
                new_node = self.node_class(point)
                return new_node, new_node

            axis = depth % self.dimensions
            if point[axis] < node.point[axis]:
                node.left, new_node = _insert_rec(node.left, point, depth + 1)
            else:
                node.right, new_node = _insert_rec(node.right, point, depth + 1)

            return node, new_node

        self.root, new_node = _insert_rec(self.root, point, 0)
        self.nodes.append(new_node)
        return new_node

    def nearest_neighbor(self, query_point: np.ndarray) -> Optional[KDTreeNode]:
        def _nearest(node, query_point, depth):
            if node is None:
                return float('inf'), None

            axis = depth % self.dimensions
            next_branch = None
            opposite_branch = None

            if query_point[axis] < node.point[axis]:
                next_branch = node.left
                opposite_branch = node.right
            else:
                next_branch = node.right
                opposite_branch = node.left

            best_dist, best_node = _nearest(next_branch, query_point, depth + 1)
            d = np.linalg.norm(query_point - node.point)
            
            if d < best_dist:
                best_dist = d
                best_node = node

            if abs(query_point[axis] - node.point[axis]) < best_dist:
                dist, node = _nearest(opposite_branch, query_point, depth + 1)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node

            return best_dist, best_node

        return _nearest(self.root, query_point, 0)[1]

    def query_spheroid(self, center: np.ndarray, radius: float) -> List[KDTreeNode]:
        def _query(node, center, radius, depth):
            if node is None:
                return []

            axis = depth % self.dimensions
            dist = np.linalg.norm(center - node.point)
            results = []
            if dist <= radius:
                results.append(node)

            if center[axis] - radius < node.point[axis]:
                results.extend(_query(node.left, center, radius, depth + 1))
            
            if center[axis] + radius > node.point[axis]:
                results.extend(_query(node.right, center, radius, depth + 1))

            return results

        return _query(self.root, center, radius, 0)