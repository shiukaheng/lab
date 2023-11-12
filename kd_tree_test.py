import unittest
import numpy as np
from kd_tree import KDTree

class TestKDTree(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)  # for reproducible tests
        self.points = np.random.rand(100, 3)  # 100 points in 3 dimensions
        self.tree = KDTree(3)
        for point in self.points:
            self.tree.insert(point)

    def test_nearest_neighbor(self):
        for point in self.points:  # test with first 10 points
            expected_nearest = self.brute_force_nearest_neighbor(point)
            kd_tree_nearest = self.tree.nearest_neighbor(point)
            np.testing.assert_array_almost_equal(kd_tree_nearest.point, expected_nearest)

    def test_query_spheroid(self):
        center = np.array([0.5, 0.5, 0.5])
        radius = 0.2
        expected_points = self.brute_force_query_spheroid(center, radius)
        kd_tree_points = self.tree.query_spheroid(center, radius)
        self.assertEqual(len(expected_points), len(kd_tree_points))
        for node in kd_tree_points:
            self.assertTrue(any(np.array_equal(node.point, point) for point in expected_points))

    def brute_force_nearest_neighbor(self, query_point):
        closest_point = None
        closest_distance = float('inf')
        for point in self.points:
            dist = np.linalg.norm(point - query_point)
            if dist < closest_distance:
                closest_distance = dist
                closest_point = point
        return closest_point

    def brute_force_query_spheroid(self, center, radius):
        return [point for point in self.points if np.linalg.norm(point - center) <= radius]

if __name__ == '__main__':
    unittest.main()
