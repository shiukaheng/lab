from path import computepath, computepathwithcubepos
from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_path_2d(robot, cube, viz, iters=10, step_size=0.05, color='blue', neighborhood_radius=0.1, collision_samples=3, max_neighbors=10, max_iterations=100, post_goal_iterations=0, shortcut_iterations=0, plot_title="RRT Path Visualization"):
    path_costs = []
    solve_durations = []
    plt.figure()  # Initialize a new figure
    for i in range(iters):
        q = robot.q0.copy()
        q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
        qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
        if not (successinit and successend):
            print("error: invalid initial or end configuration")
            continue  # Skip to the next iteration if there's an error
        print(f"Pass {i+1} of {iters}")
        # Time the solve
        start_time = time.time()
        path = computepathwithcubepos(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube, viz, step_size=step_size, neighborhood_radius=neighborhood_radius, collision_samples=collision_samples, max_neighbors=max_neighbors, max_iterations=max_iterations, post_goal_iterations=post_goal_iterations, shortcut_iterations=shortcut_iterations)
        solve_duration = time.time() - start_time
        solve_durations.append(solve_duration)
        path = [node[0] for node in path]
        # Calculate the path cost scalar
        cost = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        path_costs.append(cost)
        plt.axis([-0.4, 0.2, 0.9, 1.3])
        plt.plot([node[1] for node in path], [node[2] for node in path], color=color, linewidth=2)

    print(f"Average solve duration: {np.mean(solve_durations)}")
    print(f"Average path cost: {np.mean(path_costs)}")

    plt.title(plot_title)  # Set the plot title
    plt.xlabel("X-axis")  # Optionally, set labels for the axes
    plt.ylabel("Y-axis")
    plt.grid(True)  # Adding grid for better visualization
    plt.show()