import numpy as np
import pinocchio as pin 
import numpy as np
from tools import collision, getcubeplacement, jointlimitsviolated, jointlimitscost, get_colliding_pairs, setcubeplacement
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from tools import setupwithmeshcat

from inverse_geometry import computeqgrasppose
import time
from pinocchio.utils import rotate

def effector_distance_cost(robot, cube, inflationRadius=0.0005):
    # Calculate the distance cost from the effectors to the cube hooks
    oMl = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]  # Assuming 'left_hand' is the correct frame name
    oMr = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]  # Assuming 'right_hand' is the correct frame name
    oMhl = getcubeplacement(cube, LEFT_HOOK)  # Assuming 'left_hook' is the correct frame name
    oMhr = getcubeplacement(cube, RIGHT_HOOK)  # Assuming 'right_hook' is the correct frame name
    oMcube = getcubeplacement(cube)

    # Calculate the direction vectors from the cube to the hooks
    direction_l = oMhl.translation - oMcube.translation
    direction_r = oMhr.translation - oMcube.translation

    # Normalize the direction vectors
    norm_l = np.linalg.norm(direction_l)
    norm_r = np.linalg.norm(direction_r)
    direction_l = direction_l / norm_l if norm_l > 0 else direction_l
    direction_r = direction_r / norm_r if norm_r > 0 else direction_r

    # Apply the inflation radius to shift the hooks' positions outwards
    oMhl_inflated_translation = oMhl.translation + direction_l * inflationRadius
    oMhr_inflated_translation = oMhr.translation + direction_r * inflationRadius

    # Calculate the squared distances including the inflation
    dist_l = np.linalg.norm(oMl.translation - oMhl_inflated_translation) ** 2 + np.linalg.norm(oMl.rotation - oMhl.rotation) ** 2
    dist_r = np.linalg.norm(oMr.translation - oMhr_inflated_translation) ** 2 + np.linalg.norm(oMr.rotation - oMhr.rotation) ** 2

    # Return the sum of the squared distances
    return dist_l + dist_r

def get_colliding_pairs(robot, q):
    """
    Returns a list of tuples representing the colliding pairs of geometry objects.
    
    Parameters:
    robot: The robot instance with collision model and data.
    q: The configuration of the robot (joint positions).
    
    Returns:
    A list of tuples, where each tuple contains the names of the colliding geometry objects.
    """
    # Update the robot's geometry placements based on the current configuration q
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    
    # Compute collisions
    pin.computeCollisions(robot.collision_model, robot.collision_data, False)
    
    # List to hold the names of colliding pairs
    colliding_pairs = []
    
    # Iterate over all collision pairs and check if they are in collision
    for k in range(len(robot.collision_model.collisionPairs)):
        cr = robot.collision_data.collisionResults[k]
        cp = robot.collision_model.collisionPairs[k]
        if cr.isCollision():
            # Get the names of the colliding objects
            name1 = robot.collision_model.geometryObjects[cp.first].name
            name2 = robot.collision_model.geometryObjects[cp.second].name
            # Add the colliding pair to the list
            colliding_pairs.append((name1, name2))
    
    return colliding_pairs


def generate_cube_pos(x=None, y=None, z=None):
    x = x if x is not None else np.random.uniform(0.4, 0.5)
    y = y if y is not None else np.random.uniform(-0.4, 0.4)
    z = z if z is not None else np.random.uniform(0.9, 1.1)
    return pin.SE3(rotate('z', 0.),np.array([x, y, z]))

def success(robot, cube, q):
    # No collisions, no joint limits violated, and the cube is grasped (cost < 0.1)
    collision_ok = not collision(robot, q)
    joint_limits_ok = not jointlimitsviolated(robot, q)
    effector_distance_ok = effector_distance_cost(robot, cube) < 0.05
    issue = ""
    if not (collision_ok and joint_limits_ok and effector_distance_ok):
        issue = f"Collision: {'✅' if collision_ok else '❌'}, Joint Limits: {'✅' if joint_limits_ok else '❌'}, Effector Distance Cost: {'✅' if effector_distance_ok else '❌'}, Colliding Pairs: {get_colliding_pairs(robot, q)}"

    return collision_ok and joint_limits_ok and effector_distance_ok, issue

def success_debug(robot, cube, q):
    # No collisions, no joint limits violated, and the cube is grasped (cost < 0.1)
    collision_ok = not collision(robot, q)
    joint_limits_ok = not jointlimitsviolated(robot, q)
    effector_distance_ok = effector_distance_cost(robot, cube) < 0.05
    issue = ""
    if not (collision_ok and joint_limits_ok and effector_distance_ok):
        issue = f"Collision: {'✅' if collision_ok else '❌'}, Joint Limits: {'✅' if joint_limits_ok else '❌'}, Effector Distance Cost: {'✅' if effector_distance_ok else '❌'}, Colliding Pairs: {get_colliding_pairs(robot, q)}"

    return collision_ok, joint_limits_ok, effector_distance_ok, issue

def random_tests(robot, cube, viz, iters=50, seed=42, interactive=False):
    """
    Generate random tests and evaluate the results. Random tests are not guaranteed to have solutions but we can
    still evaluate relative performance.
    """

    q_init = robot.q0.copy()
    if not interactive:
        viz = None

    # Seed random number generator
    np.random.seed(seed)
    
    # Run tests
    test_results = []
    print(f"Running {iters} tests")
    for i in range(iters):
        cubetarget = generate_cube_pos()
        print("=====================================")
        print(f"Test {i+1}/{iters}: {cubetarget.translation}")
        res, is_success = computeqgrasppose(robot, q_init, cube, cubetarget, viz)
        if is_success:
            print("✅ Successfully solved")
        else:
            print("❌ Failed to solve")
        print("=====================================")
        print()
        col, joint, eff, iss = success_debug(robot, cube, res) # Evaluate why the test failed
        colliding_pairs = get_colliding_pairs(robot, res)
        test_results.append({
            "cube_target": cubetarget.translation,
            "result_pose": res,
            "success": is_success,
            "collision_ok": col,
            "colliding_pairs": colliding_pairs,
            "joint_limits_ok": joint,
            "effector_distance_ok": eff,
            "issue": iss
        })
        
    print()
    # Calculate the statistics
    num_success = sum([r["success"] for r in test_results])
    num_collision = iters - sum([r["collision_ok"] for r in test_results])
    num_joint_limits = iters - sum([r["joint_limits_ok"] for r in test_results])
    
    print("=====================================")
    print("Test Results")
    print()
    print(f"Success Rate: {num_success}/{iters} ({num_success/iters*100:.2f}%)")
    print(f"Collision Rate: {num_collision}/{iters} ({num_collision/iters*100:.2f}%)")
    print(f"Joint Limits Violated Rate: {num_joint_limits}/{iters} ({num_joint_limits/iters*100:.2f}%)")
    print("=====================================")
    print()
    if interactive:
        failed_tests = [r for r in test_results if not r["success"]]
        current_test_index = 0

        while current_test_index < len(failed_tests):
            test = failed_tests[current_test_index]
            print(f"Inspecting Failed Test {current_test_index + 1}/{len(failed_tests)}")
            print("Target Position:", test["cube_target"])
            print("Issue:", test["issue"])

            # Update robot's configuration to the failed test pose
            if viz is not None:
                setcubeplacement(robot, cube, generate_cube_pos(*test["cube_target"]))
                viz.display(test["result_pose"])

            command = input("Enter 'n' for next, 'p' for previous, 'r' to re-run, 'q' to quit: ").strip().lower()

            if command == "n":
                if current_test_index < len(failed_tests) - 1:
                    current_test_index += 1
                else:
                    print("No more tests to display.")
            elif command == "p":
                if current_test_index > 0:
                    current_test_index -= 1
                else:
                    print("This is the first test.")
            elif command == "r":
                q, is_success = computeqgrasppose(robot, robot.q0.copy(), cube, generate_cube_pos(*test["cube_target"]), viz)
                print("Re-run complete.")
            elif command == "q":
                print("Exiting test inspector.")
                break
            else:
                print("Invalid command. Please enter 'n', 'p', 'r', or 'q'.")

            print("=====================================")

        print("Completed inspection of all failed tests.")

def test_position(robot, cube, viz, q0, pose, name=None):
    print("=====================================")
    if name is not None:
        print(f"Testing {name}")
    else:
        print("Testing pose")
    q, is_success = computeqgrasppose(robot, q0, cube, pose, viz)
    print("=====================================")
    print()
    return q, is_success

def original_tests(robot, cube, viz, interactive=False):
    q = robot.q0.copy()
    test_position(robot, cube, viz, q, CUBE_PLACEMENT, "initial pose")
    test_position(robot, cube, viz, q, CUBE_PLACEMENT_TARGET, "target pose")
    test_position(robot, cube, viz, q, generate_cube_pos(0.5, 0.15, 0.93), "target pose modified")
    test_position(robot, cube, viz, q, generate_cube_pos(0.5, 0.11, 0.93), "target pose modified 2")
            
if __name__ == "__main__":
    robot, cube, viz = setupwithmeshcat()
    # original_tests(robot, cube, viz, interactive=True)
    random_tests(robot, cube, viz, interactive=True, iters=5)