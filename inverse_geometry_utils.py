import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated, jointlimitscost, distanceToObstacle, get_colliding_pairs
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from tools import setupwithmeshcat
from setup_meshcat import updatevisuals
from tools import setcubeplacement
# Import BFGS
from scipy.optimize import fmin_slsqp, fmin_bfgs, fmin
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

def distanceToObstacle(robot, q, computeFrameForwardKinematics=True, computeGeometryPlacements=True):
    '''Return the shortest distance between robot and the obstacle. '''

    geomidobs = robot.collision_model.getGeometryId('obstaclebase_0')
    geomidtable = robot.collision_model.getGeometryId('baseLink_0')

    pairs = [i for i, pair in enumerate(robot.collision_model.collisionPairs) if pair.second == geomidobs or pair.second == geomidtable]

    if computeFrameForwardKinematics:
        pin.framesForwardKinematics(robot.model,robot.data,q)
    if computeGeometryPlacements:
        pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)

    dists = [pin.computeDistance(robot.collision_model, robot.collision_data, idx).min_distance for idx in pairs]      
    
    return np.min(dists)

def weirdPostureCost(robot, q):
    # L2 of the difference between the current posture and the initial posture
    return np.linalg.norm(q - robot.q0) ** 2

def selfCollisionDistance(robot, q, computeFrameForwardKinematics=True, computeGeometryPlacements=True):
    '''Return the shortest distance between robot and the obstacle. '''
    geomidobs = robot.collision_model.getGeometryId('obstaclebase_0')
    geomidtable = robot.collision_model.getGeometryId('baseLink_0')
    pairs = [i for i, pair in enumerate(robot.collision_model.collisionPairs) if not (pair.second == geomidobs or pair.second == geomidtable)]
    # print([(robot.collision_model.geometryObjects[pair.first].name, robot.collision_model.geometryObjects[pair.second].name) for pair in robot.collision_model.collisionPairs])
    if computeFrameForwardKinematics:
        pin.framesForwardKinematics(robot.model,robot.data,q)
    if computeGeometryPlacements:
        pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
    dists = [pin.computeDistance(robot.collision_model, robot.collision_data, idx).min_distance for idx in pairs]      
    # print(dists)
    return np.min(dists)

def forcefield(distance, threshold=0.1, multiplier=10):
    return max((-distance + threshold), 0) * multiplier

def success(robot, cube, q):
    # No collisions, no joint limits violated, and the cube is grasped (cost < 0.1)
    collision_ok = not collision(robot, q)
    joint_limits_ok = not jointlimitsviolated(robot, q)
    effector_distance_ok = effector_distance_cost(robot, cube) < 0.05
    issue = ""
    if not (collision_ok and joint_limits_ok and effector_distance_ok):
        issue = f"Collision: {'✅' if collision_ok else '❌'}, Joint Limits: {'✅' if joint_limits_ok else '❌'}, Effector Distance Cost: {'✅' if effector_distance_ok else '❌'}, Colliding Pairs: {get_colliding_pairs(robot, q)}"

    return collision_ok and joint_limits_ok and effector_distance_ok, issue

def to_compact(full_q):
    return np.delete(full_q, [1, 2])

def to_full(compact_q):
    return np.insert(compact_q, 1, [0, 0])

def generate_cube_pos(x=None, y=None, z=None):
    x = x if x is not None else np.random.uniform(0.4, 0.5)
    y = y if y is not None else np.random.uniform(-0.4, 0.4)
    z = z if z is not None else np.random.uniform(0.9, 1.1)
    return pin.SE3(rotate('z', 0.),np.array([x, y, z]))