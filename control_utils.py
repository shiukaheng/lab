import numpy as np
import scipy

from bezier import Bezier


def create_linear_velocity_profile(total_duration, ramp_time, n_samples):
    # Time for each sample
    dt = total_duration / n_samples
    # Create time array
    time = np.linspace(0, total_duration, n_samples)
    # Ramp up profile
    ramp_up = np.minimum(time / ramp_time, 1)
    # Ramp down profile
    ramp_down = np.minimum((total_duration - time) / ramp_time, 1)
    # Constant velocity profile
    constant_velocity = np.ones(n_samples)
    # Combine the three profiles
    velocity_profile = np.minimum(np.minimum(ramp_up, ramp_down), constant_velocity)

    return velocity_profile

def resample_path(waypoints, cube_waypoints, velocity_profile, total_time, sampling_rate):
    if len(waypoints) != len(cube_waypoints):
        raise ValueError("waypoints and cube_waypoints must be of identical length")

    # Calculate cumulative distance along cube_waypoints
    distances = np.cumsum(np.sqrt(np.sum(np.diff(cube_waypoints, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Insert 0 at the start

    # Normalize distances to [0, 1]
    normalized_distances = distances / distances[-1]

    # Normalize the velocity profile
    t_norm = np.linspace(0, 1, len(velocity_profile))
    t_total = np.linspace(0, total_time, int(total_time * sampling_rate))

    # Interpolate and integrate velocity profile to get position
    velocity_interp = scipy.interpolate.interp1d(t_norm, velocity_profile, kind='linear', fill_value="extrapolate")
    position_profile = np.cumsum(velocity_interp(np.linspace(0, 1, len(t_total))) / sampling_rate)

    # Normalize the position profile to [0, 1]
    position_profile = position_profile / position_profile[-1]

    # Interpolate joint space waypoints according to the normalized position profile
    joint_trajectory = []
    for dim in range(np.array(waypoints).shape[1]):
        waypoint_interp = scipy.interpolate.interp1d(normalized_distances, np.array(waypoints)[:, dim], kind='linear')
        joint_trajectory.append(waypoint_interp(position_profile))

    return np.array(joint_trajectory).T

def create_bezier_trajectory(trajectory, t_max):
    # Create Bezier curve from trajectory points
    bezier_curve = Bezier(trajectory, t_max=t_max)

    # Compute first and second derivatives (velocity and acceleration)
    velocity_curve = bezier_curve.derivative(1)
    acceleration_curve = bezier_curve.derivative(2)

    return bezier_curve, velocity_curve, acceleration_curve

def create_naive_bezier_trajectory(waypoints, cube_waypoints, total_time, ramp_time, n_samples=1000):
    
    # Create velocity profile
    velocity_profile = create_linear_velocity_profile(total_time, ramp_time, n_samples)

    # Resample path according to velocity profile
    pose_trajectory = resample_path(waypoints, cube_waypoints, velocity_profile, 1, 1000) # Forcing it to have 1000 samples, because higher samples cause numerical issues with Bezier curve
    # cube_trajectory = resample_path(cube_waypoints, cube_waypoints, velocity_profile, 1, 1000) # Forcing it to have 1000 samples, because higher samples cause numerical issues with Bezier curve
    # Create Bezier curve from trajectory points
    pose_bezier_curve, pose_velocity_curve, pose_acceleration_curve = create_bezier_trajectory(pose_trajectory, t_max=total_time) # Relying on Bezier t_max to stretch the trajectory to the desired duration
    return pose_bezier_curve, pose_velocity_curve, pose_acceleration_curve

def create_linear_trajectory(waypoints, cube_waypoints, total_time, ramp_time, n_samples=1000):

    # Create velocity profile
    velocity_profile = create_linear_velocity_profile(total_time, ramp_time, n_samples)
    # Resample path according to velocity profile
    pose_trajectory = resample_path(waypoints, cube_waypoints, velocity_profile, total_time, n_samples)
    return pose_trajectory