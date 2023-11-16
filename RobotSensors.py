import numpy as np
from setup_pybullet import Simulation
from pinocchio.robot_wrapper import RobotWrapper

class RobotSensors:
    def __init__(self, sim: Simulation, robot: RobotWrapper):
        self.sim = sim
        self.robot = robot
        self.joints = ['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
    def readJointPos(self):
        return np.array([self.sim.getJointPos(joint) for joint in self.joints])
    def readJointVel(self):
        return np.array([self.sim.getJointVel(joint) for joint in self.joints])