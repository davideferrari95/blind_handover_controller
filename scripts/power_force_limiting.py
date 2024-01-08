#! /usr/bin/env python3

import time, numpy as np
from termcolor import colored

# Import ROS2 Libraries
import rclpy
from rclpy.node import Node, Rate

# Import ROS2 Messages
from geometry_msgs.msg import PoseStamped, Vector3
from sensor_msgs.msg import JointState

from robot_toolbox import UR_Toolbox

class PowerForceLimitingController(Node):

    # Human Point Variables
    human_point, human_vel = Vector3(), Vector3()
    human_timer = time.time()

    def __init__(self, rate:Rate, robot:UR_Toolbox, robot_parameters:dict, human_radius:float=0.1, complete_debug:bool=False, debug:bool=False):

        """ Power Force Limiting (PFL) Controller """

        # Node Initialization
        super().__init__('pfl_controller')

        # ROS Parameters
        self.rate = rate

        # Class Parameters
        self.robot = robot
        self.complete_debug, self.debug = complete_debug, debug or complete_debug

        # Human Parameters
        self.human_radius = human_radius

        # Robot Parameters
        self.payload = robot_parameters['payload']
        self.reach = robot_parameters['reach']
        self.max_speed = robot_parameters['max_speed']
        self.stopping_time = robot_parameters['stopping_time']
        self.stopping_distance = robot_parameters['stopping_distance']
        self.position_uncertainty = robot_parameters['position_uncertainty']

        # ROS Subscribers
        self.human_pose_subscriber = self.create_subscription(PoseStamped, '/vrpn_client_node/polsino_dn/pose', self.humanPointCallback, 1)

    def humanPointCallback(self, msg:PoseStamped):

        """ Human Pose Callback (PH) """

        # Update Human PointStamped Message
        # self.human_point.header.frame_id = 'vision'
        # self.human_point.point = msg.pose.position

        # Compute Human Velocity
        self.human_vel.x, self.human_vel.y, self.human_vel.z = (msg.pose.position.x - self.human_point.x) / (time.time() - self.human_timer), \
                                                               (msg.pose.position.y - self.human_point.y) / (time.time() - self.human_timer), \
                                                               (msg.pose.position.z - self.human_point.z) / (time.time() - self.human_timer)

        # Update Human Timer
        self.human_timer = time.time()

        # Update Human Vector3 Message
        self.human_point.x, self.human_point.y, self.human_point.z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

    def compute_robot_point(self, joint_states:JointState) -> Vector3:

        """ Compute Robot Point (PR) """

        # Get the Robot Pose from the Joint States
        cartesian_pose = self.robot.matrix2pose(self.robot.ForwardKinematic(np.array(joint_states.position)))

        # Create PointStamped Message
        robot_point = Vector3()
        robot_point.x, robot_point.y, robot_point.z = cartesian_pose.position.x, cartesian_pose.position.y, cartesian_pose.position.z

        return robot_point

    def compute_versor(self, v1:Vector3, v2:Vector3) -> Vector3:

        """ Compute Versor """

        # Compute Distance (Euclidean Norm) Between Two Points
        distance = np.linalg.norm(np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z]))

        # return (P_H_ - P_R_) / (P_R_.distance(P_H_));
        return Vector3(x = (v1.x - v2.x) / distance, y = (v1.y - v2.y) / distance, z = (v1.z - v2.z) / distance)

    def compute_minimum_distance(self, v1:Vector3, v2:Vector3) -> float:

        """ Compute Minimum Distance Between Two Points """

        # Compute Distance (Euclidean Norm) Between Two Points
        return np.linalg.norm(np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z]))

    def compute_ISO_vel_lim(self, P_H:Vector3, P_R:Vector3, hr_versor:Vector3) -> float:

        """ Compute ISO/TS 15066 Velocity Limit """

        """
            A safety-aware kinodynamic architecture for human-robot collaboration
            Andrea Pupa, Mohammad Arrfou, Gildo Andreoni, Cristian Secchi

            https://arxiv.org/pdf/2103.01818.pdf

            @ARTICLE{9385862,
                author={Pupa, Andrea and Arrfou, Mohammad and Andreoni, Gildo and Secchi, Cristian},
                journal={IEEE Robotics and Automation Letters},
                title={A Safety-Aware Kinodynamic Architecture for Human-Robot Collaboration},
                year={2021},
                volume={6},
                number={3},
                pages={4465-4471},
                doi={10.1109/LRA.2021.3068634}
            }

            Sp(t0) = Sh + Sr + Ss + C + Zd + Zr

            Sh = Vh(t0) * (Ts + Tr)
            Sr = Vr(t0) * Tr
            Ss = Vr(t0) * Ts + a_max * Ts^2 / 2

                Sp = Safety Margin (Distance Between Human and Robot)
                Sh = Safety Margin from Human Velocity
                Sr = Safety Margin from Robot Velocity
                Ss = Safety Margin from Robot Stopping Time
                C = Intrusion Distance (Distance Needed for the Sensor to Detect the Human)
                Zd, Zr = Position Uncertainty (Human and Robot)
                Tr, Ts = Robot Reaction and Stopping Time
                Vh = Directed Speed of the Human Towards the Robot
                Vr = Directed Speed of the Robot Towards the Human
                Vs = Speed of the Robot in the Course of Stopping
                a_max = Maximum Robot Acceleration

            Maximum Robot Velocity (Vr_max) is computed as follows:
            Vr_max(t0) = (Sp(t0) - Vh(t0) * (Ts + Tr) - C - Zd - Zr) / (Ts + Tr) - (a_max * Ts) / (2 * (Ts + Tr))

        """

        # Compute Minimum Distance Between Human and Robot
        Sp = self.compute_minimum_distance(P_H, P_R) - self.human_radius

        # Uncertainties Parameters, Stopping Time and Reaction Time (Assume Tr, C, Zd, Zr = 0)
        Ts, Tr, C, Zd, Zr = self.stopping_time, 0.0, 0.0, 0.0, 0.0

        # Compute Human Projected Velocity
        Vh = np.array([self.human_vel.x, self.human_vel.y, self.human_vel.z]) @ np.array([hr_versor.x, hr_versor.y, hr_versor.z])
        if self.complete_debug: print(colored('Human Velocity: ', 'green'), f'{self.human_vel.x} {self.human_vel.y} {self.human_vel.z}')
        if self.complete_debug: print(colored('Human Projected Velocity: ', 'green'), Vh, '\n')

        # Robot Maximum Acceleration
        a_max = np.deg2rad(self.max_speed) / self.stopping_time

        # Compute ISO/TS 15066 Velocity Limit
        return (Sp - Vh * (Ts + Tr) - C - Zd - Zr) / (Ts + Tr) - (a_max * Ts**2) / (2 * (Ts + Tr))

    def compute_pfl_velocity(self, desired_joint_velocity:np.ndarray,  joint_states:JointState)  -> np.ndarray:

        """ Compute Power and Force Velocity Limit (PFL) """

        # Compute PH and PR Vector3
        P_H, P_R = self.human_point, self.compute_robot_point(joint_states)

        if self.debug or self.complete_debug: print(colored('\nPFL Controller:\n', 'green'))
        if self.complete_debug: print(colored('Human Point: ', 'green'), P_H)
        if self.complete_debug: print(colored('Robot Point: ', 'green'), P_R, '\n')

        # Compute Versor
        hr_versor = self.compute_versor(P_H, P_R)
        if self.complete_debug: print (colored('HR Versor: ', 'green'), f'{hr_versor.x} {hr_versor.y} {hr_versor.z}\n')

        # Compute Maximum Robot Velocity according to ISO/TS 15066
        vel_limit = self.compute_ISO_vel_lim(P_H, P_R, hr_versor)
        if self.debug or self.complete_debug: print(colored('ISO/TS 15066 Velocity Limit: ', 'green'), vel_limit, '\n')

        # Compute Robot Projected Desired Velocity
        x_dot: np.ndarray = self.robot.Jacobian(joint_states.position) @ np.array(desired_joint_velocity)
        Vr = np.array([x_dot[0], x_dot[1], x_dot[2]]) @ np.array([hr_versor.x, hr_versor.y, hr_versor.z])
        if self.complete_debug: print(colored('Robot Desired Velocity: ', 'green'), x_dot)
        if self.complete_debug: print(colored('Robot Projected Desired Velocity: ', 'green'), Vr, '\n')

        # Compute Scaling Factor (Alpha = V_max / Vr)
        scaling_factor =  vel_limit / Vr
        if self.debug or self.complete_debug: print(colored('Scaling Factor: ', 'green'), scaling_factor, '\n')

        # Compute Scaled Joint Velocity (alpha < 0 -> robot moving away from human)
        if scaling_factor >= 1: return desired_joint_velocity
        elif 0 < scaling_factor < 1: return desired_joint_velocity * scaling_factor
        elif scaling_factor <= 0: return desired_joint_velocity
