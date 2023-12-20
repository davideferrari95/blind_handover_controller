#! /usr/bin/env python3

import numpy as np
from math import pi
from typing import List
from termcolor import colored
import threading, signal, time

# Import ROS2 Libraries
import rclpy
from rclpy.node import Node

# Import ROS Messages, Services, Actions
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench

# Import Robot and UR_RTDE Move Classes
from move_robot import UR_RTDE_Move
from robot_toolbox import UR_Toolbox

# Import Admittance and PFL Controllers Classes
from admittance import AdmittanceController
from power_force_limiting import PowerForceLimitingController

def signal_handler(sig, frame):

    """ UR Stop Signal Handler """

    # Initialize `rclpy` if not initialized
    if not rclpy.ok(): rclpy.init()

    # Create ROS2 Node + Publisher
    ur_stop = rclpy.create_node('ur_stop_node', enable_rosout=False)
    joint_group_vel_controller_publisher = ur_stop.create_publisher(Float64MultiArray, '/ur_rtde/controllers/joint_velocity_controller/command', 1)

    # Create Stop Message
    stop_msgs = Float64MultiArray(data=[0.0] * 6)
    stop_msgs.layout.dim.append(MultiArrayDimension())
    stop_msgs.layout.dim[0].size = 6
    stop_msgs.layout.dim[0].stride = 1
    stop_msgs.layout.dim[0].label = 'velocity'

    # Publish Stop Message
    joint_group_vel_controller_publisher.publish(stop_msgs)
    ur_stop.get_logger().error('Stop Signal Received. Stopping UR...')

    # Shutdown ROS
    time.sleep(2)
    rclpy.try_shutdown()

class Handover_Controller(Node):

    """ Handover Controller Class """

    # Initialize Subscriber Variables
    trajectory_executed, goal_received = False, False
    handover_cartesian_goal, joint_states = Pose(), JointState()
    joint_states, ft_sensor_data = JointState(), Wrench()
    x_dot_last_cycle = np.zeros((6, ), dtype=np.float64)

    # FIX: Initialize Joint States
    joint_states.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, node_name, ros_rate):

        # Node Initialization
        super().__init__(node_name)

        # ROS2 Rate
        self.rate = self.create_rate(ros_rate)

        # Spin in a separate thread - for ROS2 Rate
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self, ), daemon=True)
        self.spin_thread.start()

        # Declare Parameters
        self.declare_parameter('human_mass',            10.0)
        self.declare_parameter('robot_mass',            10.0)
        self.declare_parameter('admittance_mass',       [1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
        self.declare_parameter('admittance_damping',    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
        self.declare_parameter('admittance_stiffness',  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
        self.declare_parameter('maximum_velocity',      [1.05, 1.05, 1.57, 1.57, 1.57, 1.57])
        self.declare_parameter('maximum_acceleration',  [0.57, 0.57, 0.57, 0.57, 0.57, 0.57])
        self.declare_parameter('use_feedback_velocity', True)
        self.declare_parameter('complete_debug',        False)
        self.declare_parameter('debug',                 False)
        self.declare_parameter('robot',                 'ur10e')

        # Read Parameters
        self.human_mass            = self.get_parameter('human_mass').get_parameter_value().double_value
        self.robot_mass            = self.get_parameter('robot_mass').get_parameter_value().double_value
        admittance_mass            = self.get_parameter('admittance_mass').get_parameter_value().double_array_value
        admittance_damping         = self.get_parameter('admittance_damping').get_parameter_value().double_array_value
        admittance_stiffness       = self.get_parameter('admittance_stiffness').get_parameter_value().double_array_value
        self.maximum_velocity      = self.get_parameter('maximum_velocity').get_parameter_value().double_array_value
        self.maximum_acceleration  = self.get_parameter('maximum_acceleration').get_parameter_value().double_array_value
        use_feedback_velocity      = self.get_parameter('use_feedback_velocity').get_parameter_value().bool_value
        self.complete_debug        = self.get_parameter('complete_debug').get_parameter_value().bool_value
        self.debug                 = self.get_parameter('debug').get_parameter_value().bool_value
        robot                      = self.get_parameter('robot').get_parameter_value().string_value

        # Print Parameters
        print(colored('\nPFL Controller Parameters:', 'yellow'), '\n')
        print(colored('    human_mass:', 'green'),            f'\t\t{self.human_mass}')
        print(colored('    robot_mass:', 'green'),            f'\t\t{self.robot_mass}')
        print(colored('    admittance_mass:', 'green'),       f'\t\t{admittance_mass}')
        print(colored('    admittance_damping:', 'green'),    f'\t{admittance_damping}')
        print(colored('    admittance_stiffness:', 'green'),  f'\t{admittance_stiffness}')
        print(colored('    maximum_velocity:', 'green'),      f'\t\t{self.maximum_velocity}')
        print(colored('    maximum_acceleration:', 'green'),  f'\t{self.maximum_acceleration}')
        print(colored('    use_feedback_velocity:', 'green'), f'\t{use_feedback_velocity}')
        print(colored('    complete_debug:', 'green'),        f'\t\t{self.complete_debug}')
        print(colored('    debug:', 'green'),                 f'\t\t\t{self.debug}')
        print(colored('    robot:', 'green'),                 f'\t\t\t"{robot}"\n')

        # Publishers
        self.joint_velocity_publisher = self.create_publisher(Float64MultiArray, '/ur_rtde/controllers/joint_velocity_controller/command', 1)

        # Subscribers
        self.joint_state_subscriber    = self.create_subscription(JointState, '/joint_states',                self.jointStatesCallback, 1)
        self.ft_sensor_subscriber      = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor',           self.FTSensorCallback, 1)
        self.cartesian_goal_subscriber = self.create_subscription(Pose,       '/handover/cartesian_goal',     self.cartesianGoalCallback, 1)
        self.trajectory_execution_sub  = self.create_subscription(Bool,       '/ur_rtde/trajectory_executed', self.trajectoryExecutionCallback, 1)

        # Initialize Robot and Toolbox Classes
        self.move_robot = UR_RTDE_Move()
        # self.robot_toolbox = UR_Toolbox(robot, self.complete_debug, self.debug)
        self.robot_toolbox = UR_Toolbox(robot, True, True)

        # Initialize Admittance Controller
        self.admittance_controller = AdmittanceController(
            self.robot_toolbox, self.rate,
            M = admittance_mass * np.eye(6),
            D = admittance_damping * np.eye(6),
            K = admittance_stiffness * np.eye(6),
            use_feedback_velocity = use_feedback_velocity,
            complete_debug = self.complete_debug, debug = self.debug
        )

        # Initialize PFL Controller
        self.pfl_controller = PowerForceLimitingController()

        self.test()

    def test(self):

        """ Test Function """

        # a = Pose()
        # a.position.x, a.position.y, a.position.z = 0.2, 0.3, 0.4
        # a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z = 1.0, 0.0, 0.0, 0.0
        # a = self.robot_toolbox.pose2matrix(a)

        a = [0, -pi/2, pi/4, -pi/4, -pi/2, 0]
        b = [0, -pi/2, pi/2, -pi/2, -pi/2, 0]
        # a = [0.6857, -1.703, 2.607, 2.238, -2.256, 0]
        print(a, '\n')
        # self.robot_toolbox.plot(a)
        # self.robot_toolbox.plot(b)

        a = self.robot_toolbox.ForwardKinematic(a)
        print(a)
        # print(b)

        # a = self.robot_toolbox.matrix2pose(self.robot_toolbox.ForwardKinematic([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # b = self.robot_toolbox.matrix2pose(self.robot_toolbox.ForwardKinematic([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        # print(a)
        # print(b)

        print(self.robot_toolbox.InverseKinematic(a))
        # print(self.robot_toolbox.InverseKinematic(b))

        # exit()

        trajectory = self.robot_toolbox.plan_trajectory(a, b, 20, 2000)
        print (trajectory, '\n\n')
        print (colored('Trajectory Positions:', 'green'),     f'\n\n {trajectory.q}\n')
        print (colored('Trajectory Velocities:', 'green'),    f'\n\n {trajectory.qd}\n')
        print (colored('Trajectory Accelerations:', 'green'), f'\n\n {trajectory.qdd}\n')

        # Convert Joint Trajectory to Cartesian Trajectory
        cartesian_trajectory = self.robot_toolbox.joint2cartesianTrajectory(trajectory)

        exit()

        for pos, vel, acc in zip (trajectory.q, trajectory.qd, trajectory.qdd):
            self.compute_admittance_velocity(pos, vel, acc)
            self.rate.sleep()

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

    def FTSensorCallback(self, data:Wrench):

        """ FT Sensor Callback """

        # Get FT Sensor Data
        self.ft_sensor_data = data

    def cartesianGoalCallback(self, data:Pose):

        """ Cartesian Goal Callback """

        # Get Cartesian Goal
        self.handover_cartesian_goal = data
        self.goal_received = True

    def trajectoryExecutionCallback(self, msg:Bool):

        """ Trajectory Execution Callback """

        # Set Trajectory Execution Flags
        self.trajectory_executed = msg.data

    def publishRobotVelocity(self, velocity:List[float]):

        """ Publish Robot Velocity """

        assert len(velocity) == 6, 'Velocity Vector Must Have 6 Elements'

        # ROS Message Creation
        msg = Float64MultiArray(data=velocity)
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].size = len(velocity)
        msg.layout.dim[0].stride = 1
        msg.layout.dim[0].label = 'velocity'

        # Publish Message
        if rclpy.ok(): self.joint_velocity_publisher.publish(msg)

    def is_goal_reached(self, goal:Pose, joint_states:JointState):

        """ Check if Goal is Reached """

        # Joint Goal to Pose
        joint_goal = self.move_robot.IK(goal)

        # Check if Goal is Reached
        if (joint_goal and joint_states):

            # If Goal is Reached -> Return True
            if (all(abs(joint_states.position[i] - joint_goal[i]) < 0.01 for i in range(len(joint_states)))): return True

        return False

    def spinner(self):

        """ Main Spinner """

        # If Goal Received -> Plan Trajectory
        if self.goal_received:

            trajectory = self.robot_toolbox.plan_trajectory(self.joint_states.position, self.robot_toolbox.InverseKinematic(self.handover_cartesian_goal), 20, 2000)
            cartesian_trajectory, i = self.robot_toolbox.joint2cartesianTrajectory(trajectory), 0

        # FIX: If Trajectory Executed -> Admittance Controller with Target = Trajectory Point until Handover is Completed
        # while (rclpy.ok() and self.goal_received and not self.handover_completed):

        # While Goal Received but Not Reached
        # while (rclpy.ok() and self.goal_received and not self.is_goal_reached(self.handover_cartesian_goal, self.joint_states)):

        # While Goal Received
        while (rclpy.ok() and self.goal_received):

            # Get Next Cartesian Goal
            cartesian_goal = cartesian_trajectory[0][i], cartesian_trajectory[1][i], cartesian_trajectory[2][i]

            # Compute Joint Velocity
            joint_velocity = self.admittance_controller.compute_admittance_velocity(self.joint_states, cartesian_goal)

            # Publish Joint Velocity
            self.publishRobotVelocity(joint_velocity)

            # Increment Counter only if i < last trajectory point
            if i < cartesian_trajectory[0].shape[0] - 1: i += 1

        self.goal_received = False

if __name__ == '__main__':

    # ROS Initialization
    rclpy.init()

    # Initialize Class
    handover_controller = Handover_Controller('handover_controller', 500)

    # Register Signal Handler (CTRL+C)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main Spinner Function
    while rclpy.ok():

        handover_controller.spinner()

    # Delete Node before Shutdown ROS
    if handover_controller: del handover_controller
