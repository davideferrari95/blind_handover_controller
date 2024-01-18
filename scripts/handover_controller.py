#! /usr/bin/env python3

import numpy as np
from math import pi
from typing import List, Any
from termcolor import colored
import threading, signal, time

# Import ROS2 Libraries
import rclpy
from rclpy.node import Node, Parameter

# Import ROS Messages, Services, Actions
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench

# Import Robot and UR_RTDE Move Classes
from utils.move_robot import UR_RTDE_Move
from utils.robot_toolbox import UR_Toolbox

# Import Admittance and PFL Controllers Classes
from admittance import AdmittanceController
from power_force_limiting import PowerForceLimitingController

def signal_handler(sig, frame):

    """ UR Stop Signal Handler """

    time.sleep(1)

    # Initialize `rclpy` if not initialized
    if not rclpy.ok(): rclpy.init()

    # Create ROS2 Node + Publisher
    ur_stop = rclpy.create_node('ur_stop_node', enable_rosout=False)
    joint_group_vel_controller_publisher = ur_stop.create_publisher(Float64MultiArray, '/ur_rtde/controllers/joint_velocity_controller/command', 1)
    time.sleep(1)

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
    time.sleep(1)
    rclpy.try_shutdown()

class Handover_Controller(Node):

    """ Handover Controller Class """

    # Initialize Subscriber Variables
    trajectory_executed, goal_received = False, False
    handover_cartesian_goal = Pose()
    joint_states, ft_sensor_data = JointState(), Wrench()
    desired_joint_velocity = [0.0] * 6

    # FIX: Remove Pause OS Variable
    pause_os = False

    def __init__(self, node_name, ros_rate):

        # Node Initialization
        super().__init__(node_name)

        # ROS2 Rate
        self.ros_rate = ros_rate
        self.rate = self.create_rate(ros_rate)

        # Spin in a separate thread - for ROS2 Rate
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self, ), daemon=True)
        self.spin_thread.start()

        # Declare Parameters
        self.declare_parameters('', [('admittance_mass', [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]), ('admittance_damping', [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]), ('admittance_stiffness', [1.00, 1.00, 1.00, 1.00, 1.00, 1.00])])
        self.declare_parameters('', [('maximum_velocity', [1.05, 1.05, 1.57, 1.57, 1.57, 1.57]), ('maximum_acceleration', [0.57, 0.57, 0.57, 0.57, 0.57, 0.57])])
        self.declare_parameters('', [('admittance_weight', 0.1), ('force_dead_zone', 4.0), ('torque_dead_zone', 1.0), ('human_radius', 0.1)])
        self.declare_parameters('', [('use_feedback_velocity', False), ('sim', False), ('complete_debug', False), ('debug', False)])

        # Declare Robot Parameters
        self.declare_parameters('', [('robot', 'ur10e'), ('payload', 12.5), ('reach', 1.30), ('tcp_speed', 1.0)]) # kg, m, m/s
        self.declare_parameters('', [('stopping_time', 0.17), ('stopping_distance', 0.25), ('position_repeatability', 0.05)]) # s, m, mm
        self.declare_parameters('', [('maximum_power', 615), ('operating_power', 350), ('operating_temperature', [0, 50])]) # W, W, °C
        self.declare_parameters('', [('ft_range', [100.0, 10.0]), ('ft_precision', [5.0, 0.2]), ('ft_accuracy', [5.5, 0.5]), ('tool', [0.0, 0.0, 0.0, 0.0, 0.0, pi])]) # N, nm
        self.declare_parameters('', [('a', [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0]), ('d', [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655]), ('alpha', [pi/2, 0.0, 0.0, pi/2, -pi/2, 0.0]), ('theta', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])    # DH Parameters
        self.declare_parameters('', [('mass', [7.369, 13.051, 3.989, 2.1, 1.98, 0.615]), ('center_of_mass', [0.021, 0.000, 0.027, 0.38, 0.000, 0.158, 0.24, 0.000, 0.068, 0.000, 0.007, 0.018, 0.000, 0.007, 0.018, 0.0, 0.0, -0.026])]) # Dynamic Parameters
        self.declare_parameters('', [('q_limits', [2*pi, 2*pi, 2*pi, 2*pi, 2*pi, 2*pi]), ('q_dot_limits', [2*pi/3, 2*pi/3, pi, pi, pi, pi]), ('q_ddot_limits', [0.20, 0.20, 0.20, 0.20, 0.20, 0.20])]) # Joint, Speed, Acceleration Limits

        # Read Parameters
        self.force_dead_zone, self.torque_dead_zone = self.get_parameter_value('force_dead_zone'), self.get_parameter_value('torque_dead_zone')
        self.sim, use_feedback_velocity = self.get_parameter_value('sim'), self.get_parameter_value('use_feedback_velocity')
        self.complete_debug, self.debug = self.get_parameter_value('complete_debug'), self.get_parameter_value('debug')
        human_radius = self.get_parameter_value('human_radius')

        # Read Robot Parameters
        robot_parameters = {param_name : self.get_parameter_value(param_name) for param_name in
                            ['robot', 'tool', 'payload', 'reach', 'tcp_speed', 'stopping_time', 'stopping_distance', 'position_repeatability',
                            'maximum_power', 'operating_power', 'operating_temperature', 'ft_range', 'ft_precision', 'ft_accuracy',
                            'a', 'd', 'alpha', 'theta', 'mass', 'center_of_mass', 'q_limits', 'q_dot_limits', 'q_ddot_limits']}

        # Print Parameters
        print(colored('\nPFL Controller Parameters:', 'yellow'), '\n')
        print(colored('    use_feedback_velocity:', 'green'), f'\t{use_feedback_velocity}')
        print(colored('    complete_debug:', 'green'),        f'\t\t{self.complete_debug}')
        print(colored('    debug:', 'green'),                 f'\t\t\t{self.debug}')
        print(colored('    sim:', 'green'),                   f'\t\t\t{self.sim}')
        print(colored('    human_radius:', 'green'),          f'\t\t{human_radius}')
        print(colored('    robot:', 'green'),                 f'\t\t\t"{robot_parameters["robot"]}"\n')

        # Publishers
        self.joint_velocity_publisher   = self.create_publisher(Float64MultiArray, '/ur_rtde/controllers/joint_velocity_controller/command', 1)
        if self.sim: self.joint_simulation_publisher = self.create_publisher(JointState, '/joint_states', 1)

        # Subscribers
        self.joint_state_subscriber    = self.create_subscription(JointState, '/joint_states',                self.jointStatesCallback, 1)
        self.ft_sensor_subscriber      = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor',           self.FTSensorCallback, 1)
        self.cartesian_goal_subscriber = self.create_subscription(Pose,       '/handover/cartesian_goal',     self.cartesianGoalCallback, 1)
        self.trajectory_execution_sub  = self.create_subscription(Bool,       '/ur_rtde/trajectory_executed', self.trajectoryExecutionCallback, 1)

        # Initialize Robot and Toolbox Classes
        self.move_robot = UR_RTDE_Move()
        self.robot_toolbox = UR_Toolbox(robot_parameters, self.complete_debug, self.debug)

        # Initialize Admittance Controller
        self.admittance_controller = AdmittanceController(
            self.robot_toolbox, self.rate, use_feedback_velocity = False if self.sim else use_feedback_velocity,
            M = self.get_parameter_value('admittance_mass') * np.eye(6), D = self.get_parameter_value('admittance_damping') * np.eye(6),
            K = self.get_parameter_value('admittance_stiffness') * np.eye(6), admittance_weight = self.get_parameter_value('admittance_weight'),
            max_vel = self.get_parameter_value('maximum_velocity'), max_acc = self.get_parameter_value('maximum_acceleration'),
            complete_debug = self.complete_debug, debug = self.debug
        )

        # Initialize PFL Controller
        # self.pfl_controller = PowerForceLimitingController(self.rate, self.robot_toolbox, robot_parameters, human_radius, self.complete_debug, self.debug)
        self.pfl_controller = PowerForceLimitingController(self.rate, self.robot_toolbox, robot_parameters, human_radius, True, True)

        # FIX: Remove Test Function
        if self.sim: self.test()
        else: self.test_real()

    def test(self):

        """ Test Function """

        from rclpy.parameter import Parameter

        self.declare_parameter('pause_execution', True)

        # a = Pose()
        # a.position.x, a.position.y, a.position.z = 0.2, 0.3, 0.4
        # a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z = 1.0, 0.0, 0.0, 0.0
        # a = self.robot_toolbox.pose2matrix(a)

        # a = [0.0, -pi/2, pi/2, -pi/2, -pi/2, 0.0]
        # b = [0.0, -pi/2, pi/4, -pi/4, -pi/2, 0.0]

        a = [-1.5395906607257288, -0.8890350025943299, -1.3985795974731445, -0.7067992252162476, 1.716660737991333, -1.0363872686969202]
        b = [-1.5718229452716272, -1.7333761654295863, -1.567663550376892, -1.551903573130705, 1.7166997194290161, -1.0364607016192835]

        print(a, '\n')
        # self.robot_toolbox.plot(a)
        # self.robot_toolbox.plot(b)

        # FIX: Initialize Joint States
        self.joint_states.position = a
        self.joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Initialize Simulation Joint States
        if self.sim: time.sleep(1); self.publishSimulationJointStates(b)
        if self.sim: time.sleep(3); self.publishSimulationJointStates(a)

        a = self.robot_toolbox.ForwardKinematic(a)
        print(a)
        print(self.robot_toolbox.ForwardKinematic(b))

        # a = self.robot_toolbox.matrix2pose(self.robot_toolbox.ForwardKinematic([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # b = self.robot_toolbox.matrix2pose(self.robot_toolbox.ForwardKinematic([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        # print(a)
        # print(b)

        a = self.robot_toolbox.matrix2array(a)
        print(a, '\n')

        a = self.robot_toolbox.array2matrix(a)
        print(a)

        print(self.robot_toolbox.InverseKinematic(a))
        # print(self.robot_toolbox.InverseKinematic(b))

        while self.pause_os and rclpy.ok() and self.get_parameter('pause_execution').get_parameter_value().bool_value:
            self.get_logger().info('Pausing execution...', once=True)
        self.set_parameters([Parameter('pause_execution', Parameter.Type.BOOL, True)])

        trajectory = self.robot_toolbox.plan_trajectory(a, b, 10, self.ros_rate)
        print (trajectory, '\n\n')
        print (colored('Trajectory Positions:', 'green'),     f'\n\n {trajectory.q}\n')
        print (colored('Trajectory Velocities:', 'green'),    f'\n\n {trajectory.qd}\n')
        print (colored('Trajectory Accelerations:', 'green'), f'\n\n {trajectory.qdd}\n')

        # Convert Joint Trajectory to Cartesian Trajectory
        cartesian_trajectory = self.robot_toolbox.joint2cartesianTrajectory(trajectory)

        # Publish Joint Trajectory
        # for pos in trajectory.q: self.publishSimulationJointStates(pos); self.rate.sleep()

        while self.pause_os and rclpy.ok() and self.get_parameter('pause_execution').get_parameter_value().bool_value:
            self.get_logger().info('Pausing execution...', once=True)
        self.set_parameters([Parameter('pause_execution', Parameter.Type.BOOL, True)])

        for x_des, x_des_dot, x_des_ddot in zip(cartesian_trajectory[0], cartesian_trajectory[1], cartesian_trajectory[2]):

            # start = time.time()

            # Break if ROS is not Ok
            if not rclpy.ok(): break

            # Compute Admittance Velocity
            desired_joint_velocity = self.admittance_controller.compute_admittance_velocity(self.joint_states, Wrench(), x_des, x_des_dot, x_des_ddot)

            # Compute PFL Velocity
            desired_joint_velocity = self.pfl_controller.compute_pfl_velocity(desired_joint_velocity,  self.joint_states)

            # FIX: Compute - Publish Simulation Joint States Positions
            if self.sim: self.joint_states.position = [self.joint_states.position[i] + desired_joint_velocity[i] * self.rate._timer.timer_period_ns * 1e-9 for i in range(len(self.joint_states.position))]
            if self.sim: self.publishSimulationJointStates(self.joint_states.position)

            # Sleep to ROS Rate
            self.rate.sleep()

            # print(time.time() - start)

        print(colored('\nAdmittance Controller Completed\n', 'green'))

        exit()

    def test_real(self):

        """ Test Function """

        # UR5e
        # goal = [-1.57, -1.75, -1.57, -1.57, 1.75, -1.0]
        # goal = [-1.0, -1.50, -1.0, -1.0, 1.75, -1.0]

        # UR10e
        goal = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
        # goal = [0.45, -2.30, 2.28, -2.13, -1.67, 0.39]

        time.sleep(1)
        print(f'JointState Position: {self.joint_states.position}\n')

        # trajectory = self.robot_toolbox.plan_trajectory(self.joint_states.position, goal, 10, self.ros_rate)
        trajectory = self.robot_toolbox.plan_trajectory([0.15, -1.71, 2.28, -2.13, -1.67, 0.39], goal, 10, self.ros_rate)

        print (trajectory, '\n\n')
        print (colored('Trajectory Positions:', 'green'),     f'\n\n {trajectory.q}\n')
        print (colored('Trajectory Velocities:', 'green'),    f'\n\n {trajectory.qd}\n')
        print (colored('Trajectory Accelerations:', 'green'), f'\n\n {trajectory.qdd}\n')

        # Convert Joint Trajectory to Cartesian Trajectory
        start = time.time()
        cartesian_trajectory, i = self.robot_toolbox.joint2cartesianTrajectory(trajectory), 0
        print(colored(f'Trajectory Converted: ', 'green'), f'{time.time() - start}\n')

        while (rclpy.ok()):

            #     start = time.time()

            # Get Next Cartesian Goal | Increment Counter only if i < last trajectory point
            cartesian_goal = cartesian_trajectory[0][i], cartesian_trajectory[1][i], cartesian_trajectory[2][i]
            if i < cartesian_trajectory[1].shape[0] - 1: i += 1

            # Compute Admittance Velocity
            self.desired_joint_velocity = self.admittance_controller.compute_admittance_velocity(self.joint_states, self.ft_sensor_data, self.desired_joint_velocity, *cartesian_goal)
            # self.desired_joint_velocity = self.admittance_controller.compute_admittance_velocity(self.joint_states, Wrench(), self.desired_joint_velocity, *cartesian_goal)

            # Compute PFL Velocity
            # self.desired_joint_velocity = self.pfl_controller.compute_pfl_velocity(self.desired_joint_velocity,  self.joint_states)

            # Publish Joint Velocity
            self.publishRobotVelocity(self.desired_joint_velocity)

            # Sleep to ROS Rate
            self.rate.sleep()

            # print(time.time() - start)

        print(colored('\nAdmittance Controller Completed\n', 'green'))

        # Stop Robot
        self.publishRobotVelocity([0.0] * 6)

        exit()

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

    def FTSensorCallback(self, data:Wrench):

        """ FT Sensor Callback """

        # Get FT Sensor Data - Apply Dead Zone
        self.ft_sensor_data.force.x = data.force.x if abs(data.force.x) > abs(self.force_dead_zone) else 0.0
        self.ft_sensor_data.force.y = data.force.y if abs(data.force.y) > abs(self.force_dead_zone) else 0.0
        self.ft_sensor_data.force.z = data.force.z if abs(data.force.z) > abs(self.force_dead_zone) else 0.0
        self.ft_sensor_data.torque.x = data.torque.x if abs(data.torque.x) > abs(self.torque_dead_zone) else 0.0
        self.ft_sensor_data.torque.y = data.torque.y if abs(data.torque.y) > abs(self.torque_dead_zone) else 0.0
        self.ft_sensor_data.torque.z = data.torque.z if abs(data.torque.z) > abs(self.torque_dead_zone) else 0.0

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

    def publishSimulationJointStates(self, pos:List[float], vel:List[float]=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], eff:List[float]=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):

        """ Publish Simulation Joint States """

        # Type Assertions
        assert len(pos) == 6 and len(vel) == 6 and len(eff) == 6, 'Position, Velocity and Effort Vectors Must Have 6 Elements'
        if type(pos) is np.ndarray: pos = pos.tolist()

        # ROS Message Creation
        joint = JointState()
        joint.header.stamp = self.get_clock().now().to_msg()
        joint.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        joint.position, joint.velocity, joint.effort = pos, vel, eff

        # Publish Joint States
        if rclpy.ok(): self.joint_simulation_publisher.publish(joint)

    def is_goal_reached(self, goal:Pose, joint_states:JointState):

        """ Check if Goal is Reached """

        # Joint Goal to Pose
        joint_goal = self.move_robot.IK(goal)

        # Check if Goal is Reached
        if (joint_goal and joint_states):

            # If Goal is Reached -> Return True
            if (all(abs(joint_states.position[i] - joint_goal[i]) < 0.01 for i in range(len(joint_states)))): return True

        return False

    def get_parameter_value(self, parameter_name:str) -> Any:

        """ Get Parameter Value """

        # Get Parameter Object
        param = self.get_parameter(parameter_name)

        # Return Parameter Value - Based on Type
        if   param.type_ == Parameter.Type.NOT_SET:       return None
        elif param.type_ == Parameter.Type.BOOL:          return self.get_parameter(parameter_name).get_parameter_value().bool_value
        elif param.type_ == Parameter.Type.INTEGER:       return self.get_parameter(parameter_name).get_parameter_value().integer_value
        elif param.type_ == Parameter.Type.DOUBLE:        return self.get_parameter(parameter_name).get_parameter_value().double_value
        elif param.type_ == Parameter.Type.STRING:        return self.get_parameter(parameter_name).get_parameter_value().string_value
        elif param.type_ == Parameter.Type.BYTE_ARRAY:    return self.get_parameter(parameter_name).get_parameter_value().byte_array_value
        elif param.type_ == Parameter.Type.BOOL_ARRAY:    return self.get_parameter(parameter_name).get_parameter_value().bool_array_value
        elif param.type_ == Parameter.Type.INTEGER_ARRAY: return self.get_parameter(parameter_name).get_parameter_value().integer_array_value
        elif param.type_ == Parameter.Type.DOUBLE_ARRAY:  return self.get_parameter(parameter_name).get_parameter_value().double_array_value
        elif param.type_ == Parameter.Type.STRING_ARRAY:  return self.get_parameter(parameter_name).get_parameter_value().string_array_value
        else: raise ValueError(f'Parameter Type Not Supported: {param.type_}')

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

            # Get Next Cartesian Goal | Increment Counter only if i < last trajectory point
            cartesian_goal = cartesian_trajectory[0][i], cartesian_trajectory[1][i], cartesian_trajectory[2][i]
            if i < cartesian_trajectory[1].shape[0] - 1: i += 1

            # Compute Admittance Velocity
            self.desired_joint_velocity = self.admittance_controller.compute_admittance_velocity(self.joint_states, self.ft_sensor_data, self.desired_joint_velocity, *cartesian_goal)

            # Compute PFL Velocity
            self.desired_joint_velocity = self.pfl_controller.compute_pfl_velocity(self.desired_joint_velocity,  self.joint_states)

            # Publish Joint Velocity
            self.publishRobotVelocity(self.desired_joint_velocity)

            # Sleep to ROS Rate
            self.rate.sleep()

        # Stop Robot
        self.publishRobotVelocity([0.0] * 6)
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
