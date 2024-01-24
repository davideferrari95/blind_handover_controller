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
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Wrench, Vector3

# Import Robot and UR_RTDE Move Classes
from utils.move_robot import UR_RTDE_Move
from utils.robot_toolbox import UR_Toolbox

# Import Admittance and PFL Controllers Classes
from admittance import AdmittanceController
from safety_controller import SafetyController

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

    # Initialize Class Variables
    goal_received, start_admittance = False, False
    joint_states, ft_sensor_data = JointState(), Wrench()

    # Initialize Robot and Human Points Variables
    human_point, robot_base = Vector3(), Pose()
    human_vel, human_timer = Vector3(), time.time()

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
        self.declare_parameters('', [('admittance_weight', 0.1), ('force_dead_zone', 4.0), ('torque_dead_zone', 1.0), ('human_radius', 0.2)])
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

        # Initialize Robot and Toolbox Classes
        self.move_robot = UR_RTDE_Move()
        self.robot_toolbox = UR_Toolbox(robot_parameters, self.complete_debug, self.debug)

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
        self.joint_state_subscriber    = self.create_subscription(JointState,        '/joint_states',            self.jointStatesCallback, 1)
        self.ft_sensor_subscriber      = self.create_subscription(Wrench,            '/ur_rtde/ft_sensor',       self.FTSensorCallback, 1)
        self.cartesian_goal_subscriber = self.create_subscription(Pose,              '/handover/cartesian_goal', self.cartesianGoalCallback, 1)
        self.joint_goal_subscriber     = self.create_subscription(Float64MultiArray, '/handover/joint_goal',     self.jointGoalCallback, 1)

        # PFL Subscribers
        self.human_pose_subscriber = self.create_subscription(PoseStamped, '/vrpn_mocap/right_wrist/pose', self.humanPointCallback, 1)
        self.robot_pose_subscriber = self.create_subscription(PoseStamped, '/vrpn_mocap/UR5/pose', self.robotPointCallback, 1)

        # Service Servers
        self.stop_admittance_server = self.create_service(Trigger, '/handover/stop', self.stopAdmittanceServerCallback)

        # Initialize Admittance Controller
        self.admittance_controller = AdmittanceController(
            self.robot_toolbox, self.rate, use_feedback_velocity = False if self.sim else use_feedback_velocity,
            M = self.get_parameter_value('admittance_mass') * np.eye(6), D = self.get_parameter_value('admittance_damping') * np.eye(6),
            K = self.get_parameter_value('admittance_stiffness') * np.eye(6), admittance_weight = self.get_parameter_value('admittance_weight'),
            max_vel = self.get_parameter_value('maximum_velocity'), max_acc = self.get_parameter_value('maximum_acceleration'),
            complete_debug = self.complete_debug, debug = self.debug
        )

        # Initialize PFL Controller
        self.safety_controller = SafetyController(self.robot_toolbox, robot_parameters, human_radius, self.ros_rate, self.complete_debug, self.debug)
        # self.safety_controller = SafetyController(self.robot_toolbox, robot_parameters, human_radius, self.ros_rate, True, True)

        # Controller Initialized
        print(colored('Handover Controller Initialized\n', 'yellow'))
        time.sleep(1)

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

        # Get Joint Goal from Cartesian Goal
        self.handover_goal = self.robot_toolbox.InverseKinematic(data).q
        self.goal_received, self.start_admittance = True, False

    def jointGoalCallback(self, data:Float64MultiArray):

        """ Joint Goal Callback """

        # Get Joint Handover Goal
        self.handover_goal = data.data
        self.goal_received, self.start_admittance = True, False

    def humanPointCallback(self, msg:PoseStamped):

        """ Human Pose Callback (PH) """

        # Transform Human Pose from World Frame to Robot Base Frame (T2 in T1​​ = T1^-1 ​@ T2)
        human_pos = np.linalg.inv(self.robot_toolbox.pose2matrix(self.robot_base).A) @ self.robot_toolbox.pose2matrix(msg.pose).A
        human_pos = self.robot_toolbox.matrix2pose(human_pos)

        # Compute Human Velocity - Update Human Timer
        # self.human_vel.x, self.human_vel.y, self.human_vel.z = (human_pos.position.x - self.human_point.x) / (time.time() - self.human_timer), \
        #                                                        (human_pos.position.y - self.human_point.y) / (time.time() - self.human_timer), \
        #                                                        (human_pos.position.z - self.human_point.z) / (time.time() - self.human_timer)
        # self.human_timer = time.time()

        # TODO: Check with Human Velocity != 0
        # self.human_vel.x, self.human_vel.y, self.human_vel.z = 0.25, 0.25, 0.25
        self.human_vel.x, self.human_vel.y, self.human_vel.z = 0.0, 0.0, 0.0

        # Update Human Vector3 Message
        self.human_point.x, self.human_point.y, self.human_point.z = human_pos.position.x, human_pos.position.y, human_pos.position.z

    def robotPointCallback(self, msg:PoseStamped):

        """ Robot Pose Callback (UR - Base) """

        # Update Robot Base Message
        self.robot_base = msg.pose

    def stopAdmittanceServerCallback(self, req:Trigger.Request, res:Trigger.Response):

        """ Stop Admittance Server Callback """

        print(colored('Admittance Controller Completed\n', 'yellow'))
        self.goal_received, self.start_admittance = False, False

        # Stop Robot
        self.old_joint_velocity = [0.0] * 6
        self.publishRobotVelocity(self.old_joint_velocity)

        # Response Filling
        res.success = True
        return res

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

    def plan_trajectory(self, handover_goal:List[float]):

        """ Plan Trajectory """

        # If Goal Received -> Plan Trajectory
        print(colored(f'Goal Received: ', 'yellow'), f'[{handover_goal}] - Planning Trajectory')
        trajectory = self.robot_toolbox.plan_trajectory(self.joint_states.position, handover_goal, 10, self.ros_rate)

        # Convert Trajectory to Spline
        self.spline_trajectory = self.robot_toolbox.trajectory2spline(trajectory)

        # Initialize Admittance Controller Variables
        self.old_joint_velocity = np.array(self.joint_states.velocity)
        self.scaling_factor, self.current_time = 0.0, 0.0
        self.goal_received, self.start_admittance = False, True

        # Start Admittance Controller
        print(colored('Trajectory Planned - Starting Admittance Controller', 'green'))

    def spinner(self):

        """ Main Spinner """

        # If Goal Received -> Plan Trajectory
        if self.goal_received: self.plan_trajectory(self.handover_goal)

        # While Goal Received
        while (rclpy.ok() and self.start_admittance):

            # Get Next Cartesian Goal | Increment Counter only if i < last trajectory point
            cartesian_goal, self.current_time = self.robot_toolbox.get_cartesian_goal(self.spline_trajectory, self.current_time, self.scaling_factor, self.ros_rate)

            # Compute Admittance Velocity
            desired_joint_velocity = self.admittance_controller.compute_admittance_velocity(self.joint_states, self.ft_sensor_data, self.old_joint_velocity, *cartesian_goal)

            # Compute Safety Scaling Factor (Safety: SSM | PFL)
            self.old_joint_velocity, self.scaling_factor = self.safety_controller.compute_safety(desired_joint_velocity, self.old_joint_velocity, self.joint_states, self.human_point, self.human_vel)

            # Publish Joint Velocity
            self.publishRobotVelocity(self.old_joint_velocity)

            # Sleep to ROS Rate
            self.rate.sleep()

if __name__ == '__main__':

    # ROS Initialization
    rclpy.init()

    # Initialize Class
    handover_controller = Handover_Controller('handover_controller', 500)

    # Register Signal Handler (CTRL+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Main Spinner Function
    while rclpy.ok():

        handover_controller.spinner()

    # Delete Node before Shutdown ROS
    if handover_controller: del handover_controller
