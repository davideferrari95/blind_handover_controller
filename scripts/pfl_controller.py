#! /usr/bin/env python3

# Import ROS2 Libraries
import rclpy
from rclpy.node import Node
from typing import List

# Import ROS Messages, Services, Actions
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench

# Import Utils Functions
from move_robot import UR10e_RTDE_Move

class PFL_Controller(Node):

    """ PFL Controller Class """

    # Initialize Subscriber Variables
    trajectory_executed, goal_received = False, False
    joint_states, ft_sensor_data = JointState(), Wrench()
    handover_cartesian_goal = Pose()

    def __init__(self, node_name, ros_rate):

        # Node Initialization
        super().__init__(node_name)

        # ROS2 Rate
        self.create_rate(ros_rate)

        # Initialize Robot
        self.robot = UR10e_RTDE_Move()

        # Declare Parameters
        self.declare_parameter('human_mass',            10.0)
        self.declare_parameter('robot_mass',            10.0)
        self.declare_parameter('admittance_mass',       10.0)
        self.declare_parameter('admittance_damping',    1.0)
        self.declare_parameter('admittance_stiffness',  0.0)

        # Read Parameters
        self.human_mass           = self.get_parameter('human_mass').get_parameter_value().double_value
        self.robot_mass           = self.get_parameter('robot_mass').get_parameter_value().double_value
        self.admittance_mass      = self.get_parameter('admittance_mass').get_parameter_value().double_value
        self.admittance_damping   = self.get_parameter('admittance_damping').get_parameter_value().double_value
        self.admittance_stiffness = self.get_parameter('admittance_stiffness').get_parameter_value().double_value

        self.get_logger().info('PFL Controller Parameters:')
        self.get_logger().info('human_mass:           ' + str(self.human_mass))
        self.get_logger().info('robot_mass:           ' + str(self.robot_mass))
        self.get_logger().info('admittance_mass:      ' + str(self.admittance_mass))
        self.get_logger().info('admittance_damping:   ' + str(self.admittance_damping))
        self.get_logger().info('admittance_stiffness: ' + str(self.admittance_stiffness))

        # Publishers
        self.joint_velocity_publisher = self.create_publisher(Float64MultiArray, '/ur_rtde/controllers/joint_velocity_controller/command', 1)

        # Subscribers
        self.joint_state_subscriber    = self.create_subscription(JointState, '/joint_states',                self.jointStatesCallback, 1)
        self.ft_sensor_subscriber      = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor',           self.FTSensorCallback, 1)
        self.cartesian_goal_subscriber = self.create_subscription(Pose,       '/handover/cartesian_goal',     self.cartesianGoalCallback, 1)
        self.trajectory_execution_sub  = self.create_subscription(Bool,       '/ur_rtde/trajectory_executed', self.trajectoryExecutionCallback, 1)

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
        joint_goal = self.robot.IK(goal)

        # Check if Goal is Reached
        if (joint_goal and joint_states):

            # If Goal is Reached -> Return True
            if (all(abs(joint_states.position[i] - joint_goal[i]) < 0.01 for i in range(len(joint_states)))): return True

        return False

    def compute_admittance_velocity(self, joint_goal:List[float]):

        """ Compute Admittance Cartesian Velocity """

        pass

    def PFL(self, joint_states:JointState, cartesian_velocity:List[float]):
        
        """ PFL Controller """

        return cartesian_velocity

    def cartesian_to_joint_velocity(self, cartesian_velocity:List[float]):

        """ Cartesian to Joint Velocity """

        # Initialize Joint Velocity
        joint_velocity = [0.0] * 6

        # If Cartesian Velocity is Valid
        if cartesian_velocity:

            # Compute Joint Velocity
            joint_velocity = self.robot.JacobianInverse(cartesian_velocity)

        return joint_velocity

    def spinner(self):

        """ Main Spinner """

        # While Goal Received but Not Reached
        while (self.goal_received and not self.is_goal_reached(self.handover_cartesian_goal, self.joint_states)):

            # Compute Joint Velocity
            cartesian_velocity = self.compute_admittance_velocity(self.handover_cartesian_goal)

            # Cartesian Velocity - PFL Controller
            cartesian_velocity = self.PFL(self.joint_states, cartesian_velocity)

            # Publish Joint Velocity
            self.publishRobotVelocity(self.cartesian_to_joint_velocity(cartesian_velocity))

        self.goal_received = False

if __name__ == '__main__':

    # ROS Initialization
    rclpy.init()

    # Initialize Class
    pfl_controller = PFL_Controller('pfl_controller', 1000)

    # Main Spinner Function
    while rclpy.ok():

        pfl_controller.spinner()

    # Delete Node before Shutdown ROS
    if pfl_controller: del pfl_controller
