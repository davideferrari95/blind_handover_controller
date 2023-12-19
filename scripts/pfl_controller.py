#! /usr/bin/env python3
import numpy as np
from termcolor import colored
import threading, signal, time
from scipy.spatial.transform import Rotation

# Import ROS2 Libraries
import rclpy
from rclpy.node import Node
from typing import List

# Import ROS Messages, Services, Actions
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench

# Import Utils Functions
from move_robot import UR_RTDE_Move, Trajectory

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

class PFL_Controller(Node):

    """ PFL Controller Class """

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
        self.use_feedback_velocity = self.get_parameter('use_feedback_velocity').get_parameter_value().bool_value
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
        print(colored('    use_feedback_velocity:', 'green'), f'\t{self.use_feedback_velocity}')
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

        # Initialize Robot
        self.robot = UR_RTDE_Move(robot_model=robot)

        # Create Mass, Damping and Stiffness Matrices
        self.M, self.D, self.K = admittance_mass * np.eye(6), admittance_damping * np.eye(6), admittance_stiffness * np.eye(6)

        a = Pose()
        a.position.x, a.position.y, a.position.z = 1.0, 1.5, 1.0
        a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z = 0.1, 0.2, 0.3, 0.4

        trajectory = self.robot.plan_trajectory(self.joint_states.position, [2.0, 3.0, 1.0, 1.0, 1.0, 0.5], 20, 2000)
        print (trajectory, '\n\n')
        print (colored('Trajectory Positions:', 'green'),     f'\n\n {trajectory.q}\n')
        print (colored('Trajectory Velocities:', 'green'),    f'\n\n {trajectory.qd}\n')
        print (colored('Trajectory Accelerations:', 'green'), f'\n\n {trajectory.qdd}\n')

        trajectory = self.robot.plan_cartesian_trajectory(Pose(), a, 20, 2000)
        print (trajectory, '\n\n')
        print (colored('Cartesian Trajectory Positions:', 'green'),     f'\n\n {trajectory.q}\n')
        print (colored('Cartesian Trajectory Velocities:', 'green'),    f'\n\n {trajectory.qd}\n')
        print (colored('Cartesian Trajectory Accelerations:', 'green'), f'\n\n {trajectory.qdd}\n')

        for pos, vel, acc in zip (trajectory.q, trajectory.qd, trajectory.qdd):
            self.compute_admittance_velocity(pos, vel, acc)
            self.rate.sleep()

        exit()

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


#   P_H_      = tf2::Vector3(0, 0, 1);
#   P_R_      = tf2::Vector3(0, 0, 0);
# https://github.com/ARSControl/dynamic_planner/blob/master/trajectory_scaling/src/trajectory_scaling.cpp
# void TrajectoryScaling::humanPointCallback(
#   const geometry_msgs::PointStamped::ConstPtr& hp)
# {
#   geometry_msgs::PointStamped human_point = *hp;
#   tf2::doTransform(human_point, human_point, transform_);
#   //  buffer_.transform(*hp, "base_link");
#   P_H_ = tf2::Vector3(human_point.point.x, human_point.point.y, human_point.point.z);
#   // std::cout << "Human Point: " << P_H_.x() << " " << P_H_.y() << " " << P_H_.z() << std::endl;
# }

# void TrajectoryScaling::robotPointCallback(
#   const geometry_msgs::PointStamped::ConstPtr& rp)
# {
#   geometry_msgs::PointStamped robot_point = *rp;
#   // tf2::doTransform(robot_point, robot_point, transform_);
#   // buffer_.transform(*rp, "base_link");
#   P_R_ = tf2::Vector3(robot_point.point.x, robot_point.point.y, robot_point.point.z);
#   // std::cout << "Robot Point: " << P_R_.x() << " " << P_R_.y() << " " << P_R_.z() << std::endl;
# }

# tf2::Vector3 TrajectoryScaling::computeVersor()
# {
#   // std::cout << (P_H_-P_R_).x() << " " << (P_H_-P_R_).y() << " " << (P_H_-P_R_).z() << std::endl;
#   return (P_H_ - P_R_) / (P_R_.distance(P_H_));
# }


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

    def pose2matrix(self, pose:Pose) -> np.ndarray:

        """ Convert Pose to Numpy Transformation Matrix """

        # Convert Pose to Rotation Matrix and Translation Vector -> Create Transformation Matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
        transformation_matrix[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])

        return transformation_matrix

    def compute_position_error(self, x_des:np.ndarray, x_act:np.ndarray) -> np.ndarray:

        """ Compute Position Error: x_des - x_act """

        assert x_des.shape == (4, 4), 'Desired Pose Must be a 4x4 Matrix'
        assert x_act.shape == (4, 4), 'Actual Pose Must be a 4x4 Matrix'

        if self.complete_debug: print(f'x_des: {type(x_des)} \n {x_des}\n')
        if self.complete_debug: print(f'x_act: {type(x_act)} \n {x_act}\n')

        # Compute Translation Error
        position_error = np.zeros((6,))
        position_error[:3] = x_des[:3, 3] - x_act[:3, 3]

        # Compute Orientation Error
        orientation_error:Rotation = Rotation.from_matrix(x_des[:3, :3]).inv() * Rotation.from_matrix(x_act[:3, :3])
        position_error[3:] = orientation_error.as_rotvec()

        return position_error.transpose()

    def compute_admittance_velocity(self, cartesian_goal:Pose):

        """ Compute Admittance Cartesian Velocity """

        # FIX: cartesian_goal = Subscriber Pose()
        a = Pose()
        a.position.x, a.position.y, a.position.z = 0.2, 0.3, 0.4
        a.orientation.w, a.orientation.x, a.orientation.y, a.orientation.z = 0.1, 0.3, 0.4, 0.2
        if self.debug or self.complete_debug: print(colored('-'*100, 'yellow'), '\n')

        # FIX: get FK from Subscriber JointState()
        # Compute Position Error (x_des - x_act)
        # position_error = self.compute_position_error(self.pose2numpy(cartesian_goal), self.pose2numpy(self.robot.FK(self.joint_states.position)))
        position_error = self.compute_position_error(self.pose2matrix(cartesian_goal), self.pose2matrix(a))
        if self.complete_debug: print(f'position_error: {type(position_error)} | {position_error.shape} \n {position_error}\n')
        elif self.debug: print(f'position_error: {position_error}\n')

        # Compute Manipulator Jacobian
        J = self.robot.Jacobian(self.joint_states.position)
        if self.complete_debug: print(f'J: {type(J)} | {J.shape} \n {J}\n')

        # Compute Cartesian Velocity
        x_dot: np.ndarray = np.matmul(J, self.joint_states.velocity) if self.use_feedback_velocity else self.x_dot_last_cycle
        if self.complete_debug: print(f'x_dot: {type(x_dot)} | {x_dot.shape} \n {x_dot}\n')
        elif self.debug: print(f'x_dot:     {x_dot}')

        # Compute Acceleration with Admittance (Mx'' + Dx' + Kx = 0) (x = x_des - x_act) (x'_des, X''_des = 0) -> x_act'' = -M^-1 * (- Dx_act' + K(x_des - x_act))
        x_dot_dot: np.ndarray = np.matmul(np.linalg.inv(self.M), - np.matmul(self.D, x_dot) + np.matmul(self.K, position_error))
        if self.complete_debug: print(f'M: {type(self.M)} \n {self.M}\n\n', f'D: {type(self.D)} \n {self.D}\n\n', f'K: {type(self.K)} \n {self.K}\n')
        if self.complete_debug: print(f'x_dot_dot: {type(x_dot_dot)} | {x_dot_dot.shape} \n {x_dot_dot}\n')
        elif self.debug: print(f'x_dot_dot: {x_dot_dot}')

        # Integrate for Velocity Based Interface
        x_dot = x_dot + x_dot_dot * self.rate._timer.timer_period_ns * 1e-9
        if self.complete_debug: print(f'self.ros_rate: {self.rate._timer.timer_period_ns * 1e-9} | 1/self.ros_rate: {1/(self.rate._timer.timer_period_ns * 1e-9)}\n')
        if self.complete_debug: print(f'new x_dot: {type(x_dot)} | {x_dot.shape} \n {x_dot}\n')
        elif self.debug: print(f'new x_dot: {x_dot}\n')

        # PFL Safety Controller
        x_dot = self.PFL(self.joint_states, x_dot)

        # Limit System Dynamic - Update `x_dot_last_cycle`
        q_dot = self.limit_joint_dynamics(x_dot)
        self.x_dot_last_cycle = np.matmul(J, q_dot)

        # FIX: Update Joint Velocity
        self.joint_states.velocity = q_dot.tolist()

        return q_dot

    def limit_joint_dynamics(self, x_dot:np.ndarray) -> np.ndarray:

        """ Limit Joint Dynamics """

        # Compute Joint Velocity (q_dot = J^-1 * x_dot)
        q_dot: np.ndarray = np.matmul(self.robot.JacobianInverse(self.joint_states.position), x_dot)
        assert q_dot.shape == (6,), f'Joint Velocity Must be a 6x1 Vector | Shape: {q_dot.shape}'
        if self.complete_debug: print(f'q_dot: {type(q_dot)} | {q_dot.shape} \n {q_dot}\n')
        elif self.debug: print(f'q_dot: {q_dot}\n')

        # Limit Joint Velocity - Max Manipulator Joint Velocity
        q_dot = np.array([np.sign(vel) * max_vel if abs(vel) > max_vel else vel for vel, max_vel in zip(q_dot, self.maximum_velocity)])

        # Limit Joint Acceleration - Max Manipulator Joint Acceleration
        q_dot = np.array([joint_vel + np.sign(vel - joint_vel) * max_acc * self.rate._timer.timer_period_ns * 1e-9
                 if abs(vel - joint_vel) > max_acc * self.rate._timer.timer_period_ns * 1e-9 else vel 
                 for vel, joint_vel, max_acc in zip(q_dot, self.joint_states.velocity, self.maximum_velocity)])

        if self.complete_debug: print(f'Limiting V_Max -> q_dot: {type(q_dot)} | {q_dot.shape} \n {q_dot}\n')
        elif self.debug: print(f'Limiting V_Max -> q_dot: {q_dot}\n')

        return q_dot

    def PFL(self, joint_states:JointState, x_dot:List[float]):

        """ PFL Controller """

        return x_dot

    def spinner(self):

        """ Main Spinner """

        # If Goal Received -> Plan Trajectory
        if self.goal_received:

            trajectory = self.robot.plan_trajectory(self.joint_states.position, self.robot.IK(self.handover_cartesian_goal), 20, 2000)
            cartesian_trajectory = self.robot.joint2cartesianTrajectory(trajectory)

        # While Goal Received but Not Reached
        while (rclpy.ok() and self.goal_received and not self.is_goal_reached(self.handover_cartesian_goal, self.joint_states)):

            # Compute Joint Velocity
            joint_velocity = self.compute_admittance_velocity(self.handover_cartesian_goal)

            # Publish Joint Velocity
            self.publishRobotVelocity(joint_velocity)

        self.goal_received = False

if __name__ == '__main__':

    # ROS Initialization
    rclpy.init()

    # Initialize Class
    pfl_controller = PFL_Controller('pfl_controller', 500)

    # Register Signal Handler (CTRL+C)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main Spinner Function
    while rclpy.ok():

        pfl_controller.spinner()

    # Delete Node before Shutdown ROS
    if pfl_controller: del pfl_controller
