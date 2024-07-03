#! /usr/bin/env python3

import rclpy, time
from rclpy.node import Node
from typing import List
from termcolor import colored

# Messages & Services
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Wrench
from std_msgs.msg import Bool, String, Float64MultiArray, MultiArrayDimension, Int64
from ur_rtde_controller.srv import RobotiQGripperControl
from handover_controller.srv import InverseKinematic

TEST = True

# Take Photo Position (X,Y,Z - R,P,Y)
TAKE_PHOTO_POSE = [5.120, -0.320, 2.650, 1.650, -31.370, 0.0]

# Home Joint Position
HOME = [-3.0, -1.62, 1.69, -1.66, -1.50, 0.0]

# Available Objects
AVAILABLE_OBJECTS = {
    'tubetto' : 0,
    'ugello'  : 1,
    'dado'    : 2,
    'vite_65' : 3,
    'chiave'  : 4,
    # "tubetto_m1200.ply": 0,
    # "ugello_l80_99.ply": 1,
    # "dado_m5.ply": 2,
    # "vite_65.ply": 3,
    # "vite_20.ply": 4,
    # "chiave_brugola_6.ply": 5,
    # "deviatore_boccaglio.ply": 6,
    # "chiave2.ply": 7,
    # "chiave_fissa_8_10_2.ply": 8,
    # "fascetta_68_73_colorized_2.ply": 9
}

import numpy as np
from scipy.spatial.transform import Rotation

def array2pose(self, array:np.ndarray) -> Pose:

    """ Convert NumPy Array to Pose """

    # Type Assertion
    assert type(array) is np.ndarray, f"Array must be a NumPy Array | {type(array)} given | {array}"
    assert len(array) == 6, f"Array Length must be 6 | {len(array)} given | {array}"

    # Create Pose
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = array[:3]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = Rotation.from_rotvec(array[3:]).as_quat()

    return pose

class Experiment(Node):

    """ FT-Sensor Experiment Node """

    # Initial Joint States Data
    joint_states, ft_sensor_data = JointState(), Wrench()
    joint_states.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joint_states.position, joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Initialize Flags
    start_handover = False
    requested_object = None

    # Initial Joint Goals - UR5e
    handover_goal = Pose()

    def __init__(self, ros_rate:int=500):

        # Node Initialization
        super().__init__('FT_Sensor_Experiment')

        # ROS2 Rate
        self.ros_rate = ros_rate
        self.rate = self.create_rate(ros_rate)

        # ROS2 Parameters
        self.declare_parameter('use_network', True)
        self.use_network = self.get_parameter('use_network').get_parameter_value().bool_value

        # Print Parameters
        print(colored('\nUse Network:', 'green'), f'\t{self.use_network}\n')

        # ROS2 Publisher Initialization
        self.joint_goal_pub      = self.create_publisher(Float64MultiArray, '/handover/joint_goal', 1)
        self.cartesian_goal_pub  = self.create_publisher(Pose, '/handover/cartesian_goal', 1)
        self.trajectory_time_pub = self.create_publisher(Int64, '/handover/set_trajectory_time', 1)
        self.track_hand_pub      = self.create_publisher(Bool, '/handover/track_hand', 1)
        self.get_object_pose_pub      = self.create_publisher(String, '/6D_pose/request_object', 1)

        # ROS2 Service Clients Initialization
        self.stop_admittance_client = self.create_client(Trigger, '/handover/stop')
        self.zero_ft_sensor_client  = self.create_client(Trigger, '/ur_rtde/zeroFTSensor')
        self.robotiq_gripper_client = self.create_client(RobotiQGripperControl, '/ur_rtde/robotiq_gripper/command')
        self.inverse_kinematic_client = self.create_client(InverseKinematic, '/handover/inverse_kinematic')

        # ROS2 Subscriber Initialization
        self.get_object_subscriber      = self.create_subscription(String, '/handover/get_object', self.objectCallback, 1)
        self.joint_state_subscriber     = self.create_subscription(JointState, '/joint_states', self.jointStatesCallback, 1)
        self.ft_sensor_subscriber       = self.create_subscription(Wrench,'/ur_rtde/ft_sensor', self.FTSensorCallback, 1)
        self.network_output_subscriber  = self.create_subscription(Bool, '/ft_network/open_gripper', self.networkCallback, 1)
        self.human_hand_pose_subscriber = self.create_subscription(Pose, '/handover/human_hand', self.humanHandPoseCallback, 1)
        self.wait_object_pose_subscriber    = self.create_subscription(Pose, '/6D_pose/object_pose', self.getObjectPoseCallback, 1)

        time.sleep(1)

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

    def FTSensorCallback(self, data:Wrench):

        """ FT Sensor Callback """

        # Get FT Sensor Data
        self.ft_sensor_data = data

    def networkCallback(self, data:Bool):

        """ Network Callback """

        print('Network Output:', data.data)

        # Get Network Output
        if data.data: self.open_gripper = True

    def objectCallback(self, data:String):

        """ Object Callback """

        # Get Object Command
        if data.data in AVAILABLE_OBJECTS:

            # Set Handover Flag and Requested Object
            self.start_handover = True
            self.requested_object = data.data

    def humanHandPoseCallback(self, data:Pose):

        """ Human Hand Pose Callback """

        # Get Human Hand Pose - Handover Goal
        self.handover_goal = data

    def publishJointGoal(self, joint_goal:List[float]):

        """ Publish Handover Joint Goal """

        assert len(joint_goal) == 6, 'Joint Goal Must be a 6-Element List'

        # ROS Message Creation
        msg = Float64MultiArray(data=joint_goal)
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].size = len(joint_goal)
        msg.layout.dim[0].stride = 1
        msg.layout.dim[0].label = 'joint_goal'

        # Publish Message
        if rclpy.ok(): self.joint_goal_pub.publish(msg)

    def stopHandover(self):

        """ Call Stop Handover Service """

        # Wait For Service
        if self.stop_admittance_client.wait_for_service(timeout_sec=0.2):

            # Call Service - Asynchronous
            future = self.stop_admittance_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            return future.result()

        else: self.get_logger().error('Stop Handover Service Not Available')

    def zeroFTSensor(self):

        """ Call Zero FT Sensor Service """

        # Wait For Service
        if self.zero_ft_sensor_client.wait_for_service(timeout_sec=1.0):

            # Call Service - Asynchronous
            future = self.zero_ft_sensor_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            return future.result()

        else: self.get_logger().error('Zero FT Sensor Service Not Available')

    def RobotiQGripperControl(self, position:int, speed:int=100, force:int=100):

        """ Call RobotiQ Gripper Service """        

        # Wait For Service
        self.robotiq_gripper_client.wait_for_service(timeout_sec=1.0)

        # Gripper Service Request
        request = RobotiQGripperControl.Request()
        request.position, request.speed, request.force = position, speed, force

        # Call Service - Asynchronous
        future = self.robotiq_gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def getIK(self, cartesian_pose:Pose):

        """ Call IK Service """

        # Wait For Service
        if self.inverse_kinematic_client.wait_for_service(timeout_sec=1.0):

            # Fill Request
            request = InverseKinematic.Request()
            request.cartesian_pose = cartesian_pose

            # Call Service - Asynchronous
            future = self.inverse_kinematic_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            return future.result()

        else: self.get_logger().error('IK Service Not Available'); return None

    def goal_reached(self, joint_goal:List[float]):
        
        """ Check if Goal Reached """

        # Spin Once
        rclpy.spin_once(self)

        # Get Joint States
        joint_states = self.joint_states

        # Check if Goal Reached
        if all([abs(joint_states.position[i] - joint_goal[i]) < 0.01 for i in range(len(joint_states.position))]): return True
        else: return False

    def move_and_wait(self, joint_goal:List[float], goal_name:str='Goal', throttle_duration_sec:float=5.0, skip_first:bool=False):

        """ Move to Joint Goal and Wait for Goal to be Reached """

        # Reset FT-Sensor
        self.zeroFTSensor()

        # Publish Joint Goal
        self.publishJointGoal(joint_goal)

        # Stop Hand Tracking
        if goal_name == 'HOME': self.track_hand_pub.publish(Bool(data=False))

        # Wait for Goal to be Reached
        while not self.goal_reached(joint_goal): self.get_logger().info(f'Moving to {goal_name}', throttle_duration_sec=throttle_duration_sec, skip_first=skip_first)
        self.get_logger().info(f'{goal_name} Reached\n')

    def move_cartesian(self, cartesian_pose:Pose, goal_name:str='Handover Goal'):

        """ Move to Cartesian Goal """

        # Publish Joint Goal
        self.cartesian_goal_pub.publish(cartesian_pose)
        self.get_logger().info(f'Moving to {goal_name}')

    def wait_for_network(self):

        """ Wait for Network Output """

        # Initialize Open Gripper Flag
        self.open_gripper = False

        # Wait for Network to Open Gripper
        while not self.open_gripper:

            # Spin Once
            rclpy.spin_once(self, timeout_sec=0.5/float(self.ros_rate))

            # Log
            self.get_logger().info('Waiting for Network to Open Gripper', throttle_duration_sec=5.0, skip_first=False)

        # Network Opened Gripper
        self.get_logger().info('Network Opened Gripper\n')

    def wait_for_ft_load(self):

        """ Wait for FT-Load Threshold Output """

        # Wait to Open Gripper
        while True:

            # Spin Once
            rclpy.spin_once(self, timeout_sec=0.5/float(self.ros_rate))

            # Check FT-Load Threshold -> Break
            if self.ft_sensor_data.force.z > 10.0: break

            # Log
            self.get_logger().info('Waiting FT-Load Threshold to Open Gripper', throttle_duration_sec=5.0, skip_first=False)

        # Network Opened Gripper
        self.get_logger().info('FT-Load Threshold Opened Gripper\n')

    def getObjectPosition(self, object_name:str):

        """ Ask Object Position to 6D Pose Estimation Node """

        # Request Object Position
        self.object_pose_received = False
        self.get_object_pose_pub.publish(String(data=object_name))

        # Wait for Object Position
        while not self.object_pose_received:

            # Spin Once
            rclpy.spin_once(self, timeout_sec=0.1/float(self.ros_rate))

            # Log Throttle
            self.get_logger().info(f'Getting {object_name} Position', throttle_duration_sec=5.0, skip_first=False)

        # Get Object Position
        pick_position, over_position = self.requested_object_position, self.requested_object_position

        # Update Object Over Position
        over_position.position.z += 0.10

        return over_position, pick_position

    def getObjectPoseCallback(self, data:Pose):

        """ Get Object Pose from 6D Pose Callback """

        # Get Object Pose
        self.requested_object_position = data

        # Set Object Pose Received Flag
        self.object_pose_received = True

    def handover(self, object_name:str):

        """ Handover """

        assert object_name in AVAILABLE_OBJECTS, f'Invalid Object Name: {object_name}'

        # Reset FT-Sensor
        self.zeroFTSensor()
        self.get_logger().warn(f"Sure, I'll get the {self.requested_object} for you.")

        # Go to Photo Position
        photo_joint_pose = self.getIK(array2pose(TAKE_PHOTO_POSE))
        if photo_joint_pose is not None: self.move_and_wait(photo_joint_pose, 'Photo Position', 5.0, False)

        # Get Object Pick Positions
        object_over_pose, object_pick_pose = self.getObjectPosition(object_name)
        object_over, object_pick = self.getIK(object_over_pose), self.getIK(object_pick_pose)

        # Go to Object Goal
        if object_over is not None: self.move_and_wait(object_over, 'Object Over', 5.0, False)
        if object_pick is not None: self.move_and_wait(object_pick, 'Object Pick', 5.0, False)
        time.sleep(1)

        # Close Gripper
        self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_CLOSED)
        time.sleep(1)

        # Compute Handover Goal - Gripper Distance
        self.handover_goal.position.x += -0.10
        self.handover_goal.position.y += -0.25
        self.handover_goal.position.z += 0.20

        # Fixed Orientation Goal
        self.handover_goal.orientation.x = -0.96
        self.handover_goal.orientation.y = 0.25
        self.handover_goal.orientation.z = 0.02
        self.handover_goal.orientation.w = 0.06

        if TEST:

            # Fixed Handover Goal - For Testing
            handover_goal_test = [-2.48739463487734, -1.3766034108451386, 1.7061370054828089, -1.8849464855589808, -1.588557545338766, 0.5314063429832458]
            self.publishJointGoal(handover_goal_test)

        else:

            # Go to Handover Goal
            self.move_cartesian(self.handover_goal)
            time.sleep(3)

        # Log Handover
        self.get_logger().warn(f"I'm handing you the {object_name}")

        # Publish Hand Tracking
        if not TEST: self.track_hand_pub.publish(Bool(data=True))

        # Wait for Network or FT-Load Threshold to Open Gripper
        if self.use_network: self.wait_for_network()
        else: self.wait_for_ft_load()

        # Open Gripper
        self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_OPENED)
        time.sleep(1)

    def main(self):

        """ Main Loop """

        # Open Gripper and Go to Home
        self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_OPENED)
        self.move_and_wait(HOME, 'HOME', 5.0, False)
        time.sleep(1)

        # Wait for Start Handover
        while rclpy.ok() and not self.start_handover:

            # Spin Once
            rclpy.spin_once(self, timeout_sec=0.1/float(self.ros_rate))

        # Handover Requested Object
        print(f'\nHandover Object: {self.requested_object}\n')
        self.handover(self.requested_object)

        # Stop Handover
        print('\nStopping Handover\n')
        self.stopHandover()

        # Initialize Flags
        self.start_handover = False
        self.requested_object = None

        # Sleep
        time.sleep(2)

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = Experiment(500)

    while rclpy.ok():

        # Run Node
        node.main()
