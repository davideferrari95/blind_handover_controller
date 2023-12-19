#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from typing import Union, List
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation

import roboticstoolbox as rtb, numpy as np
from roboticstoolbox.robot.IK import IKSolution
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.tools.trajectory import Trajectory, jtraj, ctraj
from spatialmath import SE3, SO3
from spatialmath.base.transforms3d import tr2rt, rt2tr

from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_rtde_controller.msg import CartesianPoint

from std_srvs.srv import Trigger
from ur_rtde_controller.srv import RobotiQGripperControl, GetForwardKinematic, GetInverseKinematic

GRIPPER_OPEN = 100
GRIPPER_CLOSE = 0
DYNAMIC_PLANNER = False

class UR_RTDE_Move(Node):

    trajectory_execution_received = False
    trajectory_executed = False

    def __init__(self, node_name='ur_rtde_move', robot_model='ur10e'):

        # Initialize ROS node
        super().__init__(node_name, enable_rosout=False)

        # Create Robot - Robotic Toolbox
        self.robot = rtb.models.UR10() if robot_model.lower() in ['ur10','ur10e'] else rtb.models.UR5() if robot_model.lower() in ['ur5','ur5e'] else None
        if self.robot is None: raise Exception(f"Robot Model {robot_model} not supported")

        # Initialize Jacobian (print)
        self.Jacobian([0,0,0,0,0,0]); print()

        # Publishers
        self.ur10Pub      = self.create_publisher(JointState, '/desired_joint_pose', 1)
        self.jointPub     = self.create_publisher(JointTrajectoryPoint, '/ur_rtde/controllers/joint_space_controller/command', 1)
        self.cartesianPub = self.create_publisher(CartesianPoint, '/ur_rtde/controllers/cartesian_space_controller/command', 1)

        # Subscribers
        if DYNAMIC_PLANNER: self.trajectory_execution_sub = self.create_subscription(Bool, '/trajectory_execution', self.trajectoryExecutionCallback, 1)
        else: self.trajectory_execution_sub = self.create_subscription(Bool, '/ur_rtde/trajectory_executed', self.trajectoryExecutionCallback, 1)

        # Init Gripper Service
        self.gripper_client = self.create_client(RobotiQGripperControl, '/ur_rtde/robotiq_gripper/command')

        # IK, FK Services
        self.get_FK_client = self.create_client(GetForwardKinematic, 'ur_rtde/getFK')
        self.get_IK_client = self.create_client(GetInverseKinematic, 'ur_rtde/getIK')

        # Stop Robot Service
        self.stop_service_client = self.create_client(Trigger, '/ur_rtde/controllers/stop_robot')

    def trajectoryExecutionCallback(self, msg:Bool):

        """ Trajectory Execution Callback """

        # Set Trajectory Execution Flags
        self.trajectory_execution_received = True
        self.trajectory_executed = msg.data

    def move_joint(self, joint_positions:List[float]) -> bool:

        """ Joint Space Movement """

        assert type(joint_positions) is list, f"Joint Positions must be a List | {type(joint_positions)} given | {joint_positions}"
        assert len(joint_positions) == 6, f"Joint Positions Length must be 6 | {len(joint_positions)} given"

        if DYNAMIC_PLANNER:

            # Destination Position (if `time_from_start` = 0 -> read velocity[0])
            pos = JointState()
            pos.position = joint_positions

            # Publish Joint Position
            self.ur10Pub.publish(pos)

        else:

            # Destination Position (if `time_from_start` = 0 -> read velocity[0])
            pos = JointTrajectoryPoint()
            pos.time_from_start = Duration(sec=0, nanosec=0)
            pos.velocities = [0.4]
            pos.positions = joint_positions

            # Publish Joint Position
            self.get_logger().warn('Joint Space Movement')
            self.jointPub.publish(pos)

        # Wait for Trajectory Execution
        while not self.trajectory_execution_received and rclpy.ok():

            # Debug Print
            self.get_logger().info('Waiting for Trajectory Execution', throttle_duration_sec=5, skip_first=True)

        # Reset Trajectory Execution Flag
        self.trajectory_execution_received = False

        # Exception with Trajectory Execution
        if not self.trajectory_executed: print("ERROR: An exception occurred during Trajectory Execution"); return False
        else: return True

    def move_cartesian(self, tcp_position:Pose) -> bool:

        """ Cartesian Movement """

        assert type(tcp_position) is Pose, f"Joint Positions must be a Pose | {type(tcp_position)} given | {tcp_position}"

        # Destination Position (if `time_from_start` = 0 -> read velocity[0])
        pos = CartesianPoint()
        pos.cartesian_pose = tcp_position
        pos.velocity = 0.02

        # Publish Cartesian Position
        self.get_logger().warn('Cartesian Movement')
        self.cartesianPub.publish(pos)

        # Wait for Trajectory Execution
        while not self.trajectory_execution_received and rclpy.ok():

            # Debug Print
            self.get_logger().info('Waiting for Trajectory Execution', throttle_duration_sec=5, skip_first=True)

        # Reset Trajectory Execution Flag
        self.trajectory_execution_received = False

        # Exception with Trajectory Execution
        if not self.trajectory_executed: print("ERROR: An exception occurred during Trajectory Execution"); return False
        else: return True

    def FK(self, joint_positions:List[float]) -> Pose:

        """ Forward Kinematics Using UR_RTDE Drivers """

        # Set Forward Kinematic Request
        req = GetForwardKinematic.Request()
        req.joint_position = joint_positions

        # Wait For Service
        self.get_FK_client.wait_for_service('ur_rtde/getFK')

        # Call Forward Kinematic - Asynchronous
        future = self.get_FK_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res:GetForwardKinematic.Response = future.result()

        return res.tcp_position

    def IK(self, pose:Pose, near_pose:List[float]=None) -> List[float]:

        """ Inverse Kinematics Using UR_RTDE Drivers """

        # Set Inverse Kinematic Request
        req = GetInverseKinematic.Request()
        req.tcp_position = pose

        if near_pose is not None and len(near_pose) == 6: req.near_position = near_pose
        else: req.near_position = []

        # Wait For Service
        self.get_IK_client.wait_for_service('ur_rtde/getIK')

        # Call Inverse Kinematic - Asynchronous
        future = self.get_IK_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res:GetInverseKinematic.Response = future.result()

        return list(res.joint_position)

    def ForwardKinematic(self, joint_positions:ArrayLike) -> SE3:

        """ Forward Kinematics Using Peter Corke Robotics Toolbox """

        return self.robot.fkine(joint_positions)

    def InverseKinematic(self, pose:Pose) -> IKSolution:

        """ Inverse Kinematics Using Peter Corke Robotics Toolbox """

        return self.robot.ikine_NR(pose)

    def Jacobian(self, joint_positions:ArrayLike) -> np.ndarray:

        """ Get Jacobian Matrix """

        return self.robot.jacob0(joint_positions)

    def JacobianDot(self, joint_positions:ArrayLike, joint_velocities:ArrayLike) -> np.ndarray:

        """ Get Jacobian Derivative Matrix """

        return self.robot.jacob0_dot(joint_positions, joint_velocities)

    def JacobianInverse(self, joint_positions:ArrayLike) -> np.ndarray:

        """ Get Jacobian Inverse Matrix """

        return np.linalg.inv(self.robot.jacob0(joint_positions))

    def getMaxJointVelocity(self) -> np.ndarray:

        """ Get Max Joint Velocity """

        return self.robot.qlim

    def move_gripper(self, position, speed=100, force=100,gripper_enabled=True) -> bool:

        """ Open-Close Gripper Function """

        # Return True if Gripper is not Enabled
        if not gripper_enabled: return True

        # Set Gripper Request
        req = RobotiQGripperControl.Request()
        req.position, req.speed, req.force = position, speed, force

        # Wait For Service
        self.gripper_client.wait_for_service('/ur_rtde/robotiq_gripper/command')

        # Call Gripper Service - Asynchronous
        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res:RobotiQGripperControl.Response = future.result()

        return True

    def stop_robot(self) -> bool:

        # Wait For Service
        self.stop_service_client.wait_for_service('/ur_rtde/controllers/stop_robot')

        # Call Stop Service - Asynchronous
        req = Trigger.Request()
        future = self.stop_service_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res:Trigger.Response = future.result()

        return True

    def plan_trajectory(self, start_joint_positions:List[float], end_joint_positions:List[float], duration:float=10.0, steps:int=1000) -> Trajectory:

        """ Plan Trajectory with Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(steps) is int, f"Steps must be an Integer | {type(steps)} given"
        assert type(duration) in [int, float], f"Duration must be a Int or Float | {type(duration)} given"

        # Assertion
        assert len(start_joint_positions) == 6, f"Start Joint Positions Length must be 6 | {len(start_joint_positions)} given"
        assert len(end_joint_positions) == 6, f"End Joint Positions Length must be 6 | {len(end_joint_positions)} given"
        assert duration > 0, f"Duration must be greater than 0 | {duration} given"
        assert steps > 0, f"Steps must be greater than 0 | {steps} given"

        # Create Time Vector
        time_vector = np.linspace(0, duration, steps)

        # Return Trajectory
        return jtraj(np.array(start_joint_positions), np.array(end_joint_positions), time_vector)

    # def plan_cartesian_trajectory(self, start_pose:Union[List[float], Pose], end_pose:Union[List[float], Pose], duration:float=10.0, steps:int=1000) -> Trajectory:
    def plan_cartesian_trajectory(self, start_pose:Pose, end_pose:Pose, duration:float=10.0, steps:int=1000) -> Trajectory:

        """ Plan Cartesian Trajectory with Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(steps) is int, f"Steps must be an Integer | {type(steps)} given"
        assert type(duration) in [int, float], f"Duration must be a Int or Float | {type(duration)} given"

        # Convert Geometry Pose to SE3 Matrix
        if type(start_pose) is Pose: start_pose = self.pose2matrix(start_pose)
        if type(end_pose)   is Pose: end_pose   = self.pose2matrix(end_pose)

        # Assertion
        # assert len(start_pose) == 6, f"Start Joint Positions Length must be 6 | {len(start_pose)} given"
        # assert len(end_pose) == 6, f"End Joint Positions Length must be 6 | {len(end_pose)} given"
        assert duration > 0, f"Duration must be greater than 0 | {duration} given"
        assert steps > 0, f"Steps must be greater than 0 | {steps} given"

        # Create Time Vector
        time_vector = np.linspace(0, duration, steps)

        # Return Trajectory
        traj = ctraj(start_pose, end_pose, time_vector)
        print(type(traj))
        print(traj)

        while(rclpy.ok()):pass
        # return ctraj(start_pose, end_pose, time_vector)

    def pose2numpy(self, pose:Pose) -> np.ndarray:

        """ Convert Pose to Numpy Array """

        # Type Assertion
        assert type(pose) is Pose, f"Pose must be a Pose | {type(pose)} given | {pose}"

        # Convert Position to Numpy Array
        numpy_pose = np.zeros((6,))
        numpy_pose[:3] = [pose.position.x, pose.position.y, pose.position.z]

        # Convert Orientation to Numpy Array
        orientation:Rotation = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        numpy_pose[3:] = orientation.as_rotvec()

        return numpy_pose

    def pose2matrix(self, pose:Pose) -> SE3:

        """ Convert Pose to SE3 Transformation Matrix """

        # Type Assertion
        assert type(pose) is Pose, f"Pose must be a Pose | {type(pose)} given | {pose}"

        # Convert Pose to Rotation Matrix and Translation Vector -> Create Transformation Matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
        transformation_matrix[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])

        assert transformation_matrix.shape == (4, 4), f'Transformation must a 4x4 Matrix | {transformation_matrix.shape} obtained'
        return SE3(transformation_matrix)
