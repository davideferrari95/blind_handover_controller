from math import pi
from termcolor import colored
from typing import Union, List, Tuple

import roboticstoolbox as rtb, numpy as np
from roboticstoolbox.robot.IK import IKSolution
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.tools.trajectory import Trajectory, jtraj, ctraj

from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation
from spatialmath.base.transforms3d import tr2rt, rt2tr

from geometry_msgs.msg import Pose

class UR_Toolbox():

    def __init__(self, robot_model='ur10e'):

        # Create Robot - Robotic Toolbox
        self.robot = rtb.models.UR10() if robot_model.lower() in ['ur10','ur10e'] else rtb.models.UR5() if robot_model.lower() in ['ur5','ur5e'] else None
        if self.robot is None: raise Exception(f"Robot Model {robot_model} not supported")

        # FIX: Initialize Jacobian (print)
        self.Jacobian([0,0,0,0,0,0])
        print(f'ee_links[0]: {self.robot.ee_links[0]}\n')

    def ForwardKinematic(self, joint_positions:Union[List[float], np.ndarray]) -> SE3:

        """ Forward Kinematics Using Peter Corke Robotics Toolbox """

        # Convert Joint Positions to NumPy Array
        if type(joint_positions) is list: joint_positions = np.array(joint_positions)

        # Type Assertion
        assert type(joint_positions) in [List[float], np.ndarray], f"Joint Positions must be a ArrayLike | {type(joint_positions)} given"
        assert len(joint_positions) == 6, f"Joint Positions Length must be 6 | {len(joint_positions)} given \nJoint Positions:\n{joint_positions}"

        return self.robot.fkine(joint_positions)

    def InverseKinematic(self, pose:Union[Pose, NDArray, SE3]) -> IKSolution:

        """ Inverse Kinematics Using Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(pose) in [Pose, np.ndarray, SE3], f"Pose must be a Pose, NumPy Array or SE3 | {type(pose)} given"

        # Convert Pose to SE3 Matrix
        if type(pose) is Pose: pose = self.pose2matrix(pose)

        return self.robot.ikine_NR(pose, q0=[0, -pi/2, pi/2, -pi/2, -pi/2, 0])

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

    def plot(self, joint_pos:ArrayLike=[0, -pi/2, pi/2, -pi/2, -pi/2, 0]):

        """ Plot Robot Peter Corke Robotics Toolbox """

        # TODO: Enhance Simulation (add following trajectory)
        self.robot.plot(joint_pos, block=True)

        # https://github.com/petercorke/robotics-toolbox-python

        # import swift
        # import roboticstoolbox as rtb
        # import spatialmath as sm
        # import numpy as np

        # env = swift.Swift()
        # env.launch(realtime=True)

        # panda = rtb.models.Panda()
        # panda.q = panda.qr

        # Tep = panda.fkine(panda.q) * sm.SE3.Trans(0.2, 0.2, 0.45)

        # arrived = False
        # env.add(panda)

        # dt = 0.05

        # while not arrived:

        #     v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
        #     panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
        #     env.step(dt)

        # # Uncomment to stop the browser tab from closing
        # # env.hold()

    def plan_trajectory(self, start:Union[List[float], Pose, SE3], end:Union[List[float], Pose, SE3], duration:float=10.0, steps:int=1000) -> Trajectory:

        """ Plan Trajectory with Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(steps) is int, f"Steps must be an Integer | {type(steps)} given"
        assert type(duration) in [int, float], f"Duration must be a Int or Float | {type(duration)} given"

        # Convert Geometry Pose, SE3 Matrix to Joint Position Array
        start_joint_pos = self.InverseKinematic(start).q if type(start) in [Pose, SE3] else start
        end_joint_pos   = self.InverseKinematic(end).q   if type(end)   in [Pose, SE3] else end

        # Assertion
        assert len(start_joint_pos) == 6, f"Start Joint Positions Length must be 6 | {len(start_joint_pos)} given"
        assert len(end_joint_pos) == 6, f"End Joint Positions Length must be 6 | {len(end_joint_pos)} given"
        assert duration > 0, f"Duration must be greater than 0 | {duration} given"
        assert steps > 0, f"Steps must be greater than 0 | {steps} given"

        # Create Time Vector
        time_vector = np.linspace(0, duration, steps)

        # Return Trajectory
        return jtraj(np.array(start_joint_pos), np.array(end_joint_pos), time_vector)

    def plan_cartesian_trajectory(self, start_pose:Pose, end_pose:Pose, duration:float=10.0, steps:int=1000) -> List[SE3]:

        """ Plan Cartesian Trajectory with Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(steps) is int, f"Steps must be an Integer | {type(steps)} given"
        assert type(duration) in [int, float], f"Duration must be a Int or Float | {type(duration)} given"

        # Convert Geometry Pose to SE3 Matrix
        if type(start_pose) is Pose: start_pose = self.pose2matrix(start_pose)
        if type(end_pose)   is Pose: end_pose   = self.pose2matrix(end_pose)

        # Assertion
        assert duration > 0, f"Duration must be greater than 0 | {duration} given"
        assert steps > 0, f"Steps must be greater than 0 | {steps} given"

        # Create Time Vector
        time_vector = np.linspace(0, duration, steps)

        # Return Cartesian Trajectory (Array of SE3)
        return ctraj(start_pose, end_pose, time_vector)

    def joint2cartesianTrajectory(self, joint_trajectory:Trajectory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """ Convert Joint Trajectory to Cartesian Trajectory """

        # Type Assertion
        assert type(joint_trajectory) is Trajectory, f"Joint Trajectory must be a Trajectory | {type(joint_trajectory)} given"

        # Initialize Cartesian Trajectory
        cartesian_positions, cartesian_velocities, cartesian_accelerations = [], [], []

        # Print Joint Trajectory Shape
        print(f'Joint Trajectory Shape | Positions: {joint_trajectory.q.shape} | Velocities: {joint_trajectory.qd.shape} | Accelerations: {joint_trajectory.qdd.shape}')

        # Convert Joint to Cartesian Positions
        for q, q_dot, q_ddot in zip(joint_trajectory.q, joint_trajectory.qd, joint_trajectory.qdd):

            # Convert Joint Position to Cartesian Position (x = ForwardKinematic(q))
            x = self.matrix2numpy(self.ForwardKinematic(q))
            cartesian_positions.append(x)

            # Convert Joint Velocity to Cartesian Velocity (x_dot = Jacobian(q) * q_dot)
            x_dot = self.Jacobian(q) @ q_dot
            cartesian_velocities.append(x_dot)

            # Convert Joint Acceleration to Cartesian Acceleration (x_ddot = Jacobian(q) * q_ddot + JacobianDot(q, q_dot) * q_dot)
            x_ddot = self.Jacobian(q) @ q_ddot + self.JacobianDot(q, q_dot) @ q_dot
            cartesian_accelerations.append(x_ddot)

        print(f'Cartesian Trajectory Shape | Positions: {np.array(cartesian_positions).shape} | Velocities: {np.array(cartesian_velocities).shape} | Accelerations: {np.array(cartesian_accelerations).shape}\n')
        print (colored('Cartesian Positions:', 'green'),     f'\n\n {np.array(cartesian_positions)}\n')
        print (colored('Cartesian Velocities:', 'green'),    f'\n\n {np.array(cartesian_velocities)}\n')
        print (colored('Cartesian Accelerations:', 'green'), f'\n\n {np.array(cartesian_accelerations)}\n')

        return np.array(cartesian_positions), np.array(cartesian_velocities), np.array(cartesian_accelerations)

    def pose2array(self, pose:Pose) -> np.ndarray:

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

    def matrix2pose(self, matrix:SE3) -> Pose:

        """ Convert SE3 Transformation Matrix to Pose """

        # Type Assertion
        assert type(matrix) is SE3, f"Matrix must be a SE3 | {type(matrix)} given | {matrix}"

        # Convert Rotation Matrix to Quaternion (x,y,z,w)
        quaternion = Rotation.from_matrix(matrix.R).as_quat()

        # Create Pose
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = matrix.t
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion

        return pose

    def matrix2array(self, matrix:SE3) -> np.ndarray:

        """ Convert SE3 Transformation Matrix to NumPy Array """

        # Type Assertion
        assert type(matrix) is SE3, f"Matrix must be a SE3 | {type(matrix)} given | {matrix}"

        # Convert Rotation Matrix to Rotation
        rotation:Rotation = Rotation.from_matrix(matrix.R)

        # Create Numpy Pose Array
        numpy_pose = np.zeros((6,))
        numpy_pose[:3] = matrix.t
        numpy_pose[3:] = rotation.as_rotvec()

        return numpy_pose

    def matrix2numpy(self, matrix:SE3) -> np.ndarray:

        """ Convert SE3 Transformation Matrix to NumPy 4x4 Matrix """

        # Type Assertion
        assert type(matrix) is SE3, f"Matrix must be a SE3 | {type(matrix)} given | {matrix}"

        # Convert to Numpy
        return matrix.A
