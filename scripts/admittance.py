import rclpy
from rclpy.node import Rate

import numpy as np
from termcolor import colored
from scipy.spatial.transform import Rotation
from move_robot import UR_RTDE_Move

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

class AdmittanceController():

    def __init__(self, robot:UR_RTDE_Move, rate:Rate, M:np.ndarray, D:np.ndarray, K:np.ndarray, use_feedback_velocity:bool, complete_debug:bool, debug:bool):

        # Controller Parameters
        self.robot, self.rate  = robot, rate
        self.use_feedback_velocity = use_feedback_velocity
        self.complete_debug, self.debug = complete_debug, debug

        # Admittance Parameters
        self.M, self.D, self.K = M, D, K
        
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

    def compute_admittance_velocity(self, joint_states:JointState, x_des:np.ndarray, x_des_dot:np.ndarray, x_des_ddot:np.ndarray) -> np.ndarray:

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
