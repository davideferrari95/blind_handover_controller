import numpy as np, quaternion as quat
from typing import Tuple, List
from termcolor import colored

from rclpy.node import Rate
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import JointState

# Import UR Toolbox
from robot_toolbox import UR_Toolbox, SE3

class AdmittanceController():

    def __init__(self, robot:UR_Toolbox, rate:Rate, M:np.ndarray, D:np.ndarray, K:np.ndarray, max_vel:np.ndarray, max_acc:np.ndarray,
                 admittance_weight:float=1.0, use_feedback_velocity:bool=False, complete_debug:bool=False, debug:bool=False):

        # Controller Parameters
        self.robot:UR_Toolbox = robot
        self.rate:Rate = rate
        self.maximum_velocity, self.maximum_acceleration = max_vel, max_acc
        self.admittance_weight = admittance_weight
        self.use_feedback_velocity = use_feedback_velocity
        self.complete_debug, self.debug = complete_debug, debug

        # Admittance Parameters
        self.M, self.D, self.K = M, D, K

    def position_difference(self, pos_1:SE3, pos_2:SE3) -> np.ndarray:

        """ Compute Position Difference: pos_1 - pos_2 """

        # Type Assertions
        assert isinstance(pos_1, SE3), f'Position 1 Must be a SE3 Object, given: {type(pos_1)}'
        assert isinstance(pos_2, SE3), f'Position 2 Must be a SE3 Object, given: {type(pos_2)}'

        # Compute Translation Error
        position_error = np.zeros((6,))
        position_error[:3] = pos_1.t - pos_2.t

        # Convert Rotation Matrix to Quaternion
        quat_1, quat_2 = quat.as_float_array(quat.from_rotation_matrix(pos_1.R)), quat.as_float_array(quat.from_rotation_matrix(pos_2.R))

        # Check for Quaternion Sign
        if np.dot(quat_2, quat_1) < 0: quat_1[1:] = -quat_1[1:]

        # Compute Theta
        theta = np.arccos(np.dot(quat_2, quat_1))
        if theta == 0.0: dq  = [0.0, 0.0, 0.0, 0.0]
        else: dq = theta / np.sin(theta) * (quat_1 - np.cos(theta) * quat_2)

        # Compute Orientation Error
        position_error[3:] = quat.as_float_array(2 * quat.from_float_array(dq) * quat.from_float_array(quat_2).conj())[1:]

        return position_error

    def compute_admittance_velocity(self, joint_states:JointState, external_force:Wrench, old_joint_velocity:List[float], x_des:SE3, x_des_dot:np.ndarray, x_des_ddot:np.ndarray) -> np.ndarray:

        """ Compute Admittance Cartesian Velocity """

        # Compute Manipulator Jacobian
        J = self.robot.Jacobian(joint_states.position)
        if self.complete_debug: print(colored('J: ', 'green'), f'{type(J)} | {J.shape} \n {J}\n')

        # Compute Cartesian Position
        x = self.robot.ForwardKinematic(np.array(joint_states.position))
        if self.complete_debug: print(colored('x: ', 'green'), f'{type(x)} | {x.shape} \n {x}\n')
        elif self.debug: print(colored('x:         ', 'green'), f'{self.robot.matrix2array(x)}')

        # Compute Cartesian Velocity
        x_dot: np.ndarray = J @ np.array(joint_states.velocity) if self.use_feedback_velocity else J @ np.asarray(old_joint_velocity)
        if self.complete_debug: print(colored('x_dot: ', 'green'), f'{type(x_dot)} | {x_dot.shape} \n {x_dot}\n')
        elif self.debug: print(colored('x_dot:     ', 'green'), f'{x_dot}')

        # Compute Acceleration Admittance (Mx'' + Dx' + Kx = Fe) (x = x_des - x_act) (x'' = x_des'' - u) -> u = M^-1 * (D (x_des' - x_act') + K (x_des - x_act) - Fe) + x_des'')
        x_ddot: np.ndarray = np.linalg.inv(self.M) @ (self.D @ (x_des_dot - x_dot) + self.K @ self.position_difference(x_des, x) - self.get_external_forces(external_force)) + x_des_ddot
        if self.complete_debug: print(colored('M: ', 'green'), f'{type(self.M)} \n {self.M}\n\n', colored('D: ', 'green'), f'{type(self.D)} \n {self.D}\n\n', colored('K: ', 'green'), f'{type(self.K)} \n {self.K}\n')
        if self.complete_debug: print(colored('x_dot_dot: ', 'green'), f'{type(x_ddot)} | {x_ddot.shape} \n {x_ddot}\n')
        elif self.debug: print(colored('x_dot_dot: ', 'green'), f'{x_ddot}')

        # Integrate for Velocity Based Interface
        new_x_dot = x_dot + x_ddot * self.rate._timer.timer_period_ns * 1e-9
        if self.complete_debug: print(colored('ros_rate: ', 'green'), f'{self.rate._timer.timer_period_ns * 1e-9} | ',colored('1/ros_rate: ', 'green'), f'{1/(self.rate._timer.timer_period_ns * 1e-9)}\n')
        if self.complete_debug: print(colored('new x_dot: ', 'green'), f'{type(new_x_dot)} | {new_x_dot.shape} \n {new_x_dot}\n')
        elif self.debug: print(colored('new x_dot: ', 'green'), f'{new_x_dot}\n')

        # Compute Joint Velocity (q_dot = J^-1 * new_x_dot)
        q_dot: np.ndarray = np.linalg.inv(J) @ new_x_dot

        return q_dot

    def get_external_forces(self, external_forces:Wrench) -> np.ndarray:

        """ Get FT Sensor External Forces """

        # Get External Forces
        Fe = np.zeros((6, ), dtype=np.float64)
        Fe[:3] = np.array([external_forces.force.x, external_forces.force.y, -external_forces.force.z])
        Fe[3:] = np.array([external_forces.torque.x, external_forces.torque.y, -external_forces.torque.z])

        # Remap External Forces from FT Sensor Frame to End-Effector Frame
        return self.admittance_weight * self.get_tool_transform() @ Fe

    def get_tool_transform(self) -> np.ndarray:

        """ Get End-Effector Rotation Matrix for FT Sensor """

        # Computing the actual position of the end-effector
        tool_transform = self.robot.robot.tool.R

        # Rotation Matrix 6x6
        rotation_matrix = np.zeros((6,6), dtype=np.float64)
        rotation_matrix[0:3, 0:3], rotation_matrix[3:6, 3:6] = tool_transform, tool_transform

        return rotation_matrix

    def joint2cartesian_states(self, joint_states:JointState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """ Convert Joint States to Cartesian States """

        # Compute Cartesian States
        x_act = self.robot.matrix2array(self.robot.ForwardKinematic(np.array(joint_states.position)))
        x_act_dot = self.robot.Jacobian(joint_states.position) @ np.array(joint_states.velocity)
        x_act_ddot = self.robot.Jacobian(joint_states.position) @ np.array(joint_states.effort) + self.robot.JacobianDot(joint_states.position, joint_states.velocity) @ np.array(joint_states.velocity)

        return x_act, x_act_dot, x_act_ddot
