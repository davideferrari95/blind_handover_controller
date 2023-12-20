import numpy as np
from typing import Tuple

from rclpy.node import Rate
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import JointState

# Import UR Toolbox and Scipy Rotation
from scipy.spatial.transform import Rotation
from robot_toolbox import UR_Toolbox

class AdmittanceController():

    def __init__(self, robot:UR_Toolbox, rate:Rate, M:np.ndarray, D:np.ndarray, K:np.ndarray, max_vel:np.ndarray, max_acc:np.ndarray,
                 use_feedback_velocity:bool, complete_debug:bool, debug:bool):

        # Controller Parameters
        self.robot:UR_Toolbox = robot
        self.rate:Rate = rate
        self.maximum_velocity, self.maximum_acceleration = max_vel, max_acc
        self.use_feedback_velocity = use_feedback_velocity
        self.complete_debug, self.debug = complete_debug, debug

        # Admittance Parameters
        self.M, self.D, self.K = M, D, K
        self.x_dot_last_cycle = np.zeros((6, ), dtype=np.float64)

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

    def compute_admittance_velocity(self, joint_states:JointState, external_force:Wrench, x_des:np.ndarray, x_des_dot:np.ndarray, x_des_ddot:np.ndarray) -> np.ndarray:

        """ Compute Admittance Cartesian Velocity """

        # Compute Manipulator Jacobian
        J = self.robot.Jacobian(joint_states.position)
        if self.complete_debug: print(f'J: {type(J)} | {J.shape} \n {J}\n')

        # Compute Cartesian Position
        x = self.robot.matrix2array(self.robot.ForwardKinematic(np.array(joint_states.position)))
        if self.complete_debug: print(f'x: {type(x)} | {x.shape} \n {x}\n')
        elif self.debug: print(f'x:         {x}')

        # Compute Cartesian Velocity
        x_dot: np.ndarray = J @ np.array(joint_states.velocity) if self.use_feedback_velocity else self.x_dot_last_cycle
        if self.complete_debug: print(f'x_dot: {type(x_dot)} | {x_dot.shape} \n {x_dot}\n')
        elif self.debug: print(f'x_dot:     {x_dot}')

        # Compute Acceleration Admittance (Mx'' + Dx' + Kx = Fe) (x = x_des - x_act) (x'' = x_des'' - u) -> u = M^-1 * (D (x_act' - x_des') + K (x_act - x_des) - Fe) + x_des'')
        x_ddot: np.ndarray = np.linalg.inv(self.M) @ (self.D @ (x_dot - x_des_dot) + self.K @ (x_des - x) - self.get_external_forces(external_force)) + x_des_ddot
        if self.complete_debug: print(f'M: {type(self.M)} \n {self.M}\n\n', f'D: {type(self.D)} \n {self.D}\n\n', f'K: {type(self.K)} \n {self.K}\n')
        if self.complete_debug: print(f'x_dot_dot: {type(x_ddot)} | {x_ddot.shape} \n {x_ddot}\n')
        elif self.debug: print(f'x_dot_dot: {x_ddot}')

        # Integrate for Velocity Based Interface
        new_x_dot = x_dot + x_ddot * self.rate._timer.timer_period_ns * 1e-9
        if self.complete_debug: print(f'self.ros_rate: {self.rate._timer.timer_period_ns * 1e-9} | 1/self.ros_rate: {1/(self.rate._timer.timer_period_ns * 1e-9)}\n')
        if self.complete_debug: print(f'new x_dot: {type(new_x_dot)} | {new_x_dot.shape} \n {new_x_dot}\n')
        elif self.debug: print(f'new x_dot: {new_x_dot}\n')

        # Compute Joint Velocity (q_dot = J^-1 * new_x_dot)
        q_dot: np.ndarray = self.robot.JacobianInverse(joint_states.position) @ new_x_dot

        # Limit System Dynamic
        # limited_q_dot, limited_x_dot = self.limit_joint_dynamics(joint_states, q_dot)

        # Update `x_dot_last_cycle`
        self.x_dot_last_cycle = new_x_dot

        return q_dot

    def get_external_forces(self, external_forces:Wrench) -> np.ndarray:

        """ Get FT Sensor External Forces """

        # Compute External Forces
        Fe = np.zeros((6, ), dtype=np.float64)
        Fe[:3] = np.array([external_forces.force.x, external_forces.force.y, external_forces.force.z])
        Fe[3:] = np.array([external_forces.torque.x, external_forces.torque.y, external_forces.torque.z])

        return Fe

    def joint2cartesian_states(self, joint_states:JointState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """ Convert Joint States to Cartesian States """

        # Compute Cartesian States
        x_act = self.robot.matrix2array(self.robot.ForwardKinematic(np.array(joint_states.position)))
        x_act_dot = self.robot.Jacobian(joint_states.position) @ np.array(joint_states.velocity)
        x_act_ddot = self.robot.Jacobian(joint_states.position) @ np.array(joint_states.effort) + self.robot.JacobianDot(joint_states.position, joint_states.velocity) @ np.array(joint_states.velocity)

        return x_act, x_act_dot, x_act_ddot

    def limit_joint_dynamics(self, joint_states:JointState, q_dot:np.ndarray) -> np.ndarray:

        """ Limit Joint Dynamics """

        # Type Assertions
        assert q_dot.shape == (6,), f'Joint Velocity Must be a 6x1 Vector | Shape: {q_dot.shape}'
        if self.complete_debug: print(f'q_dot: {type(q_dot)} | {q_dot.shape} \n {q_dot}\n')
        elif self.debug: print(f'q_dot: {q_dot}\n')

        # Limit Joint Velocity - Max Manipulator Joint Velocity
        q_dot = np.array([np.sign(vel) * max_vel if abs(vel) > max_vel else vel for vel, max_vel in zip(q_dot, self.maximum_velocity)])

        # Limit Joint Acceleration - Max Manipulator Joint Acceleration
        q_dot = np.array([joint_vel + np.sign(vel - joint_vel) * max_acc * self.rate._timer.timer_period_ns * 1e-9
                    if abs(vel - joint_vel) > max_acc * self.rate._timer.timer_period_ns * 1e-9 else vel 
                    for vel, joint_vel, max_acc in zip(q_dot, joint_states.velocity, self.maximum_velocity)])

        if self.complete_debug: print(f'Limiting V_Max -> q_dot: {type(q_dot)} | {q_dot.shape} \n {q_dot}\n')
        elif self.debug: print(f'Limiting V_Max -> q_dot: {q_dot}\n')

        return q_dot
