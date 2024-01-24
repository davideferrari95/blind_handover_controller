#! /usr/bin/env python3

import os, numpy as np
from termcolor import colored

# Import ROS2 Messages
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState

from utils.robot_toolbox import UR_Toolbox
from scipy.optimize import linprog

class SafetyController():

    def __init__(self, robot:UR_Toolbox, robot_parameters:dict, human_radius:float=0.1, ros_rate:int=500, complete_debug:bool=False, debug:bool=False):

        """ Power Force Limiting (PFL) Controller """

        # Class Parameters
        self.robot, self.ros_rate = robot, ros_rate
        self.complete_debug, self.debug = complete_debug, debug or complete_debug

        # Human Parameters
        self.human_radius = human_radius

        # Robot Parameters
        self.stopping_time, self.stopping_distance = robot_parameters['stopping_time'], robot_parameters['stopping_distance']
        self.max_speed = max(robot_parameters['q_limits'])
        self.robot_mass = robot_parameters['mass']

    def compute_robot_point(self, joint_states:JointState) -> Vector3:

        """ Compute Robot Point (PR) """

        # Get the Robot Pose from the Joint States
        cartesian_pose = self.robot.matrix2pose(self.robot.ForwardKinematic(np.array(joint_states.position)))

        # Create PointStamped Message
        robot_point = Vector3()
        robot_point.x, robot_point.y, robot_point.z = cartesian_pose.position.x, cartesian_pose.position.y, cartesian_pose.position.z

        return robot_point

    def compute_versor(self, v1:Vector3, v2:Vector3) -> Vector3:

        """ Compute Versor """

        # Compute Distance (Euclidean Norm) Between Two Points
        distance = np.linalg.norm(np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z]))

        # return (P_H_ - P_R_) / (P_R_.distance(P_H_));
        return Vector3(x = (v1.x - v2.x) / distance, y = (v1.y - v2.y) / distance, z = (v1.z - v2.z) / distance)

    def compute_minimum_distance(self, v1:Vector3, v2:Vector3) -> float:

        """ Compute Minimum Distance Between Two Points """

        # Compute Distance (Euclidean Norm) Between Two Points
        return np.linalg.norm(np.array([v1.x - v2.x, v1.y - v2.y, v1.z - v2.z]))

    def compute_SSM_vel_lim(self, P_H:Vector3, P_R:Vector3, hr_versor:Vector3, human_vel:Vector3) -> float:

        """ Compute Speed-And-Separation Monitoring ISO/TS 15066 Velocity Limit """

        """
            A Safety-Aware Kinodynamic Architecture for Human-Robot Collaboration
            Andrea Pupa, Mohammad Arrfou, Gildo Andreoni, Cristian Secchi

            https://arxiv.org/pdf/2103.01818.pdf

            @ARTICLE{9385862,
                author={Pupa, Andrea and Arrfou, Mohammad and Andreoni, Gildo and Secchi, Cristian},
                journal={IEEE Robotics and Automation Letters},
                title={A Safety-Aware Kinodynamic Architecture for Human-Robot Collaboration},
                year={2021},
                volume={6},
                number={3},
                pages={4465-4471},
                doi={10.1109/LRA.2021.3068634}
            }

            Sp(t0) = Sh + Sr + Ss + C + Zd + Zr

            Sh = Vh(t0) * (Ts + Tr)
            Sr = Vr(t0) * Tr
            Ss = Vr(t0) * Ts + a_max * Ts^2 / 2

                Sp = Safety Margin (Distance Between Human and Robot)
                Sh = Safety Margin from Human Velocity
                Sr = Safety Margin from Robot Velocity
                Ss = Safety Margin from Robot Stopping Time
                C = Intrusion Distance (Distance Needed for the Sensor to Detect the Human)
                Zd, Zr = Position Uncertainty (Human and Robot)
                Tr, Ts = Robot Reaction and Stopping Time
                Vh = Directed Speed of the Human Towards the Robot
                Vr = Directed Speed of the Robot Towards the Human
                Vs = Speed of the Robot in the Course of Stopping
                a_max = Maximum Robot Acceleration

            Maximum Robot Velocity (Vr_max) is computed as follows:
            Vr_max(t0) = (Sp(t0) - Vh(t0) * (Ts + Tr) - C - Zd - Zr) / (Ts + Tr) - (a_max * Ts) / (2 * (Ts + Tr))

        """

        # Compute Minimum Distance Between Human and Robot
        Sp = self.compute_minimum_distance(P_H, P_R) - self.human_radius

        # Uncertainties Parameters, Stopping Time and Reaction Time (Assume Tr, C, Zd, Zr = 0)
        Ts, Tr, C, Zd, Zr = self.stopping_time, 0.0, 0.0, 0.0, 0.0

        # Compute Human Projected Velocity
        Vh = np.array([human_vel.x, human_vel.y, human_vel.z]) @ np.array([hr_versor.x, hr_versor.y, hr_versor.z])
        if self.complete_debug: print(colored('Human Velocity: ', 'green'), f'{human_vel.x} {human_vel.y} {human_vel.z}')
        if self.complete_debug: print(colored('Human Projected Velocity: ', 'green'), Vh, '\n')

        # Robot Maximum Acceleration
        a_max = np.deg2rad(self.max_speed) / self.stopping_time

        # Compute ISO/TS 15066 Velocity Limit
        return (Sp - Vh * (Ts + Tr) - C - Zd - Zr) / (Ts + Tr) - (a_max * Ts**2) / (2 * (Ts + Tr))

    def compute_PFL_vel_lim(self) -> float:

        """ Compute Power and Force Limiting ISO/TS 15066 Velocity Limit """

        """ ISO/TS 15066 - Appendix A

            E = F^2 / 2k = A^2 * p^2 / 2k = 1/2 * μ * v_rel^2

                E:     transfer energy.
                F:     maximum contact force for specific body region (Table A.2).
                p:     maximum contact pressure for specific body area (Table A.2).
                k:     effective spring constant for specific body region (Table A.3).
                A:     area of contact between robot and body region.
                v_rel: relative speed between the robot and the human body region.
                μ:     reduced mass of the two-body system.

            μ = (1/m_h + 1/m_r)^-1

                m_h: effective mass of the human body region (Table A.3).
                m_r: effective mass of the robot as a function of the robot posture and motion.

            m_r = M/2 + m_l

                m_l: effective payload of the robot system, including tooling and workpiece.
                M:   total mass of the moving parts of the robot.

            v_rel = F / sqrt(μ*k) = p*A / sqrt(μ*k) -> v_rel_max = F_max / sqrt(μ*k) = p_max*A / sqrt(μ*k)

        """

        """ Table A.2 - Biomechanical limits

                Body Region     |  Specific Body Area  | Maximum Pressure | Maximum Force 
                                                            [p, N/cm^2]          [N]

            Skull and forehead    Middle of forehead            130              130
            Skull and forehead    Temple                        110              130

            Face                  Masticatory muscle            110              65

            Neck                  Neck muscle                   140              150
            Neck                  Seventh neck muscle           210              150

            Back and shoulders    Shoulder joint                160              210
            Back and shoulders    Fifth lumbar vertebra         210              210

            Chest                 Sternum                       120              140
            Chest                 Pectoral muscle               170              140

            Abdomen               Abdominal muscle              140              110

            Pelvis                Pelvic bone                   210              180

            Upper arms            Deltoid muscle                190              150
            Upper arms            Humerus                       220              150

            Lower arms            Radial bone                   190              160
            Lower arms            Forearm muscle                180              160
            Lower arms            Arm nerve                     180              160

            Hands and fingers     Forefinger pad                300              140
            Hands and fingers     Forefinger end joint          280              140
            Hands and fingers     Thenar eminence               200              140
            Hands and fingers     Palm                          260              140
            Hands and fingers     Back of the hand              200              140

            Thighs and knees      Thigh muscle                  250              220
            Thighs and knees      Kneecap                       220              220

            Lower legs            Middle of shin                220              130
            Lower legs            Calf muscle                   210              130

        """

        """ Table A.3 - Effective masses and spring constants for the body model

            Body region         | Effective spring constant | Effective mass 
                                          [K, N/mm]              [mH, kg]

            Skull and forehead               150                    4,4
            Face                             75                     4,4
            Neck                             50                     1,2
            Back and shoulders               35                     40
            Chest                            25                     40
            Abdomen                          10                     40
            Pelvis                           25                     40
            Upper arms and elbow joints      30                      3
            Lower arms and wrist joints      40                      2
            Hands and fingers                75                     0,6
            Thighs and knees                 50                     75
            Lower legs                       60                     75

        """

        """ PFL Parameters:

            m_l = 1.0 kg               | effective payload of the robot system, including tooling and workpiece.
            M   = sum(link 2 - link 6) | total mass of the moving parts of the robot.
            m_h = 3.6 kg               | effective mass of the human body region (Table A.3) - Upper arms and elbow joints + Hands and fingers.
            k   = 75 N/mm              | effective spring constant for specific body region (Table A.3) - max (Hands and fingers, Upper arms and elbow joints).
            F   = 140 N                | maximum contact force for specific body region (Table A.2) - min (Hands and fingers, Upper arms and elbow joints).
            p   = 190 N/cm^2           | maximum contact pressure for specific body area (Table A.2) - min (Hands and fingers, Upper arms and elbow joints).
            A   = 1 cm^2               | Contact area A is defined by the smaller of the surface areas of the robot or the operator.
                                         In situations where the body contact surface area is smaller than robot contact surface area, such as the operator’s
                                         hands or fingers, the body contact surface area shall be used. If contact between multiple body areas with  different
                                         potential surface contact areas could occur, the value A that yields the lowest v_rel_max shall be used. - Hands and fingers.
        """

        # Define PFL Parameters (kg, kg, N/m, N/cm^2, cm^2)
        m_l, M, m_h, k, F = 1.0, np.sum(self.robot_mass[1:]), 3.6, 75e3, 140

        # Effective Mass of the Robot
        m_r = M/2 + m_l

        # Reduced Mass of the Two-Body System
        μ = (1/m_h + 1/m_r)**-1

        # Compute Relative Speed Between the Robot and the Human Body Region
        v_rel_max = F / np.sqrt(μ*k)

        return v_rel_max

    def compute_alpha_optim(self, joint_states:JointState, desired_joint_velocity:np.ndarray, old_joint_velocity:np.ndarray, hr_versor:Vector3, vel_limit:float) -> float:

        """ Compute Scaling Factor α - Optimization Problem """

        """
        minimize

            c @ x

        such that

            A_ub @ x <= b_ub
            A_eq @ x == b_eq
            lb <= x <= ub

        """

        # Compute Modified Jacobian Matrix (Jri)
        Jri = np.transpose([hr_versor.x, hr_versor.y, hr_versor.z, 0, 0, 0]) @ self.robot.Jacobian(np.array(joint_states.position))

        # Get Robot Velocity, Acceleration Limits - Convert to Numpy Array
        q_dot_lim, q_ddot_lim = np.array(self.robot.qd_lim), np.array(self.robot.qdd_lim)

        """ Problem Definition

            minimize

                -1 @ alpha

            such that

                Jri @ desired_joint_velocity @ alpha <= vel_limit
                desired_joint_velocity @ alpha <= q_dot_lim
                desired_joint_velocity @ alpha >= -q_dot_lim -> -desired_joint_velocity @ alpha <= q_dot_lim
                (desired_joint_velocity @ alpha - old_joint_velocity) * self.ros_rate <= q_ddot_lim
                (desired_joint_velocity @ alpha - old_joint_velocity) * self.ros_rate >= -q_ddot_lim -> - (desired_joint_velocity @ alpha - old_joint_velocity) * self.ros_rate <= q_ddot_lim

         """

        # Objective Function
        c = [-1]

        # ISO/TS 15066 Velocity Constraint
        A, b = [[Jri @ desired_joint_velocity]], [vel_limit]

        # Velocity Limits Constraints
        A = A + [[+desired_joint_velocity[i]] for i in range(len(desired_joint_velocity))]
        A = A + [[-desired_joint_velocity[i]] for i in range(len(desired_joint_velocity))]
        b = b + [q_dot_lim[i] for i in range(len(q_dot_lim))]
        b = b + [q_dot_lim[i] for i in range(len(q_dot_lim))]

        # Acceleration Limits Constraints
        A = A + [[+desired_joint_velocity[i]] for i in range(len(desired_joint_velocity))]
        A = A + [[-desired_joint_velocity[i]] for i in range(len(desired_joint_velocity))]
        b = b + [q_ddot_lim[i]/self.ros_rate + old_joint_velocity[i] for i in range(len(q_ddot_lim))]
        b = b + [q_ddot_lim[i]/self.ros_rate - old_joint_velocity[i] for i in range(len(q_ddot_lim))]

        # Optimization Problem (Bounds: 0 <= alpha <= 1)
        result = linprog(c, A_ub=A, b_ub=b, bounds=[(0.0, 1.0)])

        # Return Optimal Scaling Factor
        return result.x.item()

    def compute_alpha(self, joint_states:JointState, desired_joint_velocity:np.ndarray, hr_versor:Vector3, vel_limit:float) -> float:

        """ Compute Scaling Factor α """

        # Compute Robot Projected Desired Velocity
        x_dot: np.ndarray = self.robot.Jacobian(np.array(joint_states.position)) @ np.array(desired_joint_velocity)
        Vr:float = np.dot(np.array([x_dot[0], x_dot[1], x_dot[2]]), np.array([hr_versor.x, hr_versor.y, hr_versor.z]))
        if self.complete_debug: print(colored('Robot Desired Velocity: ', 'green'), x_dot)
        if self.complete_debug: print(colored('Robot Projected Desired Velocity: ', 'green'), Vr, '\n')

        # Compute Scaling Factor
        scaling_factor = 1 if Vr <= 0 else min(vel_limit / Vr, 1)
        if self.debug or self.complete_debug: print(colored('Scaling Factor: ', 'green'), scaling_factor, '\n')
        if self.debug or self.complete_debug: print('-'*100, '\n')

        return scaling_factor

    def compute_safety(self, desired_joint_velocity:np.ndarray, old_joint_velocity:np.ndarray, joint_states:JointState, human_point:Vector3, human_vel:Vector3)  -> np.ndarray:

        """ Compute Safety Limits Switching between PFL and SSM ISO/TS 15066 Limits """

        # Compute PH and PR Vector3 - HR Versor
        P_H, P_R = human_point, self.compute_robot_point(joint_states)
        hr_versor = self.compute_versor(P_H, P_R)

        if self.complete_debug: print(colored('\nSafety Controller:\n', 'green'))
        if self.complete_debug: print(colored('Human Point: ', 'green'), f'{P_H.x} {P_H.y} {P_H.z}')
        if self.complete_debug: print(colored('Robot Point: ', 'green'), f'{P_R.x} {P_R.y} {P_R.z}', '\n')
        if self.complete_debug: print (colored('HR Versor: ', 'green'), f'{hr_versor.x} {hr_versor.y} {hr_versor.z}\n')

        # Compute SSM, PFL ISO/TS 15066 Velocity Limits
        ssm_limit = self.compute_SSM_vel_lim(P_H, P_R, hr_versor, human_vel)
        pfl_limit = self.compute_PFL_vel_lim()
        if self.complete_debug or self.debug: print(colored('SSM - Velocity Limit: ', 'green'), ssm_limit)
        if self.complete_debug or self.debug: print(colored('PFL - Velocity Limit: ', 'green'), pfl_limit, '\n')

        # Compute Velocity Limit (Maximum Between SSM and PFL ISO/TS 15066 Velocity Limits)
        # DEBUG: vel_limit = max(ssm_limit, pfl_limit)
        vel_limit = ssm_limit

        # Compute Scaling Factor α
        scaling_factor = self.compute_alpha_optim(joint_states, desired_joint_velocity, old_joint_velocity, hr_versor, vel_limit)
        # scaling_factor = self.compute_alpha(joint_states, desired_joint_velocity, hr_versor, vel_limit)

        print('-'*100, '\n')
        print(colored('Human Point: ', 'green'), f'{P_H.x} {P_H.y} {P_H.z}')
        print(colored('Robot Point: ', 'green'), f'{P_R.x} {P_R.y} {P_R.z}')
        print(colored('HR Versor: ', 'green'), f'{hr_versor.x} {hr_versor.y} {hr_versor.z}')
        print(colored('\nISO/TS 15066 SSM Velocity Limit: ', 'green'), ssm_limit)
        print(colored('ISO/TS 15066 PFL Velocity Limit: ', 'green'), pfl_limit)
        print(colored('\nRobot Desired Velocity: ', 'green'), desired_joint_velocity)
        print(colored('Scaling Factor: ', 'green'), scaling_factor)
        print('-'*100, '\n')
        os.system('clear')

        # Return Scaled Joint Velocity
        return scaling_factor * desired_joint_velocity, scaling_factor
