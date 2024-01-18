from math import pi
from termcolor import colored
from typing import Union, List, Tuple, Callable

# Import Peter Corke Robotics Toolbox
import roboticstoolbox as rtb, numpy as np
from roboticstoolbox.robot import DHRobot, RevoluteDH
from roboticstoolbox.robot.IK import IKSolution
from roboticstoolbox.tools.types import ArrayLike, NDArray
from roboticstoolbox.tools.trajectory import Trajectory, jtraj, ctraj

# Import Sympy
import sympy as sym
from sympy import lambdify

# Import Spatial Math
from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation

# Import ROS2 Libraries
from geometry_msgs.msg import Pose

# Get Functions Path
import sys, pathlib
FUNCTIONS_PATH = f'{str(pathlib.Path(__file__).resolve().parents[2])}/functions'
sys.path.append(f'{str(pathlib.Path(__file__).resolve().parents[1])}')

class UR_Toolbox():

    def __init__(self, robot_parameters:dict, sym:bool=True, complete_debug:bool=False, debug:bool=False):

        """ Initialize Robot Toolbox """

        # Create Robot - Robotic Toolbox
        self.robot = self.create_robot(robot_parameters)

        # Create Symbolic Function (if `sym`) - Speed Up Computation
        # self.fkine, self.J, self.J_dot = self.load_symbolic_functions(robot_parameters) if sym else self.use_toolbox_functions()

        # Set Debug Flags
        self.complete_debug, self.debug = complete_debug, debug or complete_debug

    def test(self):

        import time, dill
        dill.settings['recurse'] = True
        ROBOT_PATH = f'{FUNCTIONS_PATH}/ur5e'
        # ROBOT_PATH = f'{FUNCTIONS_PATH}/ur10e'

        # Plan Trajectory
        traj = self.plan_trajectory([0, -pi/2, pi/2, -pi/2, -pi/2, 0], [0, -pi/2, pi/2, -pi/2, -pi/2, 0], duration=10, sampling_freq=500)

        from utils.kinematic_wrapper import Kinematic_Wrapper
        kin = Kinematic_Wrapper()
        self.fkine = kin.compute_direct_kinematic
        self.J = kin.compute_jacobian
        self.J_dot = kin.compute_jacobian_dot

        start = time.time()
        self.joint2cartesianTrajectory(traj)
        print(f'Elapsed Time - C++ Wrapper : {time.time() - start}')

        self.fkine = dill.load(open(f"{ROBOT_PATH}/FK_lambdified", "rb"))
        self.J = dill.load(open(f"{ROBOT_PATH}/J_lambdified", "rb"))

        start = time.time()
        self.joint2cartesianTrajectory(traj)
        print(f'Elapsed Time - Lambdified: {time.time() - start}')

        self.fkine = self.robot.fkine
        self.J= self.robot.jacob0

        start = time.time()
        self.joint2cartesianTrajectory(traj)
        print(f'Elapsed Time - Toolbox: {time.time() - start}')

        # self.fkine = dill.load(open(f"{ROBOT_PATH}/FK_lambdified_numpy", "rb"))
        # self.J = dill.load(open(f"{ROBOT_PATH}/J_lambdified_numpy", "rb"))

        # start = time.time()
        # self.joint2cartesianTrajectory(traj)
        # print(f'Elapsed Time - Lambdified Numpy: {time.time() - start}')

        # self.fkine = dill.load(open(f"{ROBOT_PATH}/FK_jit", "rb"))
        # self.J = dill.load(open(f"{ROBOT_PATH}/J_jit", "rb"))

        # start = time.time()
        # self.joint2cartesianTrajectory(traj)
        # print(f'Elapsed Time - Lambdified Jit: {time.time() - start}')

        # self.fkine = dill.load(open(f"{ROBOT_PATH}/FK_njit", "rb"))
        # self.J = dill.load(open(f"{ROBOT_PATH}/J_njit", "rb"))

        # start = time.time()
        # self.joint2cartesianTrajectory(traj)
        # print(f'Elapsed Time - Lambdified nJit: {time.time() - start}')

        # self.fkine = dill.load(open(f"{ROBOT_PATH}/FK_jit_numpy", "rb"))
        # self.J = dill.load(open(f"{ROBOT_PATH}/J_jit_numpy", "rb"))

        # start = time.time()
        # self.joint2cartesianTrajectory(traj)
        # print(f'Elapsed Time - Lambdified Jit numpy: {time.time() - start}')

        # self.fkine = dill.load(open(f"{ROBOT_PATH}/FK_njit_numpy", "rb"))
        # self.J = dill.load(open(f"{ROBOT_PATH}/J_njit_numpy", "rb"))

        # start = time.time()
        # self.joint2cartesianTrajectory(traj)
        # print(f'Elapsed Time - Lambdified nJit numpy: {time.time() - start}')

    def create_robot(self, robot_parameters:dict, symbolic:bool=False) -> DHRobot:

        """ Create Robot Model """

        # Robot Parameters
        robot_model = robot_parameters['robot']
        payload, reach, tcp_speed = robot_parameters['payload'], robot_parameters['reach'], robot_parameters['tcp_speed']
        stopping_time, stopping_distance, position_repeatability = robot_parameters['stopping_time'], robot_parameters['stopping_distance'], robot_parameters['position_repeatability']
        maximum_power, operating_power,operating_temperature = robot_parameters['maximum_power'], robot_parameters['operating_power'], robot_parameters['operating_temperature']
        ft_range, ft_precision, ft_accuracy = robot_parameters['ft_range'], robot_parameters['ft_precision'], robot_parameters['ft_accuracy']
        a, d, alpha, offset = robot_parameters['a'], robot_parameters['d'], robot_parameters['alpha'], robot_parameters['theta']
        mass, center_of_mass = robot_parameters['mass'], [robot_parameters['center_of_mass'][i:i+3] for i in range(0, len(robot_parameters['center_of_mass']), 3)]
        q_lim, qd_lim, qdd_lim = robot_parameters['q_limits'], robot_parameters['q_dot_limits'], robot_parameters['q_ddot_limits']

        # Create Robot Model
        return DHRobot(
            [RevoluteDH(a=a_n, d=d_n, alpha=alpha_n, offset=offset_n, q_lim=[-q_lim_n, q_lim_n], m=mass_n, r=center_of_mass_n.tolist()) 
             for a_n, d_n, alpha_n, offset_n, mass_n, center_of_mass_n, q_lim_n in zip(a, d, alpha, offset, mass, center_of_mass, q_lim)],
            name=robot_model, manufacturer='Universal Robot', symbolic=symbolic)

    def create_symbolic_functions(self, robot_parameters:dict, ROBOT_PATH) -> Tuple[Callable[[List[float]], np.ndarray], Callable[[List[float]], np.ndarray], Callable[[List[float]], np.ndarray]]:

        """ Create FK, J, J_dot Symbolic Functions """

        # Create Symbolic Robot Model
        symbolic_robot = self.create_robot(robot_parameters, symbolic=True)

        # Create Symbols
        q, q_dot = sym.symbols("q:6"), sym.symbols("q_dot:6")

        import dill, numba
        dill.settings['recurse'] = True

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Compute Jacobian, Jacobian Derivative and Forward Kinematic
        FK, J = symbolic_robot.fkine(q).A, symbolic_robot.jacob0(q)
        dill.dump(FK, open(f"{ROBOT_PATH}/FK", "wb")); dill.dump(J, open(f"{ROBOT_PATH}/J", "wb"))
        print("Done - FK, J")

        # Simplify FK, J
        FK_simplified, J_simplified = sym.simplify(symbolic_robot.fkine(q).A), sym.simplify(symbolic_robot.jacob0(q))
        dill.dump(FK_simplified, open(f"{ROBOT_PATH}/FK_simplified", "wb")); dill.dump(J_simplified, open(f"{ROBOT_PATH}/J_simplified", "wb"))
        print("Done - FK, J - Simplified")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Lambdify FK, J
        FK_lambdified, J_lambdified = lambdify([q], FK_simplified.tolist()), lambdify([q], J_simplified.tolist())
        dill.dump(FK_lambdified, open(f"{ROBOT_PATH}/FK_lambdified", "wb")); dill.dump(J_lambdified, open(f"{ROBOT_PATH}/J_lambdified", "wb"))
        print("Done - FK, J - Lambdified")

        # Lambdify FK, J - Numpy
        FK_lambdified_numpy, J_lambdified_numpy = lambdify([q], FK_simplified.tolist(), "numpy"), lambdify([q], J_simplified.tolist(), "numpy")
        dill.dump(FK_lambdified_numpy, open(f"{ROBOT_PATH}/FK_lambdified_numpy", "wb")); dill.dump(J_lambdified_numpy, open(f"{ROBOT_PATH}/J_lambdified_numpy", "wb"))
        print("Done - FK, J - Lambdified numpy")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Numba jit FK, J
        FK_jit, J_jit = numba.jit(FK_lambdified), numba.jit(J_lambdified)
        dill.dump(FK_jit, open(f"{ROBOT_PATH}/FK_jit", "wb")); dill.dump(J_jit, open(f"{ROBOT_PATH}/J_jit", "wb"))
        print("Done - FK, J - Numba jit")

        # Numba njit FK, J
        FK_njit, J_njit = numba.njit(FK_lambdified), numba.njit(J_lambdified)
        dill.dump(FK_njit, open(f"{ROBOT_PATH}/FK_njit", "wb")); dill.dump(J_njit, open(f"{ROBOT_PATH}/J_njit", "wb"))
        print("Done - FK, J - Numba njit")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Numba jit FK, J - Numpy
        FK_jit_numpy, J_jit_numpy = numba.jit(FK_lambdified_numpy), numba.jit(J_lambdified_numpy)
        dill.dump(FK_jit_numpy, open(f"{ROBOT_PATH}/FK_jit_numpy", "wb")); dill.dump(J_jit_numpy, open(f"{ROBOT_PATH}/J_jit_numpy", "wb"))
        print("Done - FK, J - Numba jit numpy")

        # Numba njit FK, J - Numpy
        FK_njit_numpy, J_njit_numpy = numba.njit(FK_lambdified_numpy), numba.njit(J_lambdified_numpy)
        dill.dump(FK_njit_numpy, open(f"{ROBOT_PATH}/FK_njit_numpy", "wb")); dill.dump(J_njit_numpy, open(f"{ROBOT_PATH}/J_njit_numpy", "wb"))
        print("Done - FK, J - Numba njit numpy")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Compute Jacobian Derivative
        J_dot = sym.diff(J, [q])
        dill.dump(J_dot, open(f"{ROBOT_PATH}/J_dot", "wb"))
        print("Done - J_dot")

        # Simplify Jacobian Derivative
        J_dot_simplified = sym.simplify(J_dot)
        dill.dump(J_dot_simplified, open(f"{ROBOT_PATH}/J_dot_simplified", "wb"))
        print("Done - J_dot - Simplified")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Lambdify Jacobian Derivative
        J_dot_lambdified = lambdify([q], J_dot.tolist())
        dill.dump(J_dot_lambdified, open(f"{ROBOT_PATH}/J_dot_lambdified", "wb"))
        print("Done - J_dot - Lambdified")

        # Lambdify Jacobian Derivative - Numpy
        J_dot_lambdified_numpy = lambdify([q], J_dot.tolist(), "numpy")
        dill.dump(J_dot_lambdified_numpy, open(f"{ROBOT_PATH}/J_dot_lambdified_numpy", "wb"))
        print("Done - J_dot - Lambdified numpy")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Numba jit Jacobian Derivative
        J_dot_jit = numba.jit(J_dot_lambdified)
        dill.dump(J_dot_jit, open(f"{ROBOT_PATH}/J_dot_jit", "wb"))
        print("Done - J_dot - Numba jit")

        # Numba njit Jacobian Derivative
        J_dot_njit = numba.njit(J_dot_lambdified)
        dill.dump(J_dot_njit, open(f"{ROBOT_PATH}/J_dot_njit", "wb"))
        print("Done - J_dot - Numba njit")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Numba jit Jacobian Derivative - Numpy
        J_dot_jit_numpy = numba.jit(J_dot_lambdified)
        dill.dump(J_dot_jit_numpy, open(f"{ROBOT_PATH}/J_dot_jit_numpy", "wb"))
        print("Done - J_dot - Numba jit numpy")

        # Numba njit Jacobian Derivative - Numpy
        J_dot_njit_numpy = numba.njit(J_dot_lambdified_numpy)
        dill.dump(J_dot_njit_numpy, open(f"{ROBOT_PATH}/J_dot_njit_numpy", "wb"))
        print("Done - J_dot - Numba njit numpy")

        # -------------------------------------------------------------------------------------------------------------------------------------------------#

        # Return Symbolic Functions (lambdify) - If `sym`
        return FK_lambdified_numpy, J_lambdified_numpy, J_dot_lambdified_numpy

    def load_symbolic_functions(self, robot_parameters:dict) -> Tuple[Callable[[List[float]], np.ndarray], Callable[[List[float]], np.ndarray], Callable[[List[float]], np.ndarray]]:

        """ Load FK, J, J_dot Symbolic Functions """

        # Get Robot Path
        ROBOT_PATH = f'{FUNCTIONS_PATH}/{robot_parameters["robot"]}'

        # if sym: self.sym_fkine, self.sym_J = self.create_symbolic_functions(robot_model)
        # TODO: if not exist sym functions self.sym_fkine, self.sym_J, self.sym_J_dot = self.create_symbolic_functions(robot_model)
        self.create_symbolic_functions(robot_parameters, ROBOT_PATH)

        import dill
        FK = dill.load(open(f"{ROBOT_PATH}/FK", "rb"))
        J = dill.load(open(f"{ROBOT_PATH}/J", "rb"))
        print("Done - FK, J")

        FK_simplified = dill.load(open(f"{ROBOT_PATH}/FK_simplified", "rb"))
        J_simplified = dill.load(open(f"{ROBOT_PATH}/J_simplified", "rb"))
        print("Done - FK, J - Simplified")

        FK_lambdified = dill.load(open(f"{ROBOT_PATH}/FK_lambdified", "rb"))
        J_lambdified = dill.load(open(f"{ROBOT_PATH}/J_lambdified", "rb"))
        print("Done - FK, J - Lambdified")

        FK_njit = dill.load(open(f"{ROBOT_PATH}/FK_njit", "rb"))
        J_njit = dill.load(open(f"{ROBOT_PATH}/J_njit", "rb"))
        print("Done - FK, J - Njit")

        # Return Symbolic Functions (lambdify) - If `sym`
        # return FK_njit, J_njit
        return SE3(FK_lambdified), np.asarray(J_lambdified)

    def use_toolbox_functions(self) -> Tuple[Callable[[List[float]], np.ndarray], Callable[[List[float]], np.ndarray], Callable[[List[float]], np.ndarray]]:

        """ Use Robotics Toolbox FK, J, J_dot Functions """

        # Return Toolbox Functions
        return self.robot.fkine, self.robot.jacob0, self.robot.jacob0_dot

    def ForwardKinematic(self, joint_positions:Union[List[float], np.ndarray], end:str=None, start:str=None) -> SE3:

        """ Forward Kinematics Using Peter Corke Robotics Toolbox """

        # Convert Joint Positions to NumPy Array
        if type(joint_positions) is list: joint_positions = np.array(joint_positions)

        # Type Assertion
        assert type(joint_positions) in [List[float], np.ndarray], f"Joint Positions must be a ArrayLike | {type(joint_positions)} given"
        assert len(joint_positions) == 6, f"Joint Positions Length must be 6 | {len(joint_positions)} given \nJoint Positions:\n{joint_positions}"

        # Return Forward Kinematic
        return self.fkine(joint_positions)
        # return self.robot.fkine(joint_positions, end=end, start=start)
        return SE3(self.sym_fkine(joint_positions))
        if self.sym_fkine is not None: return SE3(self.sym_fkine(joint_positions))
        else: return self.robot.fkine(joint_positions, end=end, start=start)

    def InverseKinematic(self, pose:Union[Pose, NDArray, SE3]) -> IKSolution:

        """ Inverse Kinematics Using Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(pose) in [Pose, np.ndarray, SE3], f"Pose must be a Pose, NumPy Array or SE3 | {type(pose)} given"

        # Convert Pose to SE3 Matrix
        if type(pose) is Pose: pose = self.pose2matrix(pose)
        if type(pose) is np.ndarray: pose = self.array2matrix(pose)

        return self.robot.ikine_NR(pose, q0=[0, -pi/2, pi/2, -pi/2, -pi/2, 0])

    def Jacobian(self, joint_positions:ArrayLike) -> np.ndarray:

        """ Get Jacobian Matrix """

        return self.J(joint_positions)
        # return compute_ur10e_jacobian(joint_positions)
        # return self.robot.jacob0(np.asarray(joint_positions))
        return np.asarray(self.sym_J(joint_positions))
        if self.sym_J is not None: return np.asarray(self.sym_J(joint_positions))
        else: return self.robot.jacob0(np.asarray(joint_positions))

    def JacobianDot(self, joint_positions:ArrayLike, joint_velocities:ArrayLike) -> np.ndarray:

        """ Get Jacobian Derivative Matrix """

        return self.J_dot(joint_positions, joint_velocities)
        return self.robot.jacob0_dot(np.asarray(joint_positions), np.asarray(joint_velocities))

    def JacobianInverse(self, joint_positions:ArrayLike) -> np.ndarray:

        """ Get Jacobian Inverse Matrix """
        return np.linalg.inv(self.Jacobian(joint_positions))

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

    def plan_trajectory(self, start:Union[List[float], Pose, SE3], end:Union[List[float], Pose, SE3], duration:float=10.0, sampling_freq:int=500) -> Trajectory:

        """ Plan Trajectory with Peter Corke Robotics Toolbox """

        # Type Assertion
        assert type(sampling_freq) is int, f"Sampling Frequency must be an Integer | {type(sampling_freq)} given"
        assert type(duration) in [int, float], f"Duration must be a Int or Float | {type(duration)} given"

        # Convert Geometry Pose, SE3 Matrix to Joint Position Array
        start_joint_pos = self.InverseKinematic(start).q if type(start) in [Pose, SE3] else start
        end_joint_pos   = self.InverseKinematic(end).q   if type(end)   in [Pose, SE3] else end

        # Assertion
        assert len(start_joint_pos) == 6, f"Start Joint Positions Length must be 6 | {len(start_joint_pos)} given"
        assert len(end_joint_pos) == 6, f"End Joint Positions Length must be 6 | {len(end_joint_pos)} given"
        assert duration > 0, f"Duration must be greater than 0 | {duration} given"
        assert sampling_freq > 0, f"Sampling Frequency must be greater than 0 | {sampling_freq} given"

        # Create Time Vector (steps = duration * sampling_freq)
        time_vector = np.linspace(0, duration, duration * sampling_freq)

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

    def joint2cartesianTrajectory(self, joint_trajectory:Trajectory) -> Tuple[List[SE3], np.ndarray, np.ndarray]:

        """ Convert Joint Trajectory to Cartesian Trajectory """

        # Type Assertion
        assert type(joint_trajectory) is Trajectory, f"Joint Trajectory must be a Trajectory | {type(joint_trajectory)} given"

        # Initialize Cartesian Trajectory
        cartesian_positions, cartesian_velocities, cartesian_accelerations = [], [], []

        # Print Joint Trajectory Shape
        if self.complete_debug: print(f'Joint Trajectory Shape | Positions: {joint_trajectory.q.shape} | Velocities: {joint_trajectory.qd.shape} | Accelerations: {joint_trajectory.qdd.shape}')

        # Convert Joint to Cartesian Positions
        for q, q_dot, q_ddot in zip(joint_trajectory.q, joint_trajectory.qd, joint_trajectory.qdd):

            # Convert Joint Position to Cartesian Position (x = ForwardKinematic(q))
            x = self.ForwardKinematic(q)
            cartesian_positions.append(x)

            # Convert Joint Velocity to Cartesian Velocity (x_dot = Jacobian(q) * q_dot)
            x_dot = self.Jacobian(q) @ q_dot
            cartesian_velocities.append(x_dot)

            # Convert Joint Acceleration to Cartesian Acceleration (x_ddot = Jacobian(q) * q_ddot + JacobianDot(q, q_dot) * q_dot)
            x_ddot = self.Jacobian(q) @ q_ddot + self.JacobianDot(q, q_dot) @ q_dot
            cartesian_accelerations.append(x_ddot)

        if self.complete_debug: print(f'Cartesian Trajectory Shape | Positions: {np.array(cartesian_positions).shape} | Velocities: {np.array(cartesian_velocities).shape} | Accelerations: {np.array(cartesian_accelerations).shape}\n')
        if self.debug: print (colored('Cartesian Positions:', 'green'),     f'\n\n {np.array(cartesian_positions)}\n')
        if self.debug: print (colored('Cartesian Velocities:', 'green'),    f'\n\n {np.array(cartesian_velocities)}\n')
        if self.debug: print (colored('Cartesian Accelerations:', 'green'), f'\n\n {np.array(cartesian_accelerations)}\n')

        return cartesian_positions, np.array(cartesian_velocities), np.array(cartesian_accelerations)

    def pose2array(self, pose:Pose) -> np.ndarray:

        """ Convert Pose to Numpy Array (6x1) """

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

        """ Convert SE3 Transformation Matrix to NumPy Array (6x1) """

        # Type Assertion
        assert type(matrix) is SE3, f"Matrix must be a SE3 | {type(matrix)} given | {matrix}"

        # Convert Rotation Matrix to Rotation
        rotation:Rotation = Rotation.from_matrix(matrix.R)

        # Create Numpy Pose Array
        numpy_array = np.zeros((6,))
        numpy_array[:3] = matrix.t
        numpy_array[3:] = rotation.as_rotvec()

        return numpy_array

    def matrix2numpy(self, matrix:SE3) -> np.ndarray:

        """ Convert SE3 Transformation Matrix to NumPy Matrix (4x4) """

        # Type Assertion
        assert type(matrix) is SE3, f"Matrix must be a SE3 | {type(matrix)} given | {matrix}"

        # Convert to Numpy
        return matrix.A

    def array2matrix(self, array:np.ndarray) -> SE3:

        """ Convert NumPy Array to SE3 Transformation Matrix """

        # Type Assertion
        assert type(array) is np.ndarray, f"Array must be a NumPy Array | {type(array)} given | {array}"
        assert len(array) == 6, f"Array Length must be 6 | {len(array)} given | {array}"

        # Create Transformation Matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = array[:3]
        transformation_matrix[:3, :3] = Rotation.from_rotvec(array[3:]).as_matrix()

        # Convert to SE3 Matrix
        return SE3(transformation_matrix)
