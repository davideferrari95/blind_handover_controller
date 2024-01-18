import ctypes
import numpy as np

# Get Lib Path
from pathlib import Path
LIB_PATH = f'{str(Path(__file__).resolve().parents[2])}/lib'

class Kinematic_Wrapper:

    def __init__(self, robot_name:str='ur10e'):

        if robot_name.lower() == 'ur3':

            # Load UR3 Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur3_kinematic/compute_UR3_jacobian.so').compute_UR3_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur3_kinematic/compute_UR3_jacobian_dot_dq.so').compute_UR3_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur3_kinematic/compute_UR3_direct_kinematic.so').compute_UR3_direct_kinematic

        elif robot_name.lower() == 'ur5':

            # Load UR5 Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur5_kinematic/compute_UR5_jacobian.so').compute_UR5_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur5_kinematic/compute_UR5_jacobian_dot_dq.so').compute_UR5_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur5_kinematic/compute_UR5_direct_kinematic.so').compute_UR5_direct_kinematic

        elif robot_name.lower() == 'ur10':

            # Load UR10 Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur10_kinematic/compute_UR10_jacobian.so').compute_UR10_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur10_kinematic/compute_UR10_jacobian_dot_dq.so').compute_UR10_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur10_kinematic/compute_UR10_direct_kinematic.so').compute_UR10_direct_kinematic

        elif robot_name.lower() == 'ur3e':

            # Load UR3e Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur3e_kinematic/compute_UR3e_jacobian.so').compute_UR3e_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur3e_kinematic/compute_UR3e_jacobian_dot_dq.so').compute_UR3e_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur3e_kinematic/compute_UR3e_direct_kinematic.so').compute_UR3e_direct_kinematic

        elif robot_name.lower() == 'ur5e':

            # Load UR5e Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur5e_kinematic/compute_UR5e_jacobian.so').compute_UR5e_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur5e_kinematic/compute_UR5e_jacobian_dot_dq.so').compute_UR5e_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur5e_kinematic/compute_UR5e_direct_kinematic.so').compute_UR5e_direct_kinematic

        elif robot_name.lower() == 'ur10e':

            # Load UR5e Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur10e_kinematic/compute_UR10e_jacobian.so').compute_UR10e_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur10e_kinematic/compute_UR10e_jacobian_dot_dq.so').compute_UR10e_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur10e_kinematic/compute_UR10e_direct_kinematic.so').compute_UR10e_direct_kinematic

        elif robot_name.lower() == 'ur16e':

            # Load UR16e Kinematic Shared Libraries into Ctypes
            self.jacobian         = ctypes.CDLL(f'{LIB_PATH}/ur16e_kinematic/compute_UR16e_jacobian.so').compute_UR16e_jacobian
            self.jacobian_dot     = ctypes.CDLL(f'{LIB_PATH}/ur16e_kinematic/compute_UR16e_jacobian_dot_dq.so').compute_UR16e_jacobian_dot_dq
            self.direct_kinematic = ctypes.CDLL(f'{LIB_PATH}/ur16e_kinematic/compute_UR16e_direct_kinematic.so').compute_UR16e_direct_kinematic

        else: raise ValueError(f'Robot Name Must be either "ur3", "ur5", "ur10", "ur3e", "ur5e", "ur10e" or "ur16e", given: {robot_name}')

        # Define the Argument and Return types for the function
        self.jacobian.argtypes = [np.ctypeslib.ndpointer(float), np.ctypeslib.ndpointer(float)]
        self.jacobian_dot.argtypes = [np.ctypeslib.ndpointer(float), np.ctypeslib.ndpointer(float), np.ctypeslib.ndpointer(float)]
        self.direct_kinematic.argtypes = [np.ctypeslib.ndpointer(float), np.ctypeslib.ndpointer(float)]

    def compute_jacobian(self, q):

        """ Compute UR5e / UR10e Jacobian """

        assert isinstance(q, np.ndarray), f'Joint States Must be a Numpy Array, given: {type(q)}'
        assert len(q) == 6, f'Joint States Must be a 6-Dimensional Vector, given: {len(q)}'

        result = np.zeros((36,))
        self.jacobian(result, np.array(q))
        return result.reshape(6,6)

    def compute_jacobian_dot(self, q, dq):

        """ Compute UR5e / UR10e Jacobian Derivative """

        assert isinstance(q, np.ndarray), f'Joint States Must be a Numpy Array, given: {type(q)}'
        assert isinstance(dq, np.ndarray), f'Joint States Must be a Numpy Array, given: {type(dq)}'
        assert len(q) == 6, f'Joint States Must be a 6-Dimensional Vector, given: {len(q)}'
        assert len(dq) == 6, f'Joint Vel Must be a 6-Dimensional Vector, given: {len(dq)}'

        result = np.zeros((6,))
        self.jacobian_dot(result, q, dq)
        return result.reshape(6,)

    def compute_direct_kinematic(self, q):

        """ Compute UR5e / UR10e Direct Kinematic """

        assert isinstance(q, np.ndarray), f'Joint States Must be a Numpy Array, given: {type(q)}'
        assert len(q) == 6, f'Joint States Must be a 6-Dimensional Vector, given: {len(q)}'

        result = np.zeros((16,))
        self.direct_kinematic(result, np.array(q))
        return result.reshape(4,4)

    def test():

        kin = Kinematic_Wrapper('ur5e')
        kin = Kinematic_Wrapper('ur10e')

        # Call the C++ function
        print('FK:\n\n',    kin.compute_direct_kinematic(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.double)), '\n')
        print('J:\n\n',     kin.compute_jacobian(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.double)), '\n')
        print('J_dot:\n\n', kin.compute_jacobian_dot(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.double), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.double)), '\n')
