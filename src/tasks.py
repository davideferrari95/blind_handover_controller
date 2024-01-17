from invoke import task

# Get Lib Path
from pathlib import Path
LIB_PATH = f'{str(Path(__file__).resolve().parents[1])}/lib'

EIGEN_PATH = '/usr/include/eigen3'

LIB_NAMES = [
    f'{LIB_PATH}/ur5e_kinematic/compute_UR5e_direct_kinematic.so',
    f'{LIB_PATH}/ur5e_kinematic/compute_UR5e_jacobian.so',
    f'{LIB_PATH}/ur5e_kinematic/compute_UR5e_jacobian_dot_dq.so',
    f'{LIB_PATH}/ur10e_kinematic/compute_UR10e_direct_kinematic.so',
    f'{LIB_PATH}/ur10e_kinematic/compute_UR10e_jacobian.so',
    f'{LIB_PATH}/ur10e_kinematic/compute_UR10e_jacobian_dot_dq.so',
]

SOURCES = [
    'ur5e_kinematic/compute_UR5e_direct_kinematic.cpp',
    'ur5e_kinematic/compute_UR5e_jacobian.cpp',
    'ur5e_kinematic/compute_UR5e_jacobian_dot_dq.cpp',
    'ur10e_kinematic/compute_UR10e_direct_kinematic.cpp',
    'ur10e_kinematic/compute_UR10e_jacobian.cpp',
    'ur10e_kinematic/compute_UR10e_jacobian_dot_dq.cpp',
]

@task
def build(ctx):

    for i in range(len(LIB_NAMES)):
        ctx.run(f"g++ -I{EIGEN_PATH} -shared -o {LIB_NAMES[i]} -fPIC {SOURCES[i]}")
