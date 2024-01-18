# Handover Controller

This package contains an Handover Controller with Admittance and Power and Force Limit Controllers for the UR10e Manipulator in ROS 2.

## Requirements

- Ubuntu 20.04+
- Python 3.8.10
- ROS2 Foxy
- Anaconda / Miniconda

## Installation

### Prerequisites

- Install ROS2 Foxy: [Ubuntu Guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

        sudo apt install ros-foxy-desktop python3-argcomplete

- Install `miniconda`: [Official Guide](https://docs.conda.io/en/main/miniconda.html)

- Create a `conda` environment with `python=3.8.10`:

        conda create -n handover_env python=3.8.10
        conda activate handover_env

- Install Python Requirements:

        pip install -r ../path/to/this/repo/requirements.txt

## Build New Robot Kinematic Libraries

- Create the Kinematic Source Files in `src/robot_name_kinematic`:

  - `compute_robot_name_direct_kinematic.cpp`
  - `compute_robot_name_jacobian.cpp`
  - `compute_robot_name_jacobian_dot_dq.cpp`

- Use https://github.com/ARSControl/robot_kinematic to generate the Robot Kinematic Source Files (Little Manual Edit is Needed).

- Edit the `src/tasks.py` build file adding the new source and destination path.

- Build the Robot Kinematic Library:

        cd path/to/package/src
        invoke build

- Add the new libraries to the `scripts/utils/kinematic_wrapper` script file.

## Launch Instructions

- Activate the `conda` environment:

        conda activate handover_env

- Remember to source ROS2 and export the Domain ID (if not in `~/.bashrc`):

        source /opt/ros/foxy/setup.bash
        . ~/colcon_ws/install/setup.bash
        export ROS_DOMAIN_ID=10

### Launch Handover Controller

- Launch `handover_controller`:

        ros2 launch handover_controller handover_controller.launch.py

## Maintainers

- Davide Ferrari
