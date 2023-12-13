# PFL Controller

This package contains a Power and Force Limit Controller for the UR10e Manipulator in ROS 2.

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

        conda create -n pfl_env python=3.8.10
        conda activate pfl_env

- Install Python Requirements:

        pip install -r ../path/to/this/repo/requirements.txt

## Launch Instructions

- Activate the `conda` environment:

        conda activate pfl_env

- Remember to source ROS2 and export the Domain ID (if not in `~/.bashrc`):

        source /opt/ros/foxy/setup.bash
        . ~/colcon_ws/install/setup.bash
        export ROS_DOMAIN_ID=10

### Launch PFL Controller

- Launch `pfl_controller`:

        ros2 launch pfl_controller pfl_controller.launch.py

## Maintainers

- Davide Ferrari
