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
