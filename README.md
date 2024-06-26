# Blind Handover Controller

This package contains a Blind Handover Controller with the following features:

- Handover Controller with UR e-Series Robot and Robotiq 2F-85 Gripper.
- Trajectory Planning via Peter Corke Robotics Toolbox and Spline Interpolation.
- Trajectory Following with Admittance Control using the UR e-Series integrated FT-Sensor.
- Safety Limit on Desired Velocity computing ISO-TS 15066 SSM (Speed and Separation Monitoring) and PFL (Power and Force Limit).
- Trained Neural Network for FT-Load Estimation and Automatic Recognition of the Handover Phase for Object Transfer (Gripper Opening).

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

- Install `vrpn_client_ros2`:

        sudo apt install ros-foxy-vrpn
        sudo apt install ros-foxy-vrpn-mocap

- Install `ur_rtde_controller` following the instructions in the [README](https://github.com/ARSControl/ur_rtde_controller/tree/foxy).

        git clone -b foxy https://github.com/ARSControl/ur_rtde_controller.git

## Build New Robot Kinematic Libraries

The Kinematics Libraries are already available for the following robots:

- CB3 Series (UR3, UR5, UR10)
- e-Series (UR3e, UR5e, UR10e, UR16e)

If you want to add a new robot, follow these steps:

- Install `invoke`:

        pip install invoke

- Create the Kinematic Source Files in `src/robot_name_kinematic`:

  - `compute_robot_name_direct_kinematic.cpp`
  - `compute_robot_name_jacobian.cpp`
  - `compute_robot_name_jacobian_dot_dq.cpp`

- Use <https://github.com/ARSControl/robot_kinematic> to generate the Robot Kinematic Source Files (Little Manual Edit is Needed).

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

- Launch `vrpn_client_ros2`:

        ros2 launch vrpn_mocap client.launch.yaml server:=192.168.2.50

- Launch `handover_controller`:

        ros2 launch handover_controller handover_controller.launch.py

#### Experiments

- Launch Alexa :

        conda activate alexa_conversation_env
        ros2 launch alexa_conversation skill_backend.launch.py 

- Launch UR-RTDE Controller

        ros2 launch ur_rtde_controller rtde_controller.launch.py ROBOT_IP:=192.168.2.30 enable_gripper:=true

- Launch main `experiment`:

        ros2 launch vrpn_mocap client.launch.yaml server:=192.168.2.50
        ros2 launch handover_controller handover_controller.launch.py 
        ros2 launch handover_controller experiment.launch.py

- Launch comparative `standard_experiment`:

        ros2 launch vrpn_mocap client.launch.yaml server:=192.168.2.50
        ros2 launch handover_controller handover_controller.launch.py use_admittance:=False
        ros2 launch handover_controller experiment.launch.py use_network:=False

#### Training FT-Load Neural Network

- Launch `ft_load_experiment` to collect data from the FT Sensor for the Dataset:

        ros2 launch handover_controller handover_controller.launch.py
        ros2 launch handover_controller ft_experiment.launch.py

- Train the Neural Network with the collected data:

        python scripts/ft_load_network/train_network.py

## Maintainers

- Davide Ferrari
