import os
from typing import List
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

def create_pfl_controller_node(config:List[str]):

    # Python Node + Parameters + YAML Config File
    pfl_controller = Node(
        package='pfl_controller', executable='pfl_controller.py', name='pfl_controller',
        output='screen', emulate_tty=True, arguments=[('__log_level:=debug')],
        parameters=config,
    )

    return pfl_controller

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Config File Path
    config = os.path.join(get_package_share_directory('pfl_controller'), 'config','config.yaml')

    # Launch Description - Add Nodes
    launch_description.add_action(create_pfl_controller_node([config]))

    # Return Launch Description
    return launch_description
