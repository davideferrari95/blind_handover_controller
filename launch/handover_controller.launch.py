import os
from typing import List
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def create_handover_controller_node(context):

    # Python Node - Parameters
    handover_controller_parameters = {
        'use_feedback_velocity': LaunchConfiguration('use_feedback_velocity'),
        'complete_debug': LaunchConfiguration('complete_debug'),
        'debug': LaunchConfiguration('debug'),
        'sim': LaunchConfiguration('sim'),
        'human_radius': LaunchConfiguration('human_radius'),
        'robot': LaunchConfiguration('robot'),
    }

    # Config File Path
    config = os.path.join(get_package_share_directory('handover_controller'), 'config','config.yaml')

    # Robot Config File Path
    robot = LaunchConfiguration('robot').perform(context)
    if robot in ['ur5e','ur10e']: robot_config = os.path.join(get_package_share_directory('handover_controller'), 'config',f'{robot}.yaml')
    else: raise ValueError(f'No config file for robot name: {robot}')

    # Python Node + Parameters + YAML Config File
    handover_controller = Node(
        package='handover_controller', executable='handover_controller.py', #name='handover_controller',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
        parameters=[handover_controller_parameters, config, robot_config],
    )

    return [handover_controller]

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Arguments
    use_feedback_velocity_arg = DeclareLaunchArgument('use_feedback_velocity', default_value='true')
    complete_debug_arg = DeclareLaunchArgument('complete_debug', default_value='false')
    human_radius_arg = DeclareLaunchArgument('human_radius', default_value='0.1')
    debug_arg = DeclareLaunchArgument('debug', default_value='false')
    robot_arg = DeclareLaunchArgument('robot', default_value='ur5e')
    sim_arg = DeclareLaunchArgument('sim', default_value='false')
    launch_description.add_action(use_feedback_velocity_arg)
    launch_description.add_action(complete_debug_arg)
    launch_description.add_action(human_radius_arg)
    launch_description.add_action(debug_arg)
    launch_description.add_action(robot_arg)
    launch_description.add_action(sim_arg)

    # Launch Description - Add Nodes
    launch_description.add_action(OpaqueFunction(function = create_handover_controller_node))

    # Return Launch Description
    return launch_description
