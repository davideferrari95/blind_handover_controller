from launch import LaunchDescription
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def create_experiment_node():

    # Python Node - Parameters
    experiment_parameters = {
        'use_network': LaunchConfiguration('use_network'),
    }

    # Experiment Node
    experiment = Node(
        package='handover_controller', executable='experiment.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
        parameters=[experiment_parameters],
    )

    return experiment

def create_network_node():

    # Use Network Argument
    use_network = LaunchConfiguration('use_network'),

    # Network Node
    network = Node(
        package='handover_controller', executable='ft_network.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
        condition=IfCondition(use_network)
    )

    return network

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Arguments
    use_network_arg = DeclareLaunchArgument('use_network', default_value='true')
    launch_description.add_action(use_network_arg)

    # Launch Description - Add Nodes
    launch_description.add_action(create_experiment_node())
    launch_description.add_action(create_network_node())

    # Return Launch Description
    return launch_description
