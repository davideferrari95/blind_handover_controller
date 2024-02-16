from launch import LaunchDescription
from launch_ros.actions import Node

def create_experiment_node():

    # Experiment Node
    experiment = Node(
        package='handover_controller', executable='experiment.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
    )

    return experiment

def create_network_node():

    # Network Node
    network = Node(
        package='handover_controller', executable='ft_network.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
    )

    return network

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Launch Description - Add Nodes
    launch_description.add_action(create_experiment_node())
    launch_description.add_action(create_network_node())

    # Return Launch Description
    return launch_description
