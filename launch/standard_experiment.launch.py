from launch import LaunchDescription
from launch_ros.actions import Node

def create_experiment_node():

    # Experiment Node
    experiment = Node(
        package='handover_controller', executable='standard_experiment.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
    )

    return experiment

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Launch Description - Add Nodes
    launch_description.add_action(create_experiment_node())

    # Return Launch Description
    return launch_description
