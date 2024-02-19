from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def create_ft_load_node():

    # FT Sensor Experiment Node
    ft_load_experiment = Node(
        package='handover_controller', executable='ft_load_experiment.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
    )

    return ft_load_experiment

def create_save_data_node():

    # Python Node - Parameters
    save_data_parameters = {
        'disturbances': LaunchConfiguration('disturbances'),
    }

    # Save Data Node
    save_data = Node(
        package='handover_controller', executable='save_data.py',
        output='screen', emulate_tty=True, output_format='{line}', arguments=[('__log_level:=info')],
        parameters=[save_data_parameters],
    )

    return save_data

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Arguments
    disturbances_arg = DeclareLaunchArgument('disturbances', default_value='false')
    launch_description.add_action(disturbances_arg)

    # Launch Description - Add Nodes
    launch_description.add_action(create_ft_load_node())
    launch_description.add_action(create_save_data_node())

    # Return Launch Description
    return launch_description
