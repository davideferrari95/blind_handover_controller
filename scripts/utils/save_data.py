#! /usr/bin/env python3

import rclpy, time, os, datetime
from rclpy.node import Node

# Package Path
from kinematic_wrapper import PACKAGE_PATH
if not os.path.exists(f'{PACKAGE_PATH}/data'): os.mkdir(f'{PACKAGE_PATH}/data')

# Messages & Services
from std_msgs.msg import Bool
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import JointState

class SaveData(Node):

    """ FT-Sensor Experiment - Save Data Node """

    # Initial FT Sensor and Joint States Data
    save_data, stop_save_data = False, False
    ft_sensor_data = Wrench()
    joint_states = JointState()
    joint_states.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joint_states.position, joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    def __init__(self, ros_rate:int=500):

        # Node Initialization
        super().__init__('Save_Data_Node')

        # ROS2 Rate
        self.ros_rate = ros_rate
        self.rate = self.create_rate(ros_rate)

        # ROS2 Subscriber Initialization
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states',       self.jointStatesCallback, 1)
        self.ft_sensor_subscriber   = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor',  self.FTSensorCallback, 1)
        self.save_data_subscriber   = self.create_subscription(Bool,       '/handover/save_data', self.SaveDataCallback, 1)

        # Create Save Files
        self.create_save_files(f'{PACKAGE_PATH}/data/', 'ft_sensor_data.csv', 'joint_states_data.csv')
        time.sleep(1)

    def create_save_files(self, path:str, ft_sensor_file:str, joint_states_file:str):

        """ Create Save Files """

        # Find Last Folder Number
        folders = sorted(
            [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))],
            key=lambda x: os.path.getctime(os.path.join(path, x)),
            reverse=True
        )

        if not folders: last_folder_number = 0
        else: last_folder_number = int(folders[0].split('_')[1].split(' ')[0])

        # Create Save Directory
        try_dir = f'Test {last_folder_number + 1} - [{datetime.datetime.now().strftime("%Y-%m-%d -- %H-%M-%S")}]'
        os.mkdir(f'{path}/{try_dir}')

        # Create Save Files
        self.FT_SENSOR_FILE, self.JOINT_STATES_FILE = f'{path}/{try_dir}/{ft_sensor_file}', f'{path}/{try_dir}/{joint_states_file}'

        # Write Header - Save Data FT Sensor & Joint States Files
        with open(self.FT_SENSOR_FILE, 'w') as file: file.write('fx,fy,fz,tx,ty,tz\n')
        with open(self.JOINT_STATES_FILE, 'w') as file: file.write(f'{self.joint_states.name[0]},{self.joint_states.name[1]},{self.joint_states.name[2]},{self.joint_states.name[3]},{self.joint_states.name[4]},{self.joint_states.name[5]}\n')

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

    def FTSensorCallback(self, data:Wrench):

        """ FT Sensor Callback """

        # Get FT Sensor Data
        self.ft_sensor_data = data

    def SaveDataCallback(self, data:Bool):

        """ Save Data Callback """

        # Get Save Data
        self.save_data = data.data
        if self.save_data == False: self.stop_save_data = True

    def main(self):

        """ Save Data Loop """

        with open(self.FT_SENSOR_FILE, 'a') as ft_sensor_file, open(self.JOINT_STATES_FILE, 'a') as joint_states_file:

            while rclpy.ok():

                # Spin Once
                rclpy.spin_once(self)

                # Break if Stop Save Data, Continue if Save Data is False                
                if self.stop_save_data: break
                if not self.save_data: continue

                # Append FT-Sensor Data
                ft_sensor_file.write(str(time.time()) + ',')
                ft_sensor_file.write(str(self.ft_sensor_data.force.x) + ',')
                ft_sensor_file.write(str(self.ft_sensor_data.force.y) + ',')
                ft_sensor_file.write(str(self.ft_sensor_data.force.z) + ',')
                ft_sensor_file.write(str(self.ft_sensor_data.torque.x) + ',')
                ft_sensor_file.write(str(self.ft_sensor_data.torque.y) + ',')
                ft_sensor_file.write(str(self.ft_sensor_data.torque.z) + '\n')

                # Append Joint States Data
                joint_states_file.write(str(self.joint_states.velocity[0]) + ',')
                joint_states_file.write(str(self.joint_states.velocity[1]) + ',')
                joint_states_file.write(str(self.joint_states.velocity[2]) + ',')
                joint_states_file.write(str(self.joint_states.velocity[3]) + ',')
                joint_states_file.write(str(self.joint_states.velocity[4]) + ',')
                joint_states_file.write(str(self.joint_states.velocity[5]) + '\n')

                # Rate Sleep
                self.rate.sleep()

        print('Data Saved')

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = SaveData()

    # Run Node
    node.main()
