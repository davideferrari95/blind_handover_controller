#! /usr/bin/env python3

import rclpy, time, os, datetime, threading
from rclpy.node import Node
from typing import List

# Get Data Path
from pathlib import Path
PACKAGE_PATH = f'{str(Path(__file__).resolve().parents[2])}'
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

    # Data Lists
    joint_states_data_list:List[JointState] = []
    ft_sensor_data_list:List[Wrench] = []

    def __init__(self, ros_rate:int=500):

        # Node Initialization
        super().__init__('save_data_node')

        # ROS2 Rate
        self.ros_rate = ros_rate
        self.rate = self.create_rate(ros_rate)

        # Spin in a separate thread - for ROS2 Rate
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self, ), daemon=True)
        self.spin_thread.start()

        # ROS2 Subscriber Initialization
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states',       self.jointStatesCallback, 1)
        self.ft_sensor_subscriber   = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor',  self.FTSensorCallback, 1)
        self.save_data_subscriber   = self.create_subscription(Bool,       '/handover/save_data', self.SaveDataCallback, 1)

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
        else: last_folder_number = int(folders[0].split(' ')[1].split(' ')[0])

        # Create Save Directory
        try_dir = f'Test {last_folder_number + 1} - [{datetime.datetime.now().strftime("%Y-%m-%d -- %H-%M-%S")}]'
        os.mkdir(f'{path}/{try_dir}')

        # Create Save Files
        self.FT_SENSOR_FILE, self.JOINT_STATES_FILE = f'{path}/{try_dir}/{ft_sensor_file}', f'{path}/{try_dir}/{joint_states_file}'

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

    def save(self):

        """ Save Data """

        print('Saving Data...')

        # Create Save Files
        self.create_save_files(f'{PACKAGE_PATH}/data/', 'ft_sensor_data.csv', 'joint_states_data.csv')

        # Save Data - FT Sensor
        with open(self.FT_SENSOR_FILE, 'w') as file:

            # Write Header
            file.write('fx,fy,fz,tx,ty,tz\n')

            for data in self.ft_sensor_data_list:

                # Append FT Sensor Data - Force and Torque
                file.write(str(data.force.x) + ',' + str(data.force.y) + ',' + str(data.force.z) + ',' + str(data.torque.x) + ',' + str(data.torque.y) + ',' + str(data.torque.z) + '\n')

        # Save Data - Joint States
        with open(self.JOINT_STATES_FILE, 'w') as file:

            # Write Header
            file.write(f'{self.joint_states.name[0]},{self.joint_states.name[1]},{self.joint_states.name[2]},{self.joint_states.name[3]},{self.joint_states.name[4]},{self.joint_states.name[5]}\n')

            for data in self.joint_states_data_list:

                # Append Joint States Data - Velocity
                file.write(str(data.velocity[0]) + ',' + str(data.velocity[1]) + ',' + str(data.velocity[2]) + ',' + str(data.velocity[3]) + ',' + str(data.velocity[4]) + ',' + str(data.velocity[5]) + '\n')

    def main(self):

        """ Save Data Loop """

        while rclpy.ok():

            # Spin Once
            # rclpy.spin_once(self, timeout_sec=1.0/float(self.ros_rate))

            # Break if Stop Save Data, Continue if Save Data is False                
            if self.stop_save_data: break
            if not self.save_data: continue

            # Append Joint States and FT-Sensor Data
            self.get_logger().info('Collecting Data...', throttle_duration_sec=2.0)
            self.joint_states_data_list.append(self.joint_states)
            self.ft_sensor_data_list.append(self.ft_sensor_data)

            # Rate Sleep
            self.rate.sleep()

        self.save()
        print('Data Saved\n')

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = SaveData(500)

    # Run Node
    node.main()
