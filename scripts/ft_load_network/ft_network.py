#! /usr/bin/env python3

import rclpy, threading, time
from rclpy.node import Node
from typing import List

# Import ROS Messages
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench

# Import PyTorch Lightning NN Model
from train_network import torch, SEQUENCE_LENGTH
from train_network import FeedforwardModel, LSTMModel, CNNModel
from process_dataset import PACKAGE_PATH
from pl_utils import load_hyperparameters

class GripperControlNode(Node):

    """ Gripper Control Node Class """

    # Initial FT Sensor and Joint States Data
    joint_states, ft_sensor_data = JointState(), Wrench()
    joint_states.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joint_states.position, joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Data Lists
    joint_states_data_list:List[JointState] = []
    ft_sensor_data_list:List[Wrench] = []

    def __init__(self, ros_rate:int=500):

        # Node Initialization
        super().__init__('gripper_control_node')

        # ROS2 Rate
        self.ros_rate = ros_rate
        self.rate = self.create_rate(ros_rate)

        # Spin in a separate thread - for ROS2 Rate
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self, ), daemon=True)
        self.spin_thread.start()

        # Load Hyperparameters
        input_size, hidden_size, output_size, num_layers, learning_rate = load_hyperparameters(f'{PACKAGE_PATH}/model')

        # Load NN Model
        self.model = FeedforwardModel(input_size * SEQUENCE_LENGTH, hidden_size, output_size, learning_rate)
        # self.model = CNNModel(input_size, hidden_size, output_size, sequence_length, learning_rate)
        # self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, learning_rate)

        # ROS2 Subscriber Initialization
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states',      self.jointStatesCallback, 1)
        self.ft_sensor_subscriber   = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor', self.FTSensorCallback, 1)

        time.sleep(1)

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

    def FTSensorCallback(self, data:Wrench):

        """ FT Sensor Callback """

        # Get FT Sensor Data
        self.ft_sensor_data = data

    def append_new_data(self, joint_states:JointState, ft_sensor_data:Wrench):

        """ Append New Data - FIFO Buffer """

        # Append New Joint States and FT-Sensor Data
        self.joint_states_data_list.append(joint_states)
        self.ft_sensor_data_list.append(ft_sensor_data)

        # Keep the Buffer Size within the Maximum Limit - Remove the Oldest Data
        if len(self.joint_states_data_list) > SEQUENCE_LENGTH: self.joint_states_data_list.pop(0)
        if len(self.ft_sensor_data_list) > SEQUENCE_LENGTH: self.ft_sensor_data_list.pop(0)

    def get_tensor_data(self):

        """ Get the Entire Buffer """

        # Prepare the Data Vector for the Model
        # data = [joint.velocity.tolist() + [ft_sensor.force.x, ft_sensor.force.y, ft_sensor.force.z, ft_sensor.torque.x, ft_sensor.torque.y, ft_sensor.torque.z]
        #         for joint, ft_sensor in zip(self.joint_states_data_list, self.ft_sensor_data_list)]

        # Prepare the Data Vector for the Model
        data = [[ft_sensor.force.x, ft_sensor.force.y, ft_sensor.force.z, ft_sensor.torque.x, ft_sensor.torque.y, ft_sensor.torque.z]
                for joint, ft_sensor in zip(self.joint_states_data_list, self.ft_sensor_data_list)]

        # Return the Data as a Tensor
        return torch.tensor(data).unsqueeze(0)

    def clear_buffer(self):

        """Clear the buffer"""

        self.joint_states_data_list = []
        self.ft_sensor_data_list = []

    def main(self):

        """ Main Loop """

        # Spin Once
        rclpy.spin_once(self, timeout_sec=0.5/float(self.ros_rate))

        # Append New Joint States and FT-Sensor Data
        self.append_new_data(self.joint_states, self.ft_sensor_data)

        # Get the Entire Buffer
        data = self.get_tensor_data()

        # Return if the Buffer is not Full
        if data.shape[1] < SEQUENCE_LENGTH: return

        # Save the Data
        with open(f'{PACKAGE_PATH}/bag/data.csv', 'w') as file:
            for d in data.squeeze().tolist():
                file.write(','.join([str(i) for i in d]) + '\n')

        # Pass the Data to the Model
        output = self.model(data)
        print(f'Output: {output}')

        # Rate Sleep
        # self.rate.sleep()

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = GripperControlNode(500)

    while rclpy.ok():

        # Run Node
        node.main()
