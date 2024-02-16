#! /usr/bin/env python3

import rclpy, threading, time
from rclpy.node import Node
from typing import List, Tuple
from termcolor import colored

# Import ROS Messages
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench

# Import PyTorch Lightning NN Model
from train_network import FeedforwardModel, LSTMModel, CNNModel, MultiClassifierModel, BinaryClassifierModel
from process_dataset import PACKAGE_PATH, SEQUENCE_LENGTH, LOAD_VELOCITIES, MODEL_TYPE, STRIDE, BALANCE_STRATEGY
from pl_utils import torch, load_hyperparameters, load_model, get_config_name

# Output Length
OUTPUT_LENGTH = 1

class GripperControlNode(Node):

    """ Gripper Control Node Class """

    # Initial FT Sensor and Joint States Data
    joint_states, ft_sensor_data = JointState(), Wrench()
    joint_states.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joint_states.position, joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Data Lists
    joint_states_data_list:List[JointState] = []
    ft_sensor_data_list:List[Wrench] = []

    # Predicted Output List
    predicted_output_list:List[float] = []

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
        model_name, _, model_type, input_size, hidden_size, output_size, sequence_length, num_layers, _ = \
            load_hyperparameters(f'{PACKAGE_PATH}/model', get_config_name(MODEL_TYPE, SEQUENCE_LENGTH, STRIDE, BALANCE_STRATEGY))

        # Load NN Model
        print(colored(f'\nLoading Model: ', 'green'), f'{PACKAGE_PATH}/model/{model_name}.pth\n')

        # Create NN Model
        if   model_type == 'CNN':              self.model = CNNModel(input_size, hidden_size, output_size, sequence_length)
        elif model_type == 'LSTM':             self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        elif model_type == 'Feedforward':      self.model = FeedforwardModel(input_size * sequence_length, hidden_size, output_size)
        elif model_type == 'MultiClassifier':  self.model = MultiClassifierModel(input_size * sequence_length, hidden_size, output_size)
        elif model_type == 'BinaryClassifier': self.model = BinaryClassifierModel(input_size * sequence_length, hidden_size, output_size)

        # Load Model Weights
        load_model(f'{PACKAGE_PATH}/model', model_name, self.model)

        # ROS2 Subscriber Initialization
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states',      self.jointStatesCallback, 1)
        self.ft_sensor_subscriber   = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor', self.FTSensorCallback, 1)

        # ROS2 Publisher Initialization
        self.network_output_publisher = self.create_publisher(Bool, '/ft_network/open_gripper', 1)

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

    def get_tensor_data(self) -> Tuple[int, torch.Tensor]:

        """ Get the Entire Buffer """

        # Prepare the Data Vector for the Model - Joint States and FT Sensor Data
        if LOAD_VELOCITIES: data = [joint.velocity.tolist() + [ft_sensor.force.x, ft_sensor.force.y, ft_sensor.force.z, ft_sensor.torque.x, ft_sensor.torque.y, ft_sensor.torque.z]
                                    for joint, ft_sensor in zip(self.joint_states_data_list, self.ft_sensor_data_list)]

        # Prepare the Data Vector for the Model - FT Sensor Data Only
        else: data = [[ft_sensor.force.x, ft_sensor.force.y, ft_sensor.force.z, ft_sensor.torque.x, ft_sensor.torque.y, ft_sensor.torque.z]
                      for joint, ft_sensor in zip(self.joint_states_data_list, self.ft_sensor_data_list)]

        # Return the Data as a Tensor
        return len(self.joint_states_data_list), torch.tensor(data).unsqueeze(0)

    def get_predicted_output(self, new_output:float) -> float:

        """ Get the Predicted Output """

        # Append the New Output
        self.predicted_output_list.append(new_output)

        # Keep the Buffer Size within the Maximum Limit - Remove the Oldest Data
        if len(self.predicted_output_list) > OUTPUT_LENGTH: self.predicted_output_list.pop(0)

        # Get the Predicted Output
        return torch.tensor(self.predicted_output_list)

    def clear_buffer(self):

        """Clear the buffer"""

        self.joint_states_data_list = []
        self.ft_sensor_data_list = []

    def main(self):

        """ Main Loop """

        while rclpy.ok():

            # Spin Once
            rclpy.spin_once(self, timeout_sec=0.5/float(self.ros_rate))

            # Append New Joint States and FT-Sensor Data
            self.append_new_data(self.joint_states, self.ft_sensor_data)

            # Get the Entire Buffer
            length, data = self.get_tensor_data()

            # Return if the Buffer is not Full
            if length < SEQUENCE_LENGTH: continue

            # Pass the Data to the Model -> Get the Predicted Output
            output:torch.Tensor = self.model(data).detach().numpy()[0]
            # print(f'Output: {output[0]:.3f} | {output[1]:.3f}')

            # Check the Predicted Output
            if all(self.get_predicted_output(output[1]) > 0.9):

                print(colored(f'Open Gripper', 'green'))
                self.network_output_publisher.publish(Bool(data=True))
                break

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = GripperControlNode(500)

    # Run Node
    node.main()
