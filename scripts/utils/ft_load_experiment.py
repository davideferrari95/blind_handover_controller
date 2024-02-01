#! /usr/bin/env python3

import rclpy, time, os
from rclpy.node import Node
from typing import List, Union

# Keyboard Listener
from pynput import keyboard
from threading import Thread

# Package Path
from kinematic_wrapper import PACKAGE_PATH
if not os.path.exists(f'{PACKAGE_PATH}/data'): os.mkdir(f'{PACKAGE_PATH}/data')

# Messages & Services
from std_srvs.srv import Trigger
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from ur_rtde_controller.srv import RobotiQGripperControl

class FTSensorExperiment(Node):

    """ FT-Sensor Experiment Node """

    # Initial FT Sensor and Joint States Data
    key_pressed = False
    ft_sensor_data = Wrench()
    joint_states = JointState()
    joint_states.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joint_states.position, joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Save Data Files
    FT_SENSOR_FILE = f'{PACKAGE_PATH}/data/ft_sensor_data.csv'
    JOINT_STATES_FILE = f'{PACKAGE_PATH}/data/joint_states_data.csv'

    def __init__(self, robot:str='UR5e', ros_rate:int=500):

        # Node Initialization
        super().__init__('FT_Sensor_Experiment')

        # ROS2 Rate
        self.ros_rate = ros_rate
        self.rate = self.create_rate(ros_rate)

        # ROS2 Publisher & Client Initialization
        self.joint_goal_pub = self.create_publisher(Float64MultiArray, '/handover/joint_goal', 1)
        self.stop_admittance_client = self.create_client(Trigger, '/handover/stop')
        self.zero_ft_sensor_client  = self.create_client(Trigger, '/ur_rtde/zeroFTSensor')
        self.robotiq_gripper_client = self.create_client(RobotiQGripperControl, '/ur_rtde/robotiq_gripper/command')

        # ROS2 Subscriber Initialization
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states',      self.jointStatesCallback, 1)
        self.ft_sensor_subscriber   = self.create_subscription(Wrench,     '/ur_rtde/ft_sensor', self.FTSensorCallback, 1)

        # UR5e
        if robot == 'UR5e':

            self.HOME = [1.57, -1.75, -1.57, -1.57, 1.75, -1.0]
            self.OBJECT_1 = [1.57, -1.75, -1.57, -1.57, 1.75, -1.0]
            self.OBJECT_2 = [1.57, -1.75, -1.57, -1.57, 1.75, -1.0]
            self.OBJECT_3 = [1.57, -1.75, -1.57, -1.57, 1.75, -1.0]
            self.HANDOVER = [1.57, -1.75, -1.57, -1.57, 1.75, -1.0]

        # UR10e
        elif robot == 'UR10e':

            self.HOME = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.OBJECT_1 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.OBJECT_2 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.OBJECT_3 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.HANDOVER = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]

        else: raise ValueError('Invalid Robot Name')

        # Write Header - Save Data FT Sensor & Joint States Files
        with open(self.FT_SENSOR_FILE, 'w') as file: file.write('fx,fy,fz,tx,ty,tz\n')
        with open(self.JOINT_STATES_FILE, 'w') as file: file.write(f'{self.joint_states.name[0]},{self.joint_states.name[1]},{self.joint_states.name[2]},{self.joint_states.name[3]},{self.joint_states.name[4]},{self.joint_states.name[5]}\n')

        time.sleep(1)

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

    def FTSensorCallback(self, data:Wrench):

        """ FT Sensor Callback """

        # Get FT Sensor Data
        self.ft_sensor_data = data

    def publishJointGoal(self, joint_goal:List[float]):

        """ Publish Handover Joint Goal """

        assert len(joint_goal) == 6, 'Joint Goal Must be a 6-Element List'

        # ROS Message Creation
        msg = Float64MultiArray(data=joint_goal)
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].size = len(joint_goal)
        msg.layout.dim[0].stride = 1
        msg.layout.dim[0].label = 'joint_goal'

        # Publish Message
        if rclpy.ok(): self.joint_goal_pub.publish(msg)

    def stopHandover(self):

        """ Call Stop Handover Service """

        # Wait For Service
        self.stop_admittance_client.wait_for_service(timeout_sec=1.0)

        # Call Service - Asynchronous
        future = self.stop_admittance_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def zeroFTSensor(self):

        """ Call Zero FT Sensor Service """

        # Wait For Service
        self.zero_ft_sensor_client.wait_for_service(timeout_sec=1.0)

        # Call Service - Asynchronous
        future = self.zero_ft_sensor_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def RobotiQGripperControl(self, position:int, speed:int=100, force:int=100):

        """ Call RobotiQ Gripper Service """        

        # Wait For Service
        self.robotiq_gripper_client.wait_for_service(timeout_sec=1.0)

        # Gripper Service Request
        request = RobotiQGripperControl.Request()
        request.position, request.speed, request.force = position, speed, force

        # Call Service - Asynchronous
        future = self.robotiq_gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def goal_reached(self, joint_goal:List[float]):
        
        """ Check if Goal Reached """

        # Get Joint States
        joint_states = self.joint_states

        # Check if Goal Reached
        if all([abs(joint_states.position[i] - joint_goal[i]) < 0.01 for i in range(len(joint_states.position))]): return True
        else: return False

    def save_data(self):

        """ Save Data """

        while not self.key_pressed:

            # Spin Once
            rclpy.spin_once(self, timeout_sec=1/self.ros_rate)

            # Append Data to ft_sensor_data.txt
            with open(self.FT_SENSOR_FILE, 'a') as file:

                # Write FT-Sensor Data
                file.write(str(self.ft_sensor_data.force.x) + ',')
                file.write(str(self.ft_sensor_data.force.y) + ',')
                file.write(str(self.ft_sensor_data.force.z) + ',')
                file.write(str(self.ft_sensor_data.torque.x) + ',')
                file.write(str(self.ft_sensor_data.torque.y) + ',')
                file.write(str(self.ft_sensor_data.torque.z) + '\n')

            # Append Data to joint_states_data.txt
            with open(self.JOINT_STATES_FILE, 'a') as file:

                # Write Joint States Data
                file.write(str(self.joint_states.velocity[0]) + ',')
                file.write(str(self.joint_states.velocity[1]) + ',')
                file.write(str(self.joint_states.velocity[2]) + ',')
                file.write(str(self.joint_states.velocity[3]) + ',')
                file.write(str(self.joint_states.velocity[4]) + ',')
                file.write(str(self.joint_states.velocity[5]) + '\n')

            # Rate Sleep
            self.rate.sleep()

    def wait_key(self, key:str='esc'):

        """ Wait for Key Press to Save Data """

        def on_press(key:Union[keyboard.KeyCode, keyboard.Key], abortKey=key):    

            """ On Press Function """

            # Single-Char Keys or Other Keys
            try: k = key.char
            except: k = key.name

            # Stop Listener
            if k == abortKey: 

                self.key_pressed = True
                return False

        # Loop Function
        def loop_fun():
            while True: pass

        # Start to Listen on a Separate Thread
        listener = keyboard.Listener(on_press=on_press, abortKey=key)
        listener.start()

        # Start Thread with Loop
        Thread(target=loop_fun, args=(), name='loop_fun', daemon=True).start()
        listener.join()

    def handover(self, object_goal:List[float]):

        """ Handover """

        # Go to Object Goal
        self.publishJointGoal(object_goal)
        # while not self.goal_reached(object_goal): self.get_logger().info('Moving to Object Goal', throttle_duration_sec=2.0, skip_first=True)

        # Pick Object
        # self.zeroFTSensor()
        # self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_CLOSED)
        time.sleep(1)

        # Go to Handover Goal
        self.publishJointGoal(self.HANDOVER)

        # Starting Save Data Thread
        print('Saving Data | Press ENTER to Stop')
        save_data_thread = Thread(target=self.save_data, args=(), name='save_data', daemon=True)
        save_data_thread.start()

        # Wait for Key Press to Save Data
        self.wait_key('enter')

        # Stop Save Data Thread
        save_data_thread.join()
        print('Data Saved')

    def main(self):

        """ Main Loop """

        # Home
        print('Going Home')
        self.publishJointGoal(self.HOME)
        # while not self.goal_reached: self.get_logger().info('Moving to HOME', throttle_duration_sec=2.0, skip_first=True)

        # Handover Object 1
        print('Handover Object 1')
        self.handover(self.OBJECT_1)

        # Stop Handover
        print('Stopping Handover')
        self.stopHandover()

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = FTSensorExperiment('UR5e')
    # node = FTSensorExperiment('UR10e')

    # Run Node
    while rclpy.ok(): node.main()
