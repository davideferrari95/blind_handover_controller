#! /usr/bin/env python3

import rclpy, time
from rclpy.node import Node
from typing import List, Union

# Keyboard Listener
from pynput import keyboard
from threading import Thread

# Messages & Services
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
from ur_rtde_controller.srv import RobotiQGripperControl

class FTSensorExperiment(Node):

    """ FT-Sensor Experiment Node """

    # Initial Joint States Data
    joint_states = JointState()
    joint_states.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    joint_states.position, joint_states.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def __init__(self, robot:str='UR5e'):

        # Node Initialization
        super().__init__('FT_Sensor_Experiment')

        # ROS2 Publisher & Client Initialization
        self.joint_goal_pub = self.create_publisher(Float64MultiArray, '/handover/joint_goal', 1)
        self.save_data_pub  = self.create_publisher(Bool, '/handover/save_data', 1)
        self.stop_admittance_client = self.create_client(Trigger, '/handover/stop')
        self.zero_ft_sensor_client  = self.create_client(Trigger, '/ur_rtde/zeroFTSensor')
        self.robotiq_gripper_client = self.create_client(RobotiQGripperControl, '/ur_rtde/robotiq_gripper/command')

        # ROS2 Subscriber Initialization
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.jointStatesCallback, 1)

        # UR5e
        if robot == 'UR5e':

            self.HOME     = [-1.7102339903460901, -1.62247957805776, 1.6913612524615687, -1.6592804394164027, -1.5053008238421839, 3.146353244781494]
            self.OBJECT_1 = [-3.692266289387838, -1.5014120799354096, 2.3944106737719935, -2.464505811730856, -1.5677226225482386, -0.4507320976257324]
            self.HANDOVER = [-2.48739463487734, -1.3766034108451386, 1.7061370054828089, -1.8849464855589808, -1.588557545338766, 0.5314063429832458]


        # UR10e
        elif robot == 'UR10e':

            self.HOME = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.OBJECT_1 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.OBJECT_2 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.OBJECT_3 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
            self.HANDOVER = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]

        else: raise ValueError('Invalid Robot Name')

        time.sleep(1)

    def jointStatesCallback(self, data:JointState):

        """ Joint States Callback """

        # Get Joint States
        self.joint_states = data

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
        if self.stop_admittance_client.wait_for_service(timeout_sec=1.0):

            # Call Service - Asynchronous
            future = self.stop_admittance_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            return future.result()

        else: self.get_logger().error('Stop Handover Service Not Available')

    def zeroFTSensor(self):

        """ Call Zero FT Sensor Service """

        # Wait For Service
        if self.zero_ft_sensor_client.wait_for_service(timeout_sec=1.0):

            # Call Service - Asynchronous
            future = self.zero_ft_sensor_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            return future.result()

        else: self.get_logger().error('Zero FT Sensor Service Not Available')

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

        # Spin Once
        rclpy.spin_once(self)

        # Get Joint States
        joint_states = self.joint_states

        # Check if Goal Reached
        if all([abs(joint_states.position[i] - joint_goal[i]) < 0.01 for i in range(len(joint_states.position))]): return True
        else: return False

    def move_and_wait(self, joint_goal:List[float], goal_name:str='Goal', throttle_duration_sec:float=5.0, skip_first:bool=False):

        """ Move to Joint Goal and Wait for Goal to be Reached """

        # Publish Joint Goal
        self.publishJointGoal(joint_goal)

        # Wait for Goal to be Reached
        while not self.goal_reached(joint_goal): self.get_logger().info(f'Moving to {goal_name}', throttle_duration_sec=throttle_duration_sec, skip_first=skip_first)
        self.get_logger().info(f'{goal_name} Reached\n')

    def wait_key(self, key:str='esc'):

        """ Wait for Key Press to Save Data """

        def on_press(key:Union[keyboard.KeyCode, keyboard.Key], abortKey=key):    

            """ On Press Function """

            # Single-Char Keys or Other Keys
            try: k = key.char
            except: k = key.name

            # Stop Listener
            if k == abortKey: return False

        # Start to Listen on a Separate Thread
        listener = keyboard.Listener(on_press=on_press, abortKey=key)
        listener.start()

        # Start Thread with Loop
        Thread(target=None, args=(), name='loop_fun', daemon=True).start()
        listener.join()

    def handover(self, object_goal:List[float]):

        """ Handover """

        # Open Gripper and Go to Home
        self.zeroFTSensor()
        self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_OPENED)
        # self.move_and_wait(self.HOME, 'HOME', 5.0, False)
        time.sleep(1)

        # Go to Object Goal
        self.move_and_wait(object_goal, 'Object Goal', 5.0, False)
        time.sleep(1)

        # Pick Object
        self.zeroFTSensor()
        self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_CLOSED)
        time.sleep(1)

        # Go to Handover Goal
        self.publishJointGoal(self.HANDOVER)

        # Starting Save Data Thread
        print('Saving Data | Press ENTER to Stop\n')
        self.save_data_pub.publish(Bool(data=True))

        # Wait for Key Press to Save Data
        self.wait_key('enter')

        # Stop Save Data Thread
        print('Stopping Data Save')
        self.save_data_pub.publish(Bool(data=False))
        self.RobotiQGripperControl(position=RobotiQGripperControl.Request.GRIPPER_OPENED)
        time.sleep(1)

    def main(self):

        """ Main Loop """

        # Handover Object 1
        print('\nHandover Object 1')
        self.handover(self.OBJECT_1)

        # Stop Handover
        print('\nStopping Handover\n')
        self.stopHandover()

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = FTSensorExperiment('UR5e')
    # node = FTSensorExperiment('UR10e')

    # Run Node
    node.main()
