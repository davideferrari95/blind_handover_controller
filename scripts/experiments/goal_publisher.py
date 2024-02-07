#! /usr/bin/env python3

import rclpy, time
from rclpy.node import Node
from typing import List
from std_srvs.srv import Trigger
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

class HandoverGoalPublisher(Node):

    """ Handover Goal Publisher Node """

    # UR5e
    # goal1 = [1.57, -1.75, -1.57, -1.57, 1.75, -1.0]
    # goal2 = [0.57, -1.75, -1.57, -1.57, 1.75, -1.0]
    # goal2 = [1.0, -1.50, -1.0, -1.0, 1.75, -1.0]

    # UR10e
    goal1 = [0.15, -1.71, 2.28, -2.13, -1.67, 0.39]
    goal2 = [0.45, -1.71, 2.28, -2.13, -1.67, 0.39]
    # goal2 = [0.45, -2.30, 2.28, -2.13, -1.67, 0.39]

    def __init__(self):

        # Node Initialization
        super().__init__('Handover_Goal_Publisher')

        # ROS2 Publisher & Client Initialization
        self.joint_goal_pub = self.create_publisher(Float64MultiArray, '/handover/joint_goal', 1)
        self.stop_admittance_client = self.create_client(Trigger, '/handover/stop')

        time.sleep(1)

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

        """ Call Service """

        # Wait For Service
        self.stop_admittance_client.wait_for_service(timeout_sec=1.0)

        # Call Service - Asynchronous
        future = self.stop_admittance_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        return future.result()

if __name__ == '__main__':

    # ROS2 Initialization
    rclpy.init()

    # Create Node
    node = HandoverGoalPublisher()

    # Publish Joint Goal 1
    print('Publishing Joint Goal 1')
    node.publishJointGoal(node.goal1)
    time.sleep(20)

    # Stop Handover
    # node.stopHandover()
    # time.sleep(1)

    # Publish Joint Goal 2
    print('Publishing Joint Goal 2')
    node.publishJointGoal(node.goal2)
    time.sleep(20)

    # Stop Handover
    # node.stopHandover()
