# Update Meeting

## Progress

- Created a ROS2 example package
- Converted **Alexa Skill** and **Alexa Conversations** to ROS2
- Re-Written **Node-RED** for ROS2 Integration
- Converted **UR Robot Drivers** to ROS2
- Generated Trajectories with **Polynomial Trajectory Generator** (5th Order)
- Implemented the **Admittance Controller** following a Planned Trajectory
- Implemented the **SSM Controller** ensuring the Safety ISO/TS 15066
- Implemented the **PFL Controller** ensuring the Safety ISO/TS 15066

## Blind Handover - Plan

1. **Handover Procedure** reading the Hand Location with the **Optitrack**:
&nbsp;
    - UR10e/UR5e Manipulator
    - UR10e/UR5e Gripper
    - Optitrack
&nbsp;
1. Implement a Controller that ensures the Safety ISO/TS 15066 ⇾ **PFL** Controller:
&nbsp;
    - Trajectory Planning Generator (Goal = Handover Location with Optitrack / 6IMPOSE Object Recognition).
&nbsp;
      - Polynomial Trajectory Generator (5th Order).
      - Splines Trajectory Generator.
&nbsp;
    - Implement the **Admittance** Controller:
&nbsp;
      - Standard Admittance Controller (Mx'' + Bx' + Kx = Fe, x = Xdes - Xact).
      - Admittance Follow Planned Trajectory (x''des, x'des, xdes).
      - External Force to adapt Robot Dynamics to the Environment.
      - Gripper Compensation in External Force Sensor Readings. ⇾ Teach Pendant Tool Compensation.
&nbsp;
    - **ISO/TS 15066** ⇾ Power and Force Limiting Equations:
&nbsp;
      - Compute the distance between the human and the robot:
      - Compute the value of *vrel_max* according to ISO regulations, determining the moving parts of both the robot and the human.
      - Transpose the velocity (projection) of the robot along the separation direction.
      - Transpose the velocity (projection) of the human along the separation direction (direction of distance between human and robot).
      - Compute the relative velocity between the human and the robot.
      - If the relative velocity exceeds *vrel_max*, reduce the speed of the robot.
&nbsp;
    - Strictly implement **ISO/TS 15066**:
&nbsp;
      - Alpha < 0 (Robot leaving the human) is in the ISO (ask Andrea).
      - Switch between *SSM Controller* and *PFL Controller*.
&nbsp;
1. Study the **Force-Load Transfer** between the Robot and the Human:
&nbsp;
    - Can we use the Force-Load Transfer to detect the Handover Success ?
    - Find a Robust strategy to detect the Handover Success.
&nbsp;
    - Little User-Study to find the Force-Load Transfer:
&nbsp;
      - Robot pass an object to the human (Different Objects / Weights / Sizes / Positions).
      - Get the Force-Load Transfer Curve between the Robot and the Human.
      - Create a **Force-Load Transfer** Model with PyTorch.
      - Test the Model with different Objects / Weights / Sizes / Positions.
&nbsp;
1. Implement the Handover Procedure with the **Automatic Success Detection** and **Release** of the Object.
&nbsp;
1. Integrate the 6IMPOSE Object Recognition.

## Personal Safety

- Comparative experiment ? with *"ISO Safety"* vs *"Personal Safety"*.
&nbsp;
- Define Risk Indicators:
&nbsp;
  - Contact risk between *human head / body* and robot is different from *human hands / arms* and robot (ISO/TS 15066 - Appendix A.2 - 29 Body Areas / 12 Body Regions).
  - If Risk Indicator is too high ⇾ Slow Down.
&nbsp;
- Define How to Change Robot Behavior in according to the User Requests.
- Training task with ISO, communication and possible interactions.
- NB: Biomechanical limits (max pressure and force) in ISO/TS 15066 are based on conservative estimation and scientific research on pain sensation.

### Experiment

- Robot must fill some small boxes with objects (e.g. screws / pen drives etc.).
- User must apply a label on each box.
- User can ask robot to *Slow Down* / *Stop* / *Move Further* / *Hurry Up*.
- Goal for the user is to finish the task as fast as possible.
&nbsp;
- Collect data about the interaction:
&nbsp;
  - Execution Time
  - Experience in Robotics
  - Trust / Confidence in the Robot / Application / Task
  - User Satisfaction / Stress / Fear / Anxiety / Frustration
&nbsp;
- Idea:
&nbsp;
  - Graph with User Experience (trust / confidence) vs Robot Behavior (speed / acceleration) in relation with ISO.
  - How much the velocity / distance is further from the ISO standard.
  - How much the fear / anxiety / frustration is higher / lower.
  - Coward approach ⇾ ISO makes fear ⇾ Slow down the Robot.

## Why Gazebo / Coppelia

- Gazebo is heavy and carries on a lot of useless stuff.
- Gazebo cannot be used for real-time control (accurate velocity control).
- Coppelia is more lightweight and can be used for real-time control.
- Other simulators ? (e.g. Webots)
