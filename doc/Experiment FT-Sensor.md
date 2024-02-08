# FT-Sensor Load-Curve Experiment

- This experiment is used to generate a transfer load curve during an HR Handover procedure.
- The experiment is performed with a pick-and-place task, where the robot picks up an object and hand it to the user.
- When the user presses the `ENTER` key, the robot will stop the handover procedure opening the gripper and releasing the object.
- The FT-Sensor records the force data from the pick to the hand moments.

## Run the Experiment

- Launch the handover_controller:

        ros2 launch handover_controller handover_controller.launch.py

- Launch the Experiment nodes:

        ros2 launch handover_controller ft_experiment.launch.py

## Data Collection

- The data is saved in the `data` folder in the `handover_controller` package.
- The data is saved in two `.csv` file with the following data:

        ft_sensor_data.csv -> fx,fy,fz,tx,ty,tz
        joint_states_data.csv -> joint_velocity

### Tracking Table for Data Collection

| Experiment Number | Who    | Object       | Other Details         |
| :---------------: | :----: | :----------: | :-------------------: |
| 01 - 25           | Davide | Pliers       | 180° Gripper Rotation |
| 26 - 50           | Davide | Pliers       | /                     |
| 51 - 75           | Davide | Screwdriver  | Long / Heavy One      |
| 76 - 100          | Davide | Screwdriver  | 90° Gripper Rotation  |
| 101 - 125         | Davide | Scissors     | 90° Gripper Rotation  |
| 126 - 150         | Davide | Scissors     | /                     |
| 151 - 175         | Davide | Box          | UART Box              |
| 176 - 200         | Davide | Box          | UART Box - Other Pose |
| 201 - 225         | Davide | Box          | UART Box - High Pose  |
| 226 - 250         | Davide | Wrench       | /                     |
| 251 - 275         | Davide | Wrench       | 90° Gripper Rotation  |
| 276 - 300         | Davide | Pliers       | 90° Gripper Rotation  |
