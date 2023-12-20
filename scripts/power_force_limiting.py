import rclpy

class PowerForceLimitingController():

    """ Power Force Limiting (PFL) Controller """

    def __init__(self, rate, human_mass, robot_mass, complete_debug, debug):

        """ PFL Controller """

        # ROS Parameters
        self.rate = rate

        # PFL Parameters
        self.human_mass, self.robot_mass = human_mass, robot_mass

        # Set Debug Flags
        self.complete_debug, self.debug = complete_debug, debug or complete_debug

#   P_H_      = tf2::Vector3(0, 0, 1);
#   P_R_      = tf2::Vector3(0, 0, 0);
# https://github.com/ARSControl/dynamic_planner/blob/master/trajectory_scaling/src/trajectory_scaling.cpp
# void TrajectoryScaling::humanPointCallback(
#   const geometry_msgs::PointStamped::ConstPtr& hp)
# {
#   geometry_msgs::PointStamped human_point = *hp;
#   tf2::doTransform(human_point, human_point, transform_);
#   //  buffer_.transform(*hp, "base_link");
#   P_H_ = tf2::Vector3(human_point.point.x, human_point.point.y, human_point.point.z);
#   // std::cout << "Human Point: " << P_H_.x() << " " << P_H_.y() << " " << P_H_.z() << std::endl;
# }

# void TrajectoryScaling::robotPointCallback(
#   const geometry_msgs::PointStamped::ConstPtr& rp)
# {
#   geometry_msgs::PointStamped robot_point = *rp;
#   // tf2::doTransform(robot_point, robot_point, transform_);
#   // buffer_.transform(*rp, "base_link");
#   P_R_ = tf2::Vector3(robot_point.point.x, robot_point.point.y, robot_point.point.z);
#   // std::cout << "Robot Point: " << P_R_.x() << " " << P_R_.y() << " " << P_R_.z() << std::endl;
# }

# tf2::Vector3 TrajectoryScaling::computeVersor()
# {
#   // std::cout << (P_H_-P_R_).x() << " " << (P_H_-P_R_).y() << " " << (P_H_-P_R_).z() << std::endl;
#   return (P_H_ - P_R_) / (P_R_.distance(P_H_));
# }
