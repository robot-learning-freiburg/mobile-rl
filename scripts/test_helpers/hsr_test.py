import rospy
from geometry_msgs.msg import Twist
import time
import numpy as np

TOPIC_HSR = "/hsrb/command_velocity"
TOPIC_TIAGO = "/mobile_base_controller/cmd_vel"

def main():
    rospy.init_node('hsr_test', anonymous=False)

    pub = rospy.Publisher(TOPIC_HSR, Twist)
    rate = rospy.Rate(200)

    msg = Twist()
    while True:
        msg.linear.x = np.cos(time.time()) / 3
        msg.angular.z = np.cos(time.time())
        # msg.linear.y = np.sin(time.time()) / 3
        print(f"{msg.linear}")
        pub.publish(msg)
        rate.sleep()


# rqt_plot /hsrb/command_velocity/linear/x:y /hsrb/odom/twist/twist/linear/x:y /hsrb/command_velocity/angular/z /hsrb/odom/twist/twist/angular/z
# tiago:
# rqt_plot /mobile_base_controller/cmd_vel/linear/x:y /ground_truth_odom/twist/twist/linear/x:y /mobile_base_controller/cmd_vel/angular/z /ground_truth_odom/twist/twist/angular/z
# rqt_plot /ground_truth_odom/pose/pose/position/x:y


if __name__ == '__main__':
    main()
