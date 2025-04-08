#!/usr/bin/env python2
import rospy

import tf
from nav_msgs.msg import Odometry


def handle_odometry_pose(msg, topic, parent, child):
    broadcaster = tf.TransformBroadcaster()
    pose = msg.pose.pose
    # NOTE: Z=0 AS I'M USING IT ONLY FOR ODOM AND O/W PR2 TF HAS A Z OFFSET
    broadcaster.sendTransform((pose.position.x, pose.position.y, 0.0),
                              (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                              rospy.Time.now(),
                              child,
                              parent)

if __name__ == '__main__':
    rospy.init_node('odom_tf_broadcaster')
    ground_truth = rospy.get_param("~ground_truth", "ground_truth_odom")
    parent = rospy.get_param("~parent", "odom")
    child = rospy.get_param("~child", "base_footprint")

    rospy.Subscriber(ground_truth,
                     Odometry,
                     lambda msg, topic: handle_odometry_pose(msg, topic, parent, child),
                     "odom")
    rospy.spin()
