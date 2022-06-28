#!/usr/bin/env python2
import rospy
import tf
from geometry_msgs.msg import Pose, PointStamped, Vector3, PoseStamped
from functools import partial
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from std_msgs.msg import Bool

# NOTE: should match modulation.envs.tasks.GOAL_TOPIC. Cannot import due to python2/3 incompatibilities
GOAL_TOPIC = "mobilerl_interactive_marker"
GOALPOINTER_ENABLED_TOPIC = "goalpointer_enabled"


def pr2_point_to_goal(x, y, z, point_frame="map"):
    point = PointStamped()
    point.header.frame_id = point_frame
    point.point.x = x
    point.point.y = y
    point.point.z = z

    goal = PointHeadGoal()
    goal.target = point

    # we want the X axis of the camera frame to be pointing at the target
    goal.pointing_frame = "high_def_frame"
    goal.pointing_axis.x = 1
    goal.pointing_axis.y = 0
    goal.pointing_axis.z = 0

    # rad per second
    goal.max_velocity = 0.3

    action_goal = PointHeadActionGoal(goal=goal)
    action_pub.publish(action_goal)



def calc_send_gaze_cmd(whole_body, point, ref_frame_id):
    if np.isinf(point).any() or np.isnan(point).any():
        raise ValueError("The point includes inf or nan.")
    if ref_frame_id is None:
        ref_frame_id = settings.get_frame('base')
    origin_to_ref_ros_pose = whole_body._lookup_odom_to_ref(ref_frame_id)
    origin_to_ref = geometry.pose_to_tuples(origin_to_ref_ros_pose)

    origin_to_base_ros_pose = whole_body._lookup_odom_to_ref(settings.get_frame('base'))
    origin_to_base = geometry.pose_to_tuples(origin_to_base_ros_pose)
    base_to_origin = _invert_pose(origin_to_base)

    ref_to_point = geometry.Pose(point, geometry.quaternion())

    base_to_ref = geometry.multiply_tuples(base_to_origin, origin_to_ref)
    base_to_point = geometry.multiply_tuples(base_to_ref, ref_to_point)

    # output: {'head_pan_joint': 1.3224373722054485, 'head_tilt_joint': 0.52}
    joint_dict = whole_body._kinematics_interface.calculate_gazing_angles(list(base_to_point.pos), str(whole_body._setting['rgbd_sensor_frame']))
    joint_list = [(k, v) for k, v in joint_dict.items()]

    cmd = JointTrajectory()
    cmd.joint_names = [item[0] for item in joint_list]
    p = JointTrajectoryPoint()
    p.positions = [item[1] for item in joint_list]
    p.velocities = [0, 0]
    p.time_from_start = rospy.Duration(0.1)
    cmd.points = [p]
    return cmd


def hsr_point_to_goal(x, y, z, whole_body, action_pub):
    #   https://docs.hsr.io/hsr_develop_manual_en/ros_interface/ros_controller_head.html
    #   https://docs.hsr.io/hsr_develop_manual_en/python_interface/arm_python_interface.html#xtion
    #   https://git.hsr.io/keisuke_takeshita/examples/blob/master/hsrb_interface_patch/look_hand.py
    # with Robot() as robot:
    #     robot = Robot()
    #     whole_body = robot.get('whole_body')
    #     whole_body.gaze_point(point=geometry.Vector3(x=x, y=y, z=z), ref_frame_id='map')

    cmd = calc_send_gaze_cmd(whole_body, geometry.Vector3(x=x, y=y, z=z), ref_frame_id='map')
    action_pub.publish(cmd)


class GoalCallback():
    goal = None

    def callback(self, msg):
        self.goal = msg.pose


class EnabledCallback:
    enabled = True

    def callback(self, msg):
        self.enabled = msg.data


if __name__ == '__main__':
    rospy.init_node('camera_goalpointer')

    ar_marker_frame = rospy.get_param("ar_marker_frame", "ar_marker")
    ar_marker_camera_frame = rospy.get_param("ar_marker_camera_frame")
    robot_name = rospy.get_param("robot_name")
    goal_topic = rospy.get_param("goal_topic", GOAL_TOPIC)
    max_marker_height = rospy.get_param("max_marker_height", 2.5)

    rate = rospy.Rate(50)
    listener = tf.TransformListener()

    goal_cb = GoalCallback()
    goal_sub = rospy.topics.Subscriber(GOAL_TOPIC, PoseStamped, goal_cb.callback, queue_size=100)
    goal_pub = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=1, latch=True)
    enabled_cb = EnabledCallback()
    enabled_sub = rospy.topics.Subscriber(GOALPOINTER_ENABLED_TOPIC, Bool, enabled_cb.callback, queue_size=100)
    prev_enabled = True

    if robot_name == "pr2":
        from pr2_controllers_msgs.msg import PointHeadGoal, PointHeadActionGoal
        action_pub = rospy.Publisher("/head_traj_controller/point_head_action/goal", PointHeadActionGoal, queue_size=10, latch=True)
        point_fn = pr2_point_to_goal
    elif robot_name == "hsr":
        from hsrb_interface import Robot
        from hsrb_interface import geometry, settings
        from hsrb_interface.joint_group import _invert_pose
        robot = Robot()
        whole_body = robot.get('whole_body')
        action_pub = rospy.Publisher('/hsrb/head_trajectory_controller/command', JointTrajectory, queue_size=10)
        point_fn = partial(hsr_point_to_goal, whole_body=whole_body, action_pub=action_pub)
    else:
        raise ValueError(robot_name)

    def hsr_look_straight():
        cmd = JointTrajectory()
        cmd.joint_names = ['head_pan_joint', 'head_tilt_joint']
        p = JointTrajectoryPoint()
        p.positions = [0.0, 0.0]
        p.velocities = [0, 0]
        p.time_from_start = rospy.Duration(0.1)
        cmd.points = [p]
        action_pub.publish(cmd)

    while not rospy.is_shutdown():
        if not enabled_cb.enabled:
            rospy.loginfo_throttle(1, "disabled")
            if enabled_cb.enabled != prev_enabled:
                # make the camera point forward again in case it pointed somewhere else before
                if robot_name == "hsr":
                    hsr_look_straight()
                elif robot_name == "pr2":
                    pr2_point_to_goal(2, 0, 1, point_frame="high_def_frame")
                else:
                    raise NotImplementedError(robot_name)
            prev_enabled = enabled_cb.enabled
            rate.sleep()
            continue

        # if a transform of the goal object is available, point to that
        try:
            (trans, rot) = listener.lookupTransform('map', ar_marker_frame, rospy.Time(0))

            # ignore AR marker detections that don't make sense
            if trans[2] > max_marker_height:
                rospy.loginfo_throttle(1, "marker too high")
                continue
            (trans_from_camera, rot_from_camera) = listener.lookupTransform(ar_marker_camera_frame, ar_marker_frame, rospy.Time(0))
            dist_from_camera = np.linalg.norm(trans_from_camera)
            if not (0.1 < dist_from_camera < 10):
                rospy.loginfo_throttle(1, "dist_from_camera doesn't make sense")
                continue

            initial_goal = goal_cb.goal
            if initial_goal is not None:
                dist_to_initial_pose = np.linalg.norm(np.array([initial_goal.position.x, initial_goal.position.y, initial_goal.position.z]) - trans)
                if dist_to_initial_pose > 1.0:
                    rospy.loginfo_throttle(1, "dist_to_initial_pose too large")
                    continue

            rospy.loginfo("Found /ar_marker tf " + str(trans[0]) + ", " + str(trans[1]) + ", " + str(trans[2]))
            point_fn(trans[0], trans[1], trans[2])

            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()

            msg.pose.position.x = trans[0]
            msg.pose.position.y = trans[1]
            msg.pose.position.z = trans[2]
            msg.pose.orientation.x = rot[0]
            msg.pose.orientation.y = rot[1]
            msg.pose.orientation.z = rot[2]
            msg.pose.orientation.w = rot[3]
            goal_pub.publish(msg)
        # otherwise try to point towards a published initial goal
        except Exception as e:
            print(e)
            if goal_cb.goal is not None:
                initial_goal = goal_cb.goal
                rospy.loginfo("Pointing to inital goal " + str(initial_goal.position.x) + ", " + str(initial_goal.position.y) + ", " + str(initial_goal.position.z))
                point_fn(initial_goal.position.x, initial_goal.position.y, initial_goal.position.z)
            rospy.loginfo("No goal found doing nothing")

        rate.sleep()
