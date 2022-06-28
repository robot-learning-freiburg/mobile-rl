from typing import Union, List
from pathlib import Path

import time
import numpy as np
import rospy
from geometry_msgs.msg import Quaternion, Pose, Point
from pybindings import multiply_tfs
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
Transform = List[float]
SMALL_NUMBER = 1e-6
IDENTITY_TF = [0., 0., 0., 0., 0., 0., 1.]


def quaternion_to_yaw(q: Union[List, Quaternion]) -> float:
    if isinstance(q, Quaternion):
        q = [q.x, q.y, q.z, q.w]
    assert len(q) == 4, "Have to rewrite for batched"
    yaw = np.arctan2(2.0 * (q[0] * q[1] + q[3] * q[2]), q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2])
    return float(yaw)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """yaw in radians"""
    return rpy_to_quaternion(0, 0, yaw)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """rpy in radians"""
    return Quaternion(*Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat().tolist())


def quaternion_to_rpy(q: Union[list, Quaternion]) -> list:
    """rpy in radians"""
    if isinstance(q, Quaternion):
        q = quaternion_to_list(q)
    return Rotation.from_quat(q).as_euler('xyz').tolist()


def resize_to_resolution(current_resolution: float, target_resolution: float, map):
    if current_resolution == target_resolution:
        return map
    using_np = isinstance(map, np.ndarray)
    if using_np:
        map = Image.fromarray(map)
    size_orig = np.array(map.size)
    size_new = tuple((current_resolution / target_resolution * size_orig).astype(np.int))
    map = map.resize(size_new)
    if using_np:
        map = np.asarray(map)
    return map


def quaternion_to_list(q: Quaternion) -> List:
    return [q.x, q.y, q.z, q.w]


def publish_marker(namespace: str, marker_pose: Pose, marker_scale, marker_id: int, frame_id: str, geometry: str,
                   color: str = "orange", alpha: float = 1):
    assert len(marker_scale) == 3

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.get_rostime()
    marker.ns = namespace
    marker.id = marker_id
    marker.action = 0
    if geometry == "arrow":
        marker.type = Marker.ARROW
    elif geometry == "cube":
        marker.type = Marker.CUBE
    else:
        raise NotImplementedError()

    marker.pose = marker_pose
    marker.scale.x = marker_scale[0]
    marker.scale.y = marker_scale[1]
    marker.scale.z = marker_scale[2]

    assert color == "orange", "atm all orange"
    marker.color.r = 1.0
    marker.color.g = 159 / 255
    marker.color.b = 0.0
    marker.color.a = alpha

    pub = rospy.Publisher('kinematic_feasibility_py', Marker, queue_size=10)
    pub.publish(marker)


def clear_all_markers(frame_id: str):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.get_rostime()
    marker.action = 3
    pub = rospy.Publisher('kinematic_feasibility_py', Marker, queue_size=10)
    pub.publish(marker)


def calc_disc_return(rewards: list, gamma: float) -> float:
    return (gamma ** np.arange(len(rewards)) * rewards).sum()


def pose_to_list(pose: Pose) -> list:
    return [pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


def list_to_pose(l: list) -> Pose:
    if len(l) == 6:
        q = rpy_to_quaternion(*l[3:])
    elif len(l) == 7:
        q = Quaternion(l[3], l[4], l[5], l[6])
    else:
        raise ValueError(l)
    return Pose(Point(l[0], l[1], l[2]), q)


def calc_rot_dist(tf_a, tf_b):
    tf_a, tf_b = np.array(tf_a), np.array(tf_b)
    assert tf_a.shape[-1] == tf_b.shape[-1] == 7, (tf_a.shape, tf_b.shape)
    b = tf_b[..., 3:]
    if len(tf_b.shape) == 2:
        b = np.transpose(b, [1, 0])
    inner_prod = tf_a[..., 3:].dot(b)
    return 1.0 - np.power(inner_prod, 2.0)


def calc_euclidean_tf_dist(tf_a, tf_b):
    tf_a, tf_b = np.array(tf_a), np.array(tf_b)
    assert len(tf_a.shape) == len(tf_a.shape)
    return np.linalg.norm(tf_a[..., :3] - tf_b[..., :3], axis=-1)


def delete_all_markers(topic: str, is_marker_array: bool, frame_id: str = "map"):
    for _ in range(2):
        marker = Marker()
        marker.id = 0
        marker.ns = ''
        marker.header.frame_id = frame_id
        marker.action = Marker.DELETEALL

        if is_marker_array:
            marker_msg = MarkerArray()
            marker_msg.markers.append(marker)
            marker_pub = rospy.Publisher(topic, MarkerArray, queue_size=10)
        else:
            marker_msg = marker
            marker_pub = rospy.Publisher(topic, Marker, queue_size=10)

        for _ in range(3):
            marker_pub.publish(marker_msg)
            time.sleep(0.1)


def clamp_angle_minus_pi_pi(angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < - np.pi:
        angle += 2 * np.pi
    return angle


def rotate_in_place(pose: Union[Pose, List], yaw_radians: float = 0, roll_radians: float = 0, pitch_radians: float = 0) -> list:
    """Rotate a tf in-place"""
    if isinstance(pose, Pose):
        pose = pose_to_list(pose)
    rotated_orientation = multiply_tfs([0, 0, 0, roll_radians, pitch_radians, yaw_radians], pose, False)
    return pose[:3] + rotated_orientation[3:]


def translate_in_orientation(pose: list, translation_world: list):
    assert len(translation_world) == 3, translation_world
    # translation_world = [-0.2,  0., 0., 0., 0., 0., 1.]
    translation_obj = multiply_tfs([0., 0., 0.] + pose[3:], translation_world + [0., 0., 0., 1.], False)
    return multiply_tfs(translation_obj[:3] + [0., 0., 0., 1.], pose, False)
