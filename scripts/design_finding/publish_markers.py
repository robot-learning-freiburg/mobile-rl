from geometry_msgs.msg import Quaternion, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
# from modulation.envs.env_utils import rpy_to_quaternion
import rospy
from typing import Any, Tuple
import csv
from itertools import islice

import numpy as np


class RvizMarkerPublisher:
    def __init__(self, topic_name: str = "visualization_marker"):
        self.marker_pub = rospy.Publisher(
            topic_name, Marker, queue_size=2000, tcp_nodelay=True
        )
        self.marker_array_pub = rospy.Publisher(
            topic_name + "_array", MarkerArray, queue_size=2000, tcp_nodelay=True
        )

    @staticmethod
    def get_marker(
        namespace: str,
        marker_pose: Pose,
        marker_scale,
        marker_id: int,
        frame_id: str,
        geometry: str,
        color: Any,
        alpha: float = 1,
        lifetime_secs: float = None,
        min_border: float = 2.5,
        max_border: float = 5.0,
    ):
        assert len(marker_scale) == 3
        alpha = 0.6
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.ADD
        if geometry == "arrow":
            marker.type = Marker.ARROW
        elif geometry == "cube":
            marker.type = Marker.CUBE
        elif geometry == "sphere":
            marker.type = Marker.SPHERE
        else:
            raise NotImplementedError()

        marker.pose = marker_pose
        marker.scale.x = marker_scale[0]
        marker.scale.y = marker_scale[1]
        marker.scale.z = marker_scale[2]

        if isinstance(color, float):
            color += 0.3
            normalized_value = min(
                1.0, (color - min_border) / (max_border - min_border)
            )
            marker.color.g = normalized_value
            marker.color.r = 1.0 - normalized_value
            marker.color.b = 0.0
        elif color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "orange":
            marker.color.r = 1.0
            marker.color.g = 120 / 255.0
        elif color == "cyan":
            marker.color.g = 1.0
            marker.color.b = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        elif color == "black":
            pass
        else:
            raise NotImplementedError(color)
        marker.color.a = alpha
        if lifetime_secs is not None:
            marker.lifetime.secs = 1
        return marker

    def pub_single_marker(self, *args, **kwargs):
        marker = self.get_marker(*args, **kwargs)
        self.marker_pub.publish(marker)

    def pub_marker_array(self, markers: list):
        marker_array = MarkerArray()
        id = 0
        # Renumber the marker IDs
        for marker in markers:
            marker.id = id
            id += 1
            marker_array.markers.append(marker)
        self.marker_array_pub.publish(marker_array)

    def clear_all_markers(self, frame_id: str):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.get_rostime()
        marker.action = 3
        self.marker_pub.publish(marker)

    @staticmethod
    def pose_to_list(pose: Pose) -> list:
        return [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

    def list_to_pose(self, l: list) -> Pose:
        if len(l) == 6:
            # q = rpy_to_quaternion(*l[3:])
            print("ohoh")
        elif len(l) == 7:
            q = Quaternion(l[4], l[5], l[6], l[3])
        else:
            raise ValueError(l)
        return Pose(Point(l[0], l[1], l[2]), q)


def get_observation_points(filename):
    observation_points = {}
    with open(filename, mode="r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            position = tuple(map(float, row[0].strip("[]").split()))
            value = float(row[1])
            observation_points[position] = value
    return observation_points


def remove_window_or_threshold(
    point_cloud: dict,
    x_border: Tuple,
    y_border: Tuple,
    z_border: Tuple,
    threshold: float,
    remove_window: bool = False,
) -> dict:
    point_cloud_window = {}
    for point, value in point_cloud.items():
        in_x_range = x_border[0] <= point[0] <= x_border[1]
        in_y_range = y_border[0] < point[1] < y_border[1]
        in_z_range = z_border[0] <= point[2] <= z_border[1]
        if in_x_range and in_y_range and in_z_range and remove_window:
            continue
        if value <= threshold:
            continue
        else:
            point_cloud_window[point] = value
    return point_cloud_window


if __name__ == "__main__":
    rospy.init_node("kinematic_feasibility_py", anonymous=False)
    filename = "scripts/logs/design/20241119_0801_manipulabilities/ManipulabilityPoints/manipulability_points_8.csv"
    filename = "scripts/logs/design/20241204_0700/ManipulabilityPoints/manipulability_points_1.csv"
    position_mask = [0, 0, 0, 1]
    x_window = (0.7, 2.5)  # 0.3, 1.6
    y_window = (-1.7, 0.0)  # - 0.3, 0.3
    z_window = (1.0, 2.8)  # 0.3 , 1.8
    point_value_dict = get_observation_points(filename=filename)
    point_number = len(point_value_dict)
    pub_marker = RvizMarkerPublisher()
    highest_value = max(point_value_dict.values())
    threshold = 0.0

    point_value_window_dict = remove_window_or_threshold(
        point_value_dict, x_window, y_window, z_window, threshold, remove_window=False
    )
    lowest_value = min(point_value_window_dict.values())
    marker_scale = 0.04
    while not rospy.is_shutdown():
        rospy.rostime.wallsleep(1.0)  # Create a subdictionary with the first x items
        sub_point_value_dict = dict(islice(point_value_dict.items(), 99))
        markers = [
            pub_marker.get_marker(
                "manipulability",
                pub_marker.list_to_pose(np.append(np.array(point), position_mask)),
                [marker_scale, marker_scale, marker_scale],
                marker_id=0,
                frame_id="base_link",
                geometry="sphere",
                color=value,
                alpha=1.0,
                lifetime_secs=None,
                min_border=lowest_value,
                max_border=highest_value,
            )
            for point, value in point_value_window_dict.items()
        ]
        pub_marker.pub_marker_array(markers)
        # rospy.spin()
