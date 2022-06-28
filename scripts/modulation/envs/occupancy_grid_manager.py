import numpy as np
import rospy
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid


class OccupancyGridManager:
    """Credit: https://github.com/awesomebytes/occupancy_grid_python/blob/master/src/occupancy_grid_python/occupancy_grid_impl.py"""
    def __init__(self, topic: str, subscribe_to_updates: bool):
        # OccupancyGrid starts on lower left corner
        self._grid_data = None
        self._occ_grid_metadata = None
        self._reference_frame = None
        self._sub = rospy.Subscriber(topic, OccupancyGrid, self._occ_grid_cb, queue_size=1)
        if subscribe_to_updates:
            rospy.loginfo("Subscribing to updates!")
            self._updates_sub = rospy.Subscriber(topic + '_updates', OccupancyGridUpdate, self._occ_grid_update_cb, queue_size=1)
        else:
            self._updates_sub = None
        rospy.loginfo("Waiting for '" + str(self._sub.resolved_name) + "'...")
        i = 0
        while ((self._occ_grid_metadata is None) or (self._grid_data is None)) and (not rospy.is_shutdown()) and (i < 250):
            rospy.sleep(0.1)
            i += 1
            # m = rospy.wait_for_message(topic, OccupancyGrid, 5)
        if self._grid_data is None:
            # can fail to get the costmap sometimes as it's being published in a weird way where only a new subscriber receives it once
            # -> ALWAYS NEED TO CALL self._sub.unregister() / self.clear() before creating a new map instance!
            raise RuntimeError("Could not get costmap data. Forgot to call self._sub.unregister()?")
            # print("Could not get costmap data, initialising it from rosparams")
            # self._occ_grid_metadata = DotDict({n: rospy.get_param('/'.join(topic.split('/')[:-1] + [n])) for n in ['resolution', 'height', 'width']})
            # self._occ_grid_metadata.origin = tuple(rospy.get_param('/'.join(topic.split('/')[:-1] + [n])) for n in ['origin_x', 'origin_y'])
            # self._reference_frame = rospy.get_param('/'.join(topic.split('/')[:-1] + ['global_frame']))
            # self._grid_data = np.zeros((int(self.height / self.resolution), int(self.width / self.resolution)), dtype=np.int8)

        rospy.loginfo(f"OccupancyGridManager for '{self._sub.resolved_name}' initialized!")

    @property
    def resolution(self):
        return np.round(self._occ_grid_metadata.resolution, 3)

    @property
    def width(self):
        return self._occ_grid_metadata.width

    @property
    def height(self):
        return self._occ_grid_metadata.height

    @property
    def origin(self):
        return self._occ_grid_metadata.origin

    @property
    def reference_frame(self):
        return self._reference_frame

    def get_local_map(self):
        return np.flipud(self._grid_data)

    def _occ_grid_cb(self, data):
        rospy.logdebug("Got a full OccupancyGrid update")
        self._occ_grid_metadata = data.info
        # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html, first index is the row, second index the column
        self._grid_data = np.array(data.data, dtype=np.int8).reshape(data.info.height, data.info.width)
        self._reference_frame = data.header.frame_id

    def _occ_grid_update_cb(self, data):
        rospy.logdebug("Got a partial OccupancyGrid update")
        # x, y origin point of the update, width and height of the update
        # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html, first index is the row, second index the column
        if self._grid_data is not None:
            data_np = np.array(data.data, dtype=np.int8).reshape(data.height, data.width)
            self._grid_data[data.y:data.y + data.height, data.x:data.x + data.width] = data_np

    def close(self):
        self._sub.unregister()
        if self._updates_sub is not None:
            self._updates_sub.unregister()
        rospy.sleep(0.1)
