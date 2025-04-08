from pathlib import Path
from typing import Union, List, Tuple, Optional, Callable
import copy

import cv2
import numpy as np
import rospy
import yaml
from PIL import Image, ImageDraw, ImageOps, ImageChops
from geometry_msgs.msg import Pose, Point, Quaternion
from matplotlib import pyplot as plt
from nav_msgs.msg import OccupancyGrid
from pybindings import get_inflated_map
from scipy.ndimage.interpolation import rotate

from modulation.envs.env_utils import quaternion_to_yaw, yaw_to_quaternion, resize_to_resolution, Transform, SMALL_NUMBER
from modulation.envs.occupancy_grid_manager import OccupancyGridManager
from modulation.envs.simulator_api import GazeboAPI, DummySimulatorAPI, SpawnObject, ObjectGeometry, GazeboObject, get_rectangle, \
    DynObstacle
from modulation.envs.eeplanner import PointToPointPlanner
from pybindings import multiply_tfs


class Map:
    def __init__(self,
                 global_map_resolution: float,
                 local_map_resolution: float,
                 robot_frame_id: str,
                 inflation_radius: float,
                 world_type: str,
                 update_from_global_costmap: bool,
                 floorplan: Union[Path, Tuple[float, float]] = None,
                 orig_resolution: Optional[float] = None,
                 local_map_size_meters: float = 3.0,
                 origin: Optional[Tuple[float, float, float]] = None,
                 initial_base_rng_x=(0, 0),
                 initial_base_rng_y=(0, 0),
                 initial_base_rng_yaw=(0, 0),
                 pad_floorplan_img: bool = True,
                 max_map_height: float = None):
        if floorplan is None:
            # create an empty map of this size
            floorplan = (11, 11)
        elif isinstance(floorplan, (Path, str)):
            floorplan = Path(floorplan)
            if floorplan.suffix == '.yaml':
                assert orig_resolution is None, orig_resolution
                floorplan, orig_resolution = self.get_map_yaml(floorplan)

        if orig_resolution is None:
            orig_resolution = global_map_resolution

        self.max_map_height = 3.0 if (max_map_height is None) else max_map_height
        if rospy.get_param("/fake_gazebo", False):
            world_type = "sim"
            self.fake_gazebo = True
        else:
            self.fake_gazebo = False
        # m per cell (per pixel)
        if world_type == 'sim':
            assert global_map_resolution == local_map_resolution, "Will require changes to work correctly. Mostly during training the local map is a crop from the transformed global map -> resolution atm cannot be larget than global map"
            self.global_map_resolution = global_map_resolution
        else:
            self.global_map_resolution = rospy.get_param("/costmap_node_global/costmap/resolution")
        self.local_map_resolution = local_map_resolution
        self.np_random: np.random.RandomState = None  # will be set in combined_env.set_map()
        self.map_path = floorplan if isinstance(floorplan, (str, Path)) else None
        # [W, H], resized to match the defined resolution so we will have the same size for the local maps
        self._floorplan_img_orig = self.load_map(floorplan,
                                                 current_resolution=orig_resolution,
                                                 target_resolution=self.global_map_resolution,
                                                 max_map_height=self.max_map_height,
                                                 pad=pad_floorplan_img)
        self._floorplan_img = self._floorplan_img_orig.copy()
        self._floorplan_img_filled = self._floorplan_img_orig.copy()
        self._map_W_meter = self.global_map_resolution * self._floorplan_img.width
        self._map_H_meter = self.global_map_resolution * self._floorplan_img.height
        # in meters
        if origin is None:
            # take the center as the origin
            origin = (self._map_W_meter / 2, self._map_H_meter / 2, 0)
        self._origin_W_meter, self._origin_H_meter, self._origin_Z_meter = origin

        sz = int(local_map_size_meters / self.local_map_resolution)
        self.output_size = (sz, sz)

        self.robot_frame_id = robot_frame_id
        self._map_frame_rviz = robot_frame_id if world_type == 'sim' else 'map'

        # set to half of the robot base size (plus padding if desired)
        self.inflation_radius_meter = inflation_radius
        self.inflated_map = dict()

        assert len(initial_base_rng_x) == len(initial_base_rng_y) == len(initial_base_rng_yaw) == 2
        self._initial_base_rng_x = initial_base_rng_x
        self._initial_base_rng_y = initial_base_rng_y
        # in radians
        self._initial_base_rng_yaw = initial_base_rng_yaw

        self._world_type = world_type
        self._ogm = None
        self._ogm_global = None
        self.update_from_global_costmap = (world_type != 'sim') and update_from_global_costmap

    def set_world(self, world_type: str):
        if self.fake_gazebo:
            world_type = "sim"
        self._world_type = world_type

    @staticmethod
    def get_map_yaml(path: Union[str, Path]) -> Tuple:
        assert path.exists(), path
        with open(path) as f:
            map_config = yaml.safe_load(f)
            floorplan_img_path = path.parent / map_config['image']
            orig_resolution = map_config['resolution']
            # do not load the origin from the yaml as map_server uses a different way to specify it than we do
            # origin = map_config['origin']
        return floorplan_img_path, orig_resolution

    @staticmethod
    def get_empty_floorplan(size_meters: Tuple[float, float], resolution: float):
        # returns an array H x W as Image.fromarray expects
        map = np.zeros([int(size_meters[1] / resolution), int(size_meters[0] / resolution)], dtype=bool)
        # wall around the map
        map[[0, -1], :] = True
        map[:, [0, -1]] = True
        return map

    @staticmethod
    def load_map(map: Union[str, Path, Tuple[float, float]], current_resolution: float, target_resolution: float, max_map_height, pad: bool = True) -> Image:
        """
        :param map: either a path to an image or a tuple specifying the size of an empty floorplan [H x W] in meters
        :param current_resolution:
        :param target_resolution:
        :return: a binary map with occupied==1, free==0
        """
        if isinstance(map, (Path, str)):
            assert Path(map).exists()
            # '1' is a binary map occupied / free
            # NOTE: PIL interprets '0' as black and '1' as white. So always use plt.imshow(np.asarray(img)) to visualise
            map = Image.open(map).convert('1')
            # invert so that black are obstacles (True) and non-black are free (False)
            map = Image.fromarray(~np.asarray(map))
        elif isinstance(map, tuple):
            assert len(map) == 2, "Should be a tuple of HxW in meters"
            map = Map.get_empty_floorplan(map, resolution=current_resolution)
            # expects the input array to be HxW, transposing it into WxH
            map = Image.fromarray(map.astype(np.bool))
        elif isinstance(map, Image.Image):
            assert map.mode == 'F', map.mode
            return resize_to_resolution(current_resolution, target_resolution, map)
        else:
            raise ValueError(map)
        assert len(map.size) == 2, map.size

        # resize to be of target_resolution
        map = resize_to_resolution(current_resolution, target_resolution, map)

        # pad the map so the base doesn't run out of bound and always gets the collision penalty at the bounds
        if pad:
            # pad by 0.3 meter
            padding = round(0.3 / target_resolution)
            # padded_size = (map.size[0] + padding, map.size[1] + padding)
            map = ImageOps.expand(map, border=padding, fill=1)

        # [0, 255] range with occupied == 255. We will reinterpret this channel as the height in meters
        map = np.asarray(map).copy().astype(np.float32)
        map[map > 0] = max_map_height
        map = Image.fromarray(map)

        return map

    def binary_floorplan(self, ignore_below_height: float = 0.0, filled: bool = False):
        floorplan = self._floorplan_img_filled if filled else self._floorplan_img
        ignore_below_height = max(ignore_below_height, 0.001)
        map = np.asarray(floorplan).copy()
        map = map > ignore_below_height
        return Image.fromarray(map)

    def meter_to_pixels(self, xypose_meter: np.ndarray, round: bool = True, origin_W_meter: float = None, origin_H_meter: float = None, resolution: float = None):
        """In ROS the y-axis points upwards with origin at the center, in pixelspace it points downwards with origin in the top left corner"""
        if origin_W_meter is None:
            origin_W_meter = self._origin_W_meter
        if origin_H_meter is None:
            origin_H_meter = self._origin_H_meter
        if resolution is None:
            resolution = self.global_map_resolution
        xypose_meter = xypose_meter.copy()
        xypose_meter[..., 0] += origin_W_meter
        xypose_meter[..., 1] = origin_H_meter - xypose_meter[..., 1]
        xypose_pixel = xypose_meter / resolution
        if round:
            xypose_pixel = np.round(xypose_pixel).astype(np.int)
        return xypose_pixel

    def pixels_to_meter(self, xypose_pixel: np.ndarray, resolution: float = None):
        if resolution is None:
            resolution = self.global_map_resolution
        xypose_meter = xypose_pixel * resolution
        xypose_meter[..., 0] -= self._origin_W_meter
        xypose_meter[..., 1] = self._origin_H_meter - xypose_meter[..., 1]
        return xypose_meter

    def in_collision(self, xy_meters: np.ndarray, use_inflated_map: bool = False, inflation_radius_meter: float = None, ignore_below_height: float = 0.0, xy_pixels=None) -> bool:
        if inflation_radius_meter is not None:
            assert use_inflated_map
        assert (xy_meters is None) or (xy_pixels is None)
        if xy_pixels is None:
            xy_pixels = self.meter_to_pixels(xy_meters)
        if use_inflated_map:
            floor = self.binary_floorplan(filled=True) if (self._world_type == "gazebo") and self.update_from_global_costmap else None
            m = self.get_inflated_map(inflation_radius_meter=inflation_radius_meter, ignore_below_height=ignore_below_height, floorplan_img=floor)
        else:
            # asarray() returns an array HxW
            m = np.asarray(self._floorplan_img_filled)

        return m[xy_pixels[..., 1], xy_pixels[..., 0]].any()

    def _get_local_map_gt(self, current_location_tf: Transform, plan_meter: Optional[list] = None, ignore_below_height: float = 0.0) -> np.ndarray:
        """
        Return a crop around the robot, normalised to its current orientation.
        The robot is always in the center, looking towards the right
        :param current_location_tf: (List[float]) of [x, y, z, X, Y, Z, W]
        """
        yaw = quaternion_to_yaw(current_location_tf[3:])

        loc_pixel = self.meter_to_pixels(np.array(current_location_tf[:2]), round=False)
        img = np.asarray(self.binary_floorplan(ignore_below_height=ignore_below_height), dtype=np.float32)

        def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
            left = np.abs(np.minimum(0, y1))
            right = np.maximum(y2 - img.shape[0], 0)
            top = np.abs(np.minimum(0, x1))
            bottom = np.maximum(x2 - img.shape[1], 0)
            img = np.pad(img, ((left, right), (top, bottom)), mode="constant")
            y1 += left
            y2 += left
            x1 += top
            x2 += top
            return img, x1, x2, y1, y2

        def crop_fn(img: np.ndarray, center, output_size):
            h, w = np.array(output_size, dtype=int)
            x = int(center[0] - w / 2)
            y = int(center[1] - h / 2)

            y1, y2, x1, x2 = y, y + h, x, x + w
            if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
            return img[y1:y2, x1:x2]

        # already crop a bit before affine to make more efficient
        crop_multiple = 1.2
        local_map_sz = np.array(self.output_size)
        img_cropped_2x = crop_fn(img, center=loc_pixel, output_size=crop_multiple * local_map_sz)
        center_pixel_cropped_2x = crop_multiple * local_map_sz / 2.
        # take into account pixel rounding in crop_fn
        translation = loc_pixel % 1
        tf = cv2.getRotationMatrix2D(center=center_pixel_cropped_2x + translation, angle=np.rad2deg(-yaw), scale=1)
        tf[:, 2] -= translation
        local_map = cv2.warpAffine(img_cropped_2x, tf, img_cropped_2x.shape)

        local_map_cropped = crop_fn(local_map, center=center_pixel_cropped_2x, output_size=self.output_size)
        # binarize
        local_map_cropped = (local_map_cropped > 0.1)[:, :, np.newaxis]

        if plan_meter is not None:
            plan_meter = np.array(plan_meter, dtype=np.float32)
            local_map_sz_meter = self.local_map_resolution * np.array(self.output_size)
            # kick out parts outside the local map
            plan_meter = plan_meter[np.all(np.abs(plan_meter[:, :2]) < (local_map_sz_meter - self.local_map_resolution) / 2, axis=1), :]
            plan_pixels = self.meter_to_pixels(plan_meter[..., :2],
                                               round=True,
                                               origin_W_meter=local_map_sz_meter[0] / 2.,
                                               origin_H_meter=local_map_sz_meter[1] / 2.,
                                               resolution=self.local_map_resolution)

            plan_channels = np.zeros((local_map_cropped.shape[0], local_map_cropped.shape[1], 5), dtype=np.float32)
            plan_channels[plan_pixels[:, 1], plan_pixels[:, 0]] = plan_meter[:, 2:]
            # normalise the height values into [0, 1] range
            plan_channels[0] /= self.max_map_height
            local_map_cropped = np.concatenate([local_map_cropped, plan_channels], axis=2)

        return local_map_cropped

    def _get_local_map_from_costmap(self, current_location_tf):
        local_map_raw = self._ogm.get_local_map()

        # map is oriented wrt odom_combined, not with the robot -> rotate into robot orientation
        if "base" not in self._ogm.reference_frame:
            yaw_rad = quaternion_to_yaw(current_location_tf[3:])
            local_map = rotate(local_map_raw, angle=-np.rad2deg(yaw_rad), reshape=False)
        else:
            # shift by one pixel
            local_map = np.zeros_like(local_map_raw)
            local_map[:-1, 1:] = local_map_raw[1:, :-1]

        if abs(self.local_map_resolution - self._ogm.resolution) > SMALL_NUMBER:
            assert self._ogm.resolution <= self.local_map_resolution, (self._ogm.resolution, self.local_map_resolution)
            local_map = resize_to_resolution(self._ogm.resolution, self.local_map_resolution, local_map)
        return local_map[:, :, np.newaxis] >= 20

    def get_local_map(self, current_location_tf, ee_plan: Optional[List] = None, use_ground_truth: bool = False) -> np.ndarray:
        if use_ground_truth or (self._world_type == "sim"):
            return self._get_local_map_gt(current_location_tf, ee_plan)
        elif self._world_type in ["gazebo", "world"]:
            assert ee_plan is None, "Overlay not implemented yet for gazebo"
            # return self._get_local_map_gt(current_location_tf)
            return self._get_local_map_from_costmap(current_location_tf)
        else:
            raise NotImplementedError()

    def _get_global_map_from_costmap(self):
        global_map = self._ogm_global.get_local_map()

        if self._ogm_global.resolution != self.global_map_resolution:
            global_map = Image.fromarray(global_map.astype(np.bool))
            global_map = resize_to_resolution(current_resolution=self._ogm_global.resolution, target_resolution=self.global_map_resolution, map=global_map)
            global_map = np.asarray(global_map, dtype=np.float32).copy()
        else:
            global_map = global_map.astype(np.float32)
        # [0, 255] range with occupied == 255. We will reinterpret this channel as the height in meters
        # global_map[global_map > 0] = self.max_map_height
        # don't make it binary, so that the ee will prefer to drive through the middle of doors etc.
        global_map = self.max_map_height * global_map / 100
        global_map = Image.fromarray(global_map)

        # correct for map message not being published around 0 origin
        global_map = ImageChops.offset(global_map,
                                       - int((-self._ogm_global.origin.position.x - self._origin_W_meter) / self.global_map_resolution),
                                       int((-self._ogm_global.origin.position.y - self._origin_W_meter) / self.global_map_resolution))
        return global_map

    def _update_global_map_from_costmap(self):
        global_map = self._get_global_map_from_costmap()
        self._map_W_meter = self.global_map_resolution * global_map.width
        self._map_H_meter = self.global_map_resolution * global_map.height
        # take the center as the origin
        origin = (self._map_W_meter / 2, self._map_H_meter / 2, 0)
        self._origin_W_meter, self._origin_H_meter, self._origin_Z_meter = origin

        self._floorplan_img = global_map
        self._floorplan_img_filled = global_map

    def _draw_initial_base_pose(self):
        return [self.np_random.uniform(*self._initial_base_rng_x), self.np_random.uniform(*self._initial_base_rng_y), 0.,
                0., 0., self.np_random.uniform(*self._initial_base_rng_yaw)]

    def _draw_joint_values(self, robotenv, joint_distribution: Union[str, list], base_tf: list):
        """joint values for the arm"""
        if isinstance(joint_distribution, str):
            joint_values = robotenv.draw_joint_values(joint_distribution)
        else:
            joint_values = joint_distribution
            robotenv.set_joint_values(joint_values)
        assert np.all(np.array(joint_values) >= robotenv.get_joint_minima()), (robotenv.get_joint_minima(), joint_values)
        assert np.all(np.array(joint_values) <= robotenv.get_joint_maxima()), (robotenv.get_joint_maxima(), joint_values)
        robot_obs = robotenv.get_robot_obs()
        ee_pose = multiply_tfs(base_tf, robot_obs.relative_gripper_tf, False)
        return joint_values, ee_pose

    def draw_startpose_and_goal(self, robotenv, joint_distribution: Union[str, list], gripper_goal_tip_fn: Union[Callable, list]):
        base_planner_weights = PointToPointPlanner.map_to_weights(self, ignore_below_height=0)
        ee_planner_weights = PointToPointPlanner.map_to_weights(self, ignore_below_height=robotenv.robot_config['z_max'])

        real_world = (self._world_type == "world")
        i = 0
        while True:
            i += 1
            if i % 10000 == 0:
                rospy.loginfo(f"Error on map {self.map_path}")
                if robotenv.vis_env:
                    robotenv.publish_marker(base_tf, 9999, "gripper_goal", "yellow", 1.0)
                    robotenv.publish_marker(ee_tf, 5, "gripper_goal", "orange", 1.0)
                    robotenv.publish_marker(gripper_goal_tip, 10, "gripper_goal", "cyan", 1.0)
                    self.publish_floorplan_rviz()

                    f, ax = self.plot_floorplan()
                    ax.scatter(_path_pixels[:, 0], _path_pixels[:, 1])

                    for n in range(len(path_meters)):
                        x, y = path_meters[n]
                        robotenv.publish_marker([x, y, 0.0, 0, 0, 0, 1.0], 888+n, "gripper_goal", "cyan", 1.0)

                # sample a different map and try again
                self.map_reset()
                if i % 30000 == 0:
                    raise RuntimeError(f"Could not draw a goal without collision on map {self.map_path}.\n"
                                       f"{base_tf}, {ee_tf}, {joint_values}, {gripper_goal_tip}")

            # base that is not in collision
            if real_world:
                base_tf = robotenv.get_basetf_world()
                rospy.loginfo_throttle(0.5, f"Base_tf from world: {np.round(base_tf, 3)}")
            else:
                base_tf = self._draw_initial_base_pose()
                base_size = robotenv.robot_base_size[0] / 2
                if len(robotenv.robot_base_size) == 2:
                    base_size *= np.sqrt(2)

                # mask = Image.new(self._floorplan_img.mode, self._floorplan_img.size)
                # l = robotenv.robot_base_size[0]
                # w = robotenv.robot_base_size[1] if len(robotenv.robot_base_size) > 1 else robotenv.robot_base_size[0]
                # assert len(base_tf) == 6, "o/w need to calculate yaw from quaternion first"
                # self._add_rectangle(loc_x_meter=base_tf[0], loc_y_meter=base_tf[1],
                #                     length_meter=l + 0.05, width_meter=w + 0.05,
                #                     height_meter=1.0, angle_radian=base_tf[6],
                #                     floorplan=mask, filled=True)
                # if np.any(np.asarray(mask) * np.asarray(self._floorplan_img_filled)):
                #     continue
                if self.in_collision(np.array(base_tf[:2]), use_inflated_map=True,
                                     inflation_radius_meter=base_size + 0.07):
                    continue

            # ee that is not in collision
            joint_values, ee_tf = self._draw_joint_values(robotenv, joint_distribution, base_tf)
            if not real_world and self.in_collision(np.array(ee_tf[:2]), use_inflated_map=True):
                continue

            # ensure no arm collisions (i.e. no obstacle (wall) between base and ee poses)
            if not real_world and not isinstance(self, DummyMap):
                path_meters, _path_pixels, ee_cost = PointToPointPlanner.calc_xyplan(map=self,
                                                                                    start=base_tf[:2],
                                                                                    goal=ee_tf[:2],
                                                                                    weights=ee_planner_weights,
                                                                                    ignore_below_height=None)
                # as a heuristic just assume that the multiple is the number of collisions
                ee_collisions = ee_cost.sum() / PointToPointPlanner.OCCUPIED_COST
                if ee_collisions > 0.05:
                    continue

            # gripper goal that is not in collision
            if isinstance(gripper_goal_tip_fn, list):
                gripper_goal_tip = gripper_goal_tip_fn
            else:
                gripper_goal_tip = gripper_goal_tip_fn(base_tf)
                # if self.in_collision(np.array(gripper_goal_tip[:2]), use_inflated_map=True):
                if self.in_collision(np.array(gripper_goal_tip[:2]), use_inflated_map=True, ignore_below_height=gripper_goal_tip[2] - PointToPointPlanner.HEIGHT_INFLATION):
                    if real_world:
                        rospy.loginfo_throttle(0.5, f"Sampled gripper_goal_tip: {np.round(gripper_goal_tip, 3)} is in collision")
                    continue

            # ensure there exists a collision-free path to the goal (for the base, i.e. at height 0)
            if not real_world and not isinstance(self, DummyMap):
                path_meters, _path_pixels, ee_cost = PointToPointPlanner.calc_xyplan(map=self,
                                                                                    start=base_tf[:2],
                                                                                    goal=robotenv.tip_to_gripper_tf(gripper_goal_tip)[:2],
                                                                                    weights=base_planner_weights,
                                                                                    ignore_below_height=None)
                # as a heuristic just assume that the multiple is the number of collisions
                # ignore collisions on the last few centimeters to allow the gripper goal to be above an object
                ignore_last_cells = round(0.5 / self.global_map_resolution)
                nr_collisions = ee_cost[:-ignore_last_cells].sum() / PointToPointPlanner.OCCUPIED_COST
                if nr_collisions > 0.33:
                    continue

            # all conditions met
            return base_tf, joint_values, gripper_goal_tip

    def map_reset(self):
        # TODO: clear costmap after each reset if using gazebo? Using smth like this
        #   rosservice call /move_base_node/clear_costmaps "{}"
        self.clear()
        self._floorplan_img = self._floorplan_img_orig.copy()
        self._floorplan_img_filled = self._floorplan_img_orig.copy()
        self.inflated_map = dict()

        if self._world_type != 'sim':
            self._ogm = OccupancyGridManager(topic="/costmap_node/costmap/costmap", subscribe_to_updates=True)

        if self.update_from_global_costmap:
            self._ogm_global = OccupancyGridManager(topic="/costmap_node_global/costmap/costmap", subscribe_to_updates=True)
            # don't do this in gazebo, as during the reset the robot is in a different location
            if self._world_type == 'world':
                self._update_global_map_from_costmap()
            else:
                global_map = self._get_global_map_from_costmap()
                assert self._map_W_meter == self.global_map_resolution * global_map.width, "Map sizes not matching -> spawned objects / collision checking might be off"
                assert self._map_W_meter == self.global_map_resolution * global_map.height, "Map sizes not matching -> spawned objects / collision checking might be off"


    def clear(self):
        if self._ogm is not None:
            self._ogm.close()
        if self._ogm_global is not None:
            self._ogm_global.close()

    def update(self, dt: float) -> bool:
        """Update step called at every env step, e.g. for dynamic obstacles. Returns true if the map changed."""
        map_changed = False
        if self.update_from_global_costmap:
            self._update_global_map_from_costmap()
            # clear cache of inflated maps
            self.inflated_map.clear()
            map_changed = True
        return map_changed

    def _build_occgrid_msg(self, floorplan_img: Union[Image.Image, np.ndarray], origin: Tuple[float, float] = None,
                           q: Quaternion = None, tf: list = None, resolution: float = None) -> OccupancyGrid:
        if origin is None:
            origin_W_meter, origin_H_meter = self._origin_W_meter, self._origin_H_meter
        else:
            assert len(origin) == 2, origin
            origin_W_meter, origin_H_meter = origin

        if q is None:
            q = Quaternion(0, 0, 0, 1)
        if resolution is None:
            resolution = self.global_map_resolution

        if isinstance(floorplan_img, Image.Image):
            floorplan_img = np.asarray(floorplan_img, dtype=np.int)

        occ_grid = OccupancyGrid()
        occ_grid.header.frame_id = self._map_frame_rviz
        occ_grid.header.seq = 1
        # m/cell
        occ_grid.info.resolution = resolution
        occ_grid.info.width = floorplan_img.shape[1]
        occ_grid.info.height = floorplan_img.shape[0]
        # origin of cell (0, 0)
        if tf is not None:
            transformed = multiply_tfs(tf, [-origin_W_meter, -origin_H_meter, 0.0, q.x, q.y, q.z, q.w], False)
            occ_grid.info.origin = Pose(Point(*transformed[:3]), Quaternion(*transformed[3:]))
        else:
            occ_grid.info.origin = Pose(Point(-origin_W_meter, -origin_H_meter, 0), q)
        occ_grid.data = np.flipud(floorplan_img.astype(np.int)).flatten().tolist()
        return occ_grid

    def publish_floorplan_rviz(self, map=None, inflated_map=None, topic: str = 'rl_map', publish_inflated_map: bool = True):
        if map is None:
            map = np.asarray(self._floorplan_img) / self.max_map_height * 30
        if (inflated_map is None) and publish_inflated_map:
            inflated_map = self.get_inflated_map() > 0

        occ_grid = self._build_occgrid_msg(map)
        occ_grid.header.stamp = rospy.get_rostime()
        occ_grid.info.map_load_time = rospy.get_rostime()

        occ_grid_pub = rospy.Publisher(topic, OccupancyGrid, queue_size=1, latch=True)
        occ_grid_pub.publish(occ_grid)

        if publish_inflated_map:
            occ_grid_inflated = self._build_occgrid_msg(inflated_map)
            occ_grid_pub = rospy.Publisher('rl_map_inflated', OccupancyGrid, queue_size=1, latch=True)
            occ_grid_pub.publish(occ_grid_inflated)

    def get_inflated_map(self, cost_scaling_factor: float = 0.0, inflation_radius_meter: float = None, ignore_below_height: float = 0.0, floorplan_img=None) -> np.ndarray:
        if (floorplan_img is None) and self.update_from_global_costmap:
            return np.asarray(self._floorplan_img_filled) / self.max_map_height
        if inflation_radius_meter is None:
            inflation_radius_meter = self.inflation_radius_meter
        elif inflation_radius_meter == 0.0:
            return np.asarray(self.binary_floorplan(ignore_below_height=0.0))

        ignore_below_height = max(ignore_below_height, 0.001)

        cached_map = self.inflated_map.get((inflation_radius_meter, ignore_below_height, cost_scaling_factor), None)
        if (floorplan_img is None) and (cached_map is not None):
            return cached_map
        else:
            do_cache = (floorplan_img is None)
            if floorplan_img is None:
                floorplan_img = self.binary_floorplan(ignore_below_height=ignore_below_height)
            occ_grid = self._build_occgrid_msg(floorplan_img)

            origin_vector = [occ_grid.info.origin.position.x, occ_grid.info.origin.position.y, occ_grid.info.origin.position.z,
                             occ_grid.info.origin.orientation.x, occ_grid.info.origin.orientation.y, occ_grid.info.origin.orientation.z, occ_grid.info.origin.orientation.w]

            inflated_occ_grid = get_inflated_map(occ_grid.data,
                                                 occ_grid.info.resolution,
                                                 occ_grid.info.width,
                                                 occ_grid.info.height,
                                                 origin_vector,
                                                 occ_grid.header.frame_id,
                                                 inflation_radius_meter,
                                                 cost_scaling_factor)
            inflated_occ_grid = np.array(inflated_occ_grid).reshape((floorplan_img.size[1], floorplan_img.size[0]))
            inflated_occ_grid = np.flipud(inflated_occ_grid)
            # into [0, 1] range
            assert np.max(inflated_occ_grid) <= 100, np.max(inflated_occ_grid)
            inflated_occ_grid = inflated_occ_grid / 100

            # print((inflated_occ_grid > 0).sum())
            # self.publish_floorplan_rviz(inflated_occ_grid > 0, publish_inflated_map=False)
            # cache the inflated map
            if do_cache:
                self.inflated_map[(inflation_radius_meter, ignore_below_height, cost_scaling_factor)] = inflated_occ_grid
            return inflated_occ_grid

    @staticmethod
    def _plot_map(map, show: bool = False):
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(np.asarray(map).squeeze())
        if show:
            plt.show()
        return f, ax

    def plot_floorplan(self, show: bool = False, location=None):
        f, ax = self._plot_map(self._floorplan_img, show)
        if location is not None:
            location_pixels = self.meter_to_pixels(np.array(location)[..., :2])
            # NOTE: prob. not vectorised yet
            theta = quaternion_to_yaw(np.array(location)[..., 3:])
            # scale arrow length to a percentage of local map size
            c = 0.1 * np.array(self.output_size[:2])
            ax.arrow(location_pixels[..., 0], location_pixels[..., 1],
                     c[0] * np.cos(theta), c[1] * -np.sin(theta),
                     color='cyan', width=1)
        return f, ax

    def plot_local_map(self, location: list, show: bool = False):
        local_map = self.get_local_map(location)
        return self._plot_map(local_map, show=show)


class DummyMap(Map):
    def __init__(self, world_type, global_map_resolution, local_map_resolution, robot_frame_id, inflation_radius):
        super(DummyMap, self).__init__(world_type=world_type, global_map_resolution=global_map_resolution, local_map_resolution=local_map_resolution,
                                       robot_frame_id=robot_frame_id, inflation_radius=inflation_radius, floorplan=(2, 2),
                                       update_from_global_costmap=False)

    def in_collision(self, xy_meters: np.ndarray, use_inflated_map: bool = False, inflation_radius_meter: float = None, ignore_below_height: float = 0.0) -> bool:
        return False

    def get_local_map(self, current_location_tf, ee_plan: Optional[List] = None, use_ground_truth: bool = False) -> np.ndarray:
        return np.zeros((1, 1, 1), dtype=np.float32)


class SceneMap(Map):
    def __init__(self,
                 global_map_resolution: float,
                 local_map_resolution: float,
                 world_type: str,
                 robot_frame_id: str,
                 inflation_radius: float,
                 update_from_global_costmap: bool,
                 floorplan=None,
                 orig_resolution: Optional[float] = None,
                 initial_base_rng_x=(0, 0),
                 initial_base_rng_y=(0, 0),
                 initial_base_rng_yaw=(0, 0),
                 pad_floorplan_img: bool = True,
                 max_map_height: float = None,
                 requires_spawn: bool = False,
                 fixed_scene_objects: Optional[List[SpawnObject]] = None):
        super(SceneMap, self).__init__(global_map_resolution=global_map_resolution,
                                       local_map_resolution=local_map_resolution,
                                       world_type=world_type,
                                       update_from_global_costmap=update_from_global_costmap,
                                       floorplan=floorplan,
                                       orig_resolution=orig_resolution,
                                       initial_base_rng_x=initial_base_rng_x,
                                       initial_base_rng_y=initial_base_rng_y,
                                       initial_base_rng_yaw=initial_base_rng_yaw,
                                       robot_frame_id=robot_frame_id,
                                       inflation_radius=inflation_radius,
                                       pad_floorplan_img=pad_floorplan_img,
                                       max_map_height=max_map_height)
        assert requires_spawn is False, "Not using this anymore"
        self.set_world(world_type)
        self.fixed_scene_objects = fixed_scene_objects or []

    def set_world(self, world_type: str):
        super(SceneMap, self).set_world(world_type)
        if (world_type == "gazebo") or self.fake_gazebo:
            self.simulator = GazeboAPI()
        elif (world_type == "sim"):
            self.simulator = DummySimulatorAPI(frame_id=self.robot_frame_id)
        elif world_type == "world":
            self.simulator = DummySimulatorAPI(frame_id=self.robot_frame_id)
        else:
            raise NotImplementedError(world_type)

    def get_varying_scene_objects(self) -> List[SpawnObject]:
        return []

    def get_scene_objects(self) -> List[SpawnObject]:
        return self.fixed_scene_objects + self.get_varying_scene_objects()

    def _get_rectangle(self, loc_x_meter, loc_y_meter, length_meter, width_meter, angle_radian):
        x, y = self.meter_to_pixels(np.array([loc_x_meter, loc_y_meter], dtype=np.float))
        width, height = round(length_meter / self.global_map_resolution), round(width_meter / self.global_map_resolution)

        half_w = width / 2
        half_h = height / 2
        rect = np.array([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
        R = np.array([[np.cos(angle_radian), -np.sin(angle_radian)],
                      [np.sin(angle_radian), np.cos(angle_radian)]])
        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    def _add_rectangle(self, loc_x_meter, loc_y_meter, length_meter, width_meter, height_meter, angle_radian, floorplan=None, filled=False):
        if floorplan is None:
            floorplan = self._floorplan_img
        draw = ImageDraw.Draw(floorplan)
        rect = self._get_rectangle(loc_x_meter=loc_x_meter, loc_y_meter=loc_y_meter,
                                   length_meter=length_meter, width_meter=width_meter,
                                   angle_radian=angle_radian)
        draw.polygon([tuple(p) for p in rect], fill=height_meter if filled else 0.0, outline=height_meter)

    def _add_cylinder(self, loc_x_meter, loc_y_meter, length_meter, width_meter, height_meter, angle_radian, floorplan=None, filled=False):
        if floorplan is None:
            floorplan = self._floorplan_img
        if height_meter == 0:
            height_meter = 999
            erase = True
        else:
            erase = False

        # pillow cannot just draw a rotated ellipse. So draw a non-rotated, rotate it and add it to the floorplan
        overlay = Image.new(floorplan.mode, floorplan.size)
        draw = ImageDraw.Draw(overlay)
        rect = self._get_rectangle(loc_x_meter=loc_x_meter, loc_y_meter=loc_y_meter,
                                   length_meter=length_meter, width_meter=width_meter,
                                   angle_radian=0.0)
        # sort corners ascending
        x = sorted([xy[0] for xy in rect])
        y = sorted([xy[1] for xy in rect])
        draw.ellipse([x[0], y[0], x[-1], y[-1]], fill=height_meter if filled else 0.0, outline=height_meter)
        center = self.meter_to_pixels(np.array([loc_x_meter, loc_y_meter], dtype=np.float)).tolist()
        rotated = overlay.rotate(np.rad2deg(angle_radian), expand=False, center=center)
        if erase:
            floorplan.paste(Image.fromarray(np.zeros_like(rotated), mode=rotated.mode), (0, 0), Image.fromarray(np.asarray(rotated) == 999))
        else:
            floorplan.paste(rotated, (0, 0), Image.fromarray(np.asarray(rotated) > 0))

    def add_shape(self, floorplan, obj, fill_object):
        yaw = quaternion_to_yaw(obj.pose.orientation)
        if obj.world_object.geometry == ObjectGeometry.RECTANGLE:
            self._add_rectangle(loc_x_meter=obj.pose.position.x, loc_y_meter=obj.pose.position.y,
                                length_meter=obj.world_object.x, width_meter=obj.world_object.y,
                                height_meter=obj.world_object.z, angle_radian=yaw,
                                floorplan=floorplan, filled=fill_object)
        elif obj.world_object.geometry == ObjectGeometry.CYLINDER:
            self._add_cylinder(loc_x_meter=obj.pose.position.x, loc_y_meter=obj.pose.position.y,
                               length_meter=obj.world_object.x, width_meter=obj.world_object.y,
                               height_meter=obj.world_object.z, angle_radian=yaw,
                               floorplan=floorplan, filled=fill_object)
        elif obj.world_object.geometry == ObjectGeometry.UNKNOWN:
            pass
        else:
            raise NotImplementedError()

    def add_objects_to_floormap(self, objects: List[SpawnObject], dt=None):
        floorplans, fill_objects = [self._floorplan_img_filled, self._floorplan_img], [True, False]
        for obj in objects:
            if isinstance(obj, DynObstacle):
                for floorplan, fill_object in zip(floorplans, fill_objects):
                    # draw on the previous position with height 0 to free up the space
                    previous = copy.deepcopy(obj)
                    previous.world_object.z = 0.0
                    self.add_shape(floorplan, previous, fill_object=True)
                if dt is not None:
                    obj.move(dt)

        for floorplan, fill_object in zip(floorplans, fill_objects):
            for obj in objects:
                if isinstance(obj.pose, Callable):
                    i = 0
                    while True:
                        pose = obj.pose()
                        corners = self._get_rectangle(loc_x_meter=pose.position.x, loc_y_meter=pose.position.y,
                                                      length_meter=obj.world_object.x, width_meter=obj.world_object.y,
                                                      angle_radian=quaternion_to_yaw(pose.orientation))
                        if not self.in_collision(xy_meters=None, xy_pixels=corners.astype(np.int)):
                            # set the sampled pose, so the task can refer to it
                            obj.pose = pose
                            break
                        i += 1
                        assert i < 50, "Could not sample a collision-free pose for object"

                self.add_shape(floorplan, obj, fill_object)
            # ensure lower height objects cannot override walls with lower height
            floorplan = Image.fromarray(np.maximum(np.asarray(floorplan), np.asarray(floorplan)))

    def update(self, dt: float) -> bool:
        """Update step called at every env step, e.g. for dynamic obstacles. Returns true if the map changed."""
        map_changed = super().update(dt)
        dyn_obstacles = [obj for obj in self.get_scene_objects() if isinstance(obj, DynObstacle)]
        if dyn_obstacles:
            if self._world_type == 'sim':
                self.add_objects_to_floormap(dyn_obstacles, dt)
                # clear cache of inflated maps
                self.inflated_map.clear()
                self.publish_floorplan_rviz(publish_inflated_map=False)
            map_changed = True
        return map_changed

    def map_reset(self):
        super().map_reset()
        self.simulator.delete_all_spawned()
        objects = self.get_scene_objects()
        # NOTE: depending where the robot is, this could lead to collision with
        #   (i) the last position the robot had before the reset
        #   (ii) the inital position the robot will take when resetting
        self.add_objects_to_floormap(objects)
        self.simulator.spawn_scene_objects(objects)

    def clear(self):
        super().clear()
        return self.simulator.clear()


class SimpleObstacleMap(SceneMap):
    def __init__(self,
                 global_map_resolution: float,
                 local_map_resolution: float,
                 world_type: str,
                 robot_frame_id: str,
                 inflation_radius: float,
                 floorplan=None,
                 orig_resolution: Optional[float] = None,
                 initial_base_rng_x=(0, 0),
                 initial_base_rng_y=(0, 0),
                 initial_base_rng_yaw=(0, 0),
                 requires_spawn: bool = False,
                 fixed_scene_objects: Optional[List[SpawnObject]] = None,
                 obstacle_spacing: float = 1.75,
                 offset_std: float = 1.75 / 4):
        super(SimpleObstacleMap, self).__init__(global_map_resolution=global_map_resolution,
                                                  local_map_resolution=local_map_resolution,
                                                  world_type=world_type,
                                                  robot_frame_id=robot_frame_id,
                                                  inflation_radius=inflation_radius,
                                                  update_from_global_costmap=False,
                                                  floorplan=floorplan,
                                                  orig_resolution=orig_resolution,
                                                  initial_base_rng_x=initial_base_rng_x,
                                                  initial_base_rng_y=initial_base_rng_y,
                                                  initial_base_rng_yaw=initial_base_rng_yaw,
                                                  requires_spawn=requires_spawn,
                                                  fixed_scene_objects=fixed_scene_objects)
        # space between obstacle centers in meters
        self._obstacle_spacing = obstacle_spacing
        self.offset_std = offset_std

    def _draw_initial_base_pose(self):
        """Draw the base pose together with the goal in the task setup, as much easier to ensure that there is a feasible path that way"""
        max_x = (self._map_W_meter - 1.0) / 2
        max_y = (self._map_H_meter - 1.0) / 2
        x = self.np_random.uniform(-max_x, max_x)
        y = self.np_random.uniform(-max_y, max_y)
        yaw = self.np_random.uniform(0, 2 * np.pi)
        return [x, y, 0] + [0, 0, yaw]

    def get_varying_scene_objects(self) -> List[SpawnObject]:
        if self._obstacle_spacing is None:
            return []

        i = 0
        objects = []
        for xi, x in enumerate(np.arange(-self._map_W_meter / 2, self._map_W_meter / 2, self._obstacle_spacing)):
            start_offset = (xi % 2) * self._obstacle_spacing / 2
            for y in np.arange(-self._map_H_meter / 2, self._map_H_meter / 2, self._obstacle_spacing):
                xrnd, yrnd = self.np_random.normal(0.0, self.offset_std, size=2)

                q = yaw_to_quaternion(self.np_random.uniform(0, np.pi))
                obstacle_pose = Pose(Point(x + xrnd, start_offset + y + yrnd, 0.24), q)

                xx = self.np_random.uniform(0.025, 1.5)
                yy = self.np_random.uniform(0.025, 1.0)
                zz = self.np_random.uniform(0.2, 0.8) if self.np_random.random() > 0.33 else self.max_map_height

                # NOTE: gazebo does not have an ellipse model, so can only use with "sim" env atm
                if (self.np_random.random() < 0.5) or (self._world_type == "gazebo") or self.fake_gazebo:
                    geometry = ObjectGeometry.RECTANGLE
                    getter_fn = get_rectangle
                else:
                    geometry = ObjectGeometry.CYLINDER
                    getter_fn = None
                obj = GazeboObject("rectangle", x=xx, y=yy, z=zz, geometry=geometry, getter_fn=getter_fn)
                objects.append(SpawnObject(f"obstacle{i}", obj, obstacle_pose))
                i += 1
        return objects
