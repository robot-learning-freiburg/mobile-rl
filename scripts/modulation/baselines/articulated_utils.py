import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import roslaunch
import rospy
import rospkg
import rosnode
from matplotlib import pyplot as plt
from ocs2_msgs.msg import mpc_observation, mysdfgrid
from scipy import ndimage

plt.style.use('seaborn')
from modulation.evaluation import Episode
from modulation.envs.env_utils import quaternion_to_yaw, calc_euclidean_tf_dist, calc_rot_dist, SMALL_NUMBER
from pybindings import multiply_tfs
from scipy.spatial.transform import Rotation

rospack = rospkg.RosPack()
MPC_OBS_TOPIC = "/mobile_manipulator_mpc_observation"


def list_controllers():
    from pr2_mechanism_msgs.srv import ListControllers, ListControllersRequest
    list_controllers_srv = rospy.ServiceProxy('/pr2_controller_manager/list_controllers', ListControllers)
    req = ListControllersRequest()
    return list_controllers_srv(req)


def load_velocity_controllers(env_name):
    if env_name == "pr2":
        controllers = list_controllers()
        if not "r_wrist_roll_velocity_controller" in controllers.controllers:
            p = rospack.get_path("ocs2_mobile_manipulator_ros")
            cli_args = [f'{p}/launch/load_velocity_controllers.launch']
            roslaunch_args = []
            roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

            parent.start()
            return parent
        else:
            return None
    # elif env_name == "hsr":
    #     running_nodes = rosnode.get_node_names()
    #     if "pseudo_velocity_controller" not in running_nodes:
    #         p = rospack.get_path("ocs2_mobile_manipulator_ros")
    #         cli_args = [f'{p}/launch/hsr_pseudo_velocity_controller.launch']
    #         roslaunch_args = []
    #         roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    #
    #         uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    #         parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    #
    #         parent.start()
    #         return parent
    #     else:
    #         return None
    # else:
    #     raise ValueError(env_name)

def get_ocs2_msg(timeout=10) -> mpc_observation:
    return rospy.wait_for_message(MPC_OBS_TOPIC, mpc_observation, timeout=timeout)


# def start_mpc_node(robot_name: str):
#     p = rospack.get_path("ocs2_mobile_manipulator_ros")
#     cli_args = [f'{p}/launch/mobile_manipulator.launch', f'robot:={robot_name}']
#     roslaunch_args = cli_args[1:]
#     roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
#
#     uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
#     parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
#
#     parent.start()


def get_grippertf_world(env):
    gripper_tf = env.get_grippertf_world("world")
    if env.env_name == 'tiago':
        gripper_tf_oriented = gripper_tf
    else:
        rot_offset_inverse = Rotation.from_quat(env.env.robot_config['gripper_to_base_rot_offset']).inv().as_quat().tolist()
        gripper_tf_oriented = copy.deepcopy(gripper_tf)
        gripper_tf_oriented[3:] = multiply_tfs(gripper_tf, [0, 0, 0] + rot_offset_inverse, False)[3:]
    return gripper_tf_oriented


# def start_mrt_node(robot_obs, ee_plan, env):
#     # msg_from_ocs = get_ocs2_msg(5)
#     joint_values_for_articulated = get_our_joint_values_matching_articulated(env.env_name, env.get_joint_names(), robot_obs.joint_values)
#
#     initialState = [robot_obs.base_tf[0], robot_obs.base_tf[1], quaternion_to_yaw(robot_obs.base_tf[3:])] + joint_values_for_articulated
#     # assert len(initialState) == len(msg_from_ocs.state.value), (len(initialState), len(msg_from_ocs.state.value))
#     rospy.set_param("/initialState", initialState)
#     eestart = ee_plan[0]
#     if isinstance(ee_plan, np.ndarray):
#         eestart = eestart.tolist()
#     rospy.set_param("/initialTarget", eestart)
#
#     package = 'ocs2_mobile_manipulator_ros'
#     executable = 'mobile_manipulator_dummy_mrt_node'
#     node = roslaunch.core.Node(package, executable)
#
#     launch = roslaunch.scriptapi.ROSLaunch()
#     launch.start()
#
#     mrt_node = launch.launch(node)
#     return mrt_node


def start_mpc_mrt_node(robot_obs, ee_plan, env):
    if (env.get_world() == "gazebo") and (env.env_name == "pr2"):
        switch_to_velocity_controllers()

    # msg_from_ocs = get_ocs2_msg(5)
    joint_values_for_articulated = get_our_joint_values_matching_articulated(env.env_name, env.get_joint_names(), robot_obs.joint_values)

    initialState = [robot_obs.base_tf[0], robot_obs.base_tf[1], quaternion_to_yaw(robot_obs.base_tf[3:])] + joint_values_for_articulated
    # assert len(initialState) == len(msg_from_ocs.state.value), (len(initialState), len(msg_from_ocs.state.value))
    rospy.set_param("/initialState", [float(i) for i in initialState])
    eestart = ee_plan[0]
    if isinstance(ee_plan, np.ndarray):
        eestart = eestart.tolist()
    rospy.set_param("/initialTarget", eestart)

    p = rospack.get_path("ocs2_mobile_manipulator_ros")
    cli_args = [f'{p}/launch/mobile_manipulator.launch', f'robot:={env.env_name}', f"is_analytical:={env.get_world() == 'sim'}"]
    roslaunch_args = cli_args[1:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

    parent.start()
    return parent


def switch_controllers(to_velocity: bool):
    # pause_physics_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    # unpause_physics_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    from pr2_mechanism_msgs.srv import SwitchController, SwitchControllerRequest
    switch_srv = rospy.ServiceProxy('/pr2_controller_manager/switch_controller', SwitchController)
    req = SwitchControllerRequest()

    vel_controllers = ["torso_lift_velocity_controller",
                       "r_elbow_flex_velocity_controller",
                       "r_forearm_roll_velocity_controller",
                       "r_shoulder_lift_velocity_controller",
                       "r_shoulder_pan_velocity_controller",
                       "r_upper_arm_roll_velocity_controller",
                       "r_wrist_flex_velocity_controller",
                       "r_wrist_roll_velocity_controller"]

    position_controllers = ["r_arm_controller", "torso_controller"]

    if to_velocity:
        req.start_controllers = vel_controllers
        req.stop_controllers = position_controllers
    else:
        req.start_controllers = position_controllers
        req.stop_controllers = vel_controllers
    req.strictness = 2

    # pause_physics_srv()
    assert switch_srv(req)
    # unpause_physics_srv()


def switch_to_velocity_controllers():
    switch_controllers(to_velocity=True)


def switch_to_position_controllers():
    switch_controllers(to_velocity=False)


def get_articulated_joint_names(robot_name: str) -> list:
    if robot_name == "pr2":
        return ["torso_lift_joint", "r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint", "r_wrist_roll_joint"]
    elif robot_name == "tiago":
        return ["torso_lift_joint", "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"]
    elif robot_name == "hsr":
        return ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
    else:
        raise NotImplementedError(robot_name)


def get_our_joint_values_matching_articulated(env_name, joint_names, joint_values):
    jv = dict(zip(joint_names, joint_values))
    return [jv[name] for name in get_articulated_joint_names(env_name)]


def set_joints_to_articulated_values(env, ocs2_msg=None):
    if ocs2_msg is None:
        ocs2_msg = get_ocs2_msg()
    joint_names = get_articulated_joint_names(env.env_name)
    assert len(ocs2_msg.state.value) == 3 + len(joint_names), (len(ocs2_msg.state.value), len(joint_names))
    joint_values = ocs2_msg.state.value[3:]
    jv = dict(zip(joint_names, joint_values))
    jv["world_joint/x"] = ocs2_msg.state.value[0]
    jv["world_joint/y"] = ocs2_msg.state.value[1]
    jv["world_joint/theta"] = ocs2_msg.state.value[2]
    env.set_joint_values(jv, False)

    limit_joints = np.logical_not(env.no_limit_joints)
    joint_values = np.array(joint_values)

    minima_limited = env.get_joint_minima()
    minima_limited[limit_joints == False] = - np.infty
    maxima_limited = env.get_joint_maxima()
    maxima_limited[limit_joints == False] = np.infty

    minima_limited = np.array(get_our_joint_values_matching_articulated(env.env_name, env.get_joint_names(), minima_limited))
    maxima_limited = np.array(get_our_joint_values_matching_articulated(env.env_name, env.get_joint_names(), maxima_limited))

    violating_min = (joint_values < minima_limited + SMALL_NUMBER)
    violating_max = (joint_values > maxima_limited - SMALL_NUMBER)
    if np.any(violating_min):
        print(f"Violating joint min limits {np.array(joint_names)[violating_min]}: {np.round(joint_values[violating_min], 3)} < {np.round(minima_limited[violating_min], 3)}")
    if np.any(violating_max):
        print(f"Violating joint max limits {np.array(joint_names)[violating_max]}: {np.round(joint_values[violating_max], 3)} > {np.round(maxima_limited[violating_max], 3)}")
    violating_joint_limit = np.any(violating_min) or np.any(violating_max)
    return violating_joint_limit


class BaseCollisionMarker(NamedTuple):
    radius: float
    square_base: bool
    corner_radius: float
    diagonal_radius: float

    values = {
        "pr2": {"radius": 0.38,
                "square_base": True,
                "diagonal_radius": 0.40,
                "corner_radius": 0.1},
        "tiago": {"radius": 0.32,
                  "square_base": False,
                  "diagonal_radius": 0.0,
                  "corner_radius": 0.0},
        "hsr": {"radius": 0.285,
                "square_base": False,
                "diagonal_radius": 0.0,
                "corner_radius": 0.0},
    }

    @staticmethod
    def get_square_base_corners(base_tf, diagonal_radius):
        x, y, theta = base_tf[0], base_tf[1], quaternion_to_yaw(base_tf[3:])
        return [[x + diagonal_radius * np.cos(theta + fraction * np.pi), y + diagonal_radius * np.sin(theta + fraction * np.pi)] for fraction in [0.25, 0.75, 1.25, 1.75]]

    @staticmethod
    def publish_basecollision(env, base_tf, frame_id="world"):
        def rviz_circle_marker(env, x, y, radius, marker_id):
            env.publish_marker([x, y, 0, 0, 0, 0, 1], marker_id=marker_id, marker_scale=(2 * radius, 2 * radius, 0.1),
                               namespace="base_collision", color="cyan", alpha=0.5, geometry="cylinder", frame_id=frame_id)
        params = BaseCollisionMarker.values[env.env_name]

        marker_id = 0
        rviz_circle_marker(env, base_tf[0], base_tf[1], params["radius"], marker_id)
        if params["square_base"]:
            for corner in BaseCollisionMarker.get_square_base_corners(base_tf, params["diagonal_radius"]):
                marker_id += 1
                rviz_circle_marker(env, corner[0], corner[1], params["corner_radius"], marker_id)


@dataclass
class ArticulatedEpisode:
    ee_plan: np.ndarray = field(default_factory=lambda: np.empty((1, 7)))
    gripper_tfs: list = field(default_factory=list)
    base_tfs: list = field(default_factory=list)
    base_collisions: list = field(default_factory=list)
    joint_limit_violations: list = field(default_factory=list)
    dists_to_motion: list = field(default_factory=lambda: np.empty(0))
    rot_dists_to_motion: list = field(default_factory=lambda: np.empty(0))

    def add_ee_plan(self, ee_plan):
        self.ee_plan = np.concatenate([self.ee_plan, ee_plan], axis=0)

    def add_step(self, gripper_tf, base_tf, base_collision: bool, violating_joint_limit: bool):
        self.gripper_tfs.append(gripper_tf)
        self.base_tfs.append(base_tf)
        self.base_collisions.append(base_collision)
        self.joint_limit_violations.append(violating_joint_limit)
        dist, rot_dist = Episode.calc_gripper_dists_to_motion(gripper_tfs_achieved=np.array([gripper_tf]),
                                                              gripper_tfs_desired=np.array(self.ee_plan))
        self.dists_to_motion = np.append(self.dists_to_motion, dist)
        self.rot_dists_to_motion = np.append(self.rot_dists_to_motion, rot_dist)

    def gripper_dists_to_motion_below_thresh(self, max_dist: float, max_rot_dist: float) -> bool:
        return (self.dists_to_motion.max() < max_dist) and (self.rot_dists_to_motion.max() < max_rot_dist)

    @property
    def nr_base_collisions(self):
        return np.sum(self.base_collisions)

    @property
    def nr_joint_limit_violations(self):
        return np.sum(self.joint_limit_violations)

    @property
    def goal_reached(self):
        return calc_euclidean_tf_dist(self.ee_plan[-1], self.gripper_tfs[-1]) < 0.1 and calc_rot_dist(self.ee_plan[-1], self.gripper_tfs[-1]) < 0.05

    def success(self, max_dist: float, max_rot_dist: float):
        return (self.goal_reached
                and (self.nr_base_collisions == 0)
                and (self.nr_joint_limit_violations == 0)
                and self.gripper_dists_to_motion_below_thresh(max_dist, max_rot_dist))

    def __len__(self):
        return len(self.gripper_tfs)


def publish_sdf(map):
    """For articulated baseline: publish the ground-truth floorplan as signed distance field"""
    if (map._world_type != "sim") and map.update_from_global_costmap:
        # global map from costmap is inflated. This threshold doesn't count the inflation
        f = map.binary_floorplan(ignore_below_height=2.97, filled=True)
    else:
        f = map.binary_floorplan(filled=True)
    sdf = ndimage.distance_transform_edt(1 - np.array(f))
    sdf *= map.global_map_resolution
    sdf_dy, sdf_dx = np.gradient(sdf)
    sdf_dy *= -1
    # plt.imshow(sdf)

    def publish_sdf(data, topic: str):
        nav_occ_grid = map._build_occgrid_msg(data)

        occ_grid = mysdfgrid()
        occ_grid.data = np.flipud(data.astype(np.float32)).flatten().tolist()
        occ_grid.info = nav_occ_grid.info

        occ_grid.header.stamp = rospy.get_rostime()
        occ_grid.info.map_load_time = rospy.get_rostime()

        occ_grid_pub = rospy.Publisher(topic, mysdfgrid, queue_size=1, latch=True)
        occ_grid_pub.publish(occ_grid)

    publish_sdf(data=sdf, topic='sdf')
    publish_sdf(data=sdf_dx, topic='sdf_dx')
    publish_sdf(data=sdf_dy, topic='sdf_dy')

    # RVIZ: can only visualise OccupancyMap msgs, so rescale and publish there as well
    # for now use the available value range the best we can by rescaling
    def rescale_to_int8(data):
        return (100 * np.minimum(data, 1.0)).astype(np.int8)

    def rescale_gradients_to_int8(data, max_resolution: float = 0.025):
        # into [0, 1] range
        data = 0.5 + (data / (2 * max_resolution))
        return np.clip((100 * data).astype(np.int8), 0, 100)

    map.publish_floorplan_rviz(map=rescale_to_int8(sdf), publish_inflated_map=False, topic='sdf_rviz')
    map.publish_floorplan_rviz(map=rescale_gradients_to_int8(sdf_dx), publish_inflated_map=False, topic='sdf_rviz_dx')
    map.publish_floorplan_rviz(map=rescale_gradients_to_int8(sdf_dy), publish_inflated_map=False, topic='sdf_rviz_dy')
