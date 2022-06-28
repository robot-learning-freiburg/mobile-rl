import os
from typing import Optional, Tuple, List, NamedTuple
import numpy as np
import pyastar
import rospy
from PIL import Image
from matplotlib import pyplot as plt
from pybindings import RobotObs, EEObs, normscale_vel, eeplan_to_eeobs, angle_shortest_path, GaussianMixtureModel, slerp_single
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from modulation.envs.env_utils import SMALL_NUMBER, quaternion_to_yaw, \
    calc_euclidean_tf_dist, calc_rot_dist, resize_to_resolution, IDENTITY_TF, clamp_angle_minus_pi_pi
from pybindings import tip_to_gripper_goal
from pybindings import interpolate_z as interpolate_z_cpp

MIN_PLANNER_VELOCITY = 0.001
MAX_PLANNER_VELOCITY = 0.1
# rad / second, not enforced in all the planners so far
MAX_PLANNER_ANGULAR_VELOCITY = 0.30
# also defined in robot_env.cpp!
TIME_STEP_TRAIN = 0.1
SQROOT_TWO = 1.414

# transform of length 7
EEPlan = List

class NextEEPlan(NamedTuple):
    next_gripper_tf: list


def interpolate_plan(start_meter: list, goal_meter: list, resolution: float):
    dist = np.linalg.norm(np.array(start_meter[:2]) - goal_meter[:2])
    # just a straight line from current location
    n = int(np.ceil(dist / resolution))
    plan_meter_x = np.linspace(start_meter[0], goal_meter[0], n)
    plan_meter_y = np.linspace(start_meter[1], goal_meter[1], n)
    plan_meter_z = np.linspace(start_meter[2], goal_meter[2], n)

    key_rots = Rotation.from_quat([start_meter[3:], goal_meter[3:]])
    key_times = [0, dist]
    slerp = Slerp(key_times, key_rots)
    subgoal_orientations = slerp(np.arange(0, dist, resolution)).as_quat()

    plan_meter = np.concatenate([plan_meter_x[:, np.newaxis],
                                 plan_meter_y[:, np.newaxis],
                                 plan_meter_z[:, np.newaxis],
                                 subgoal_orientations[:len(plan_meter_z)]], axis=-1)
    return plan_meter


def interpolate_z(cum_dists, obstacle_zs, current_z, goal_z, max_map_height):
    """Assumes that the inputs start from the current position and end with the goal"""
    assert len(cum_dists) == len(obstacle_zs), (len(cum_dists), len(obstacle_zs))
    assert cum_dists[0] == 0.0, cum_dists[0]

    obstacle_zs = np.maximum(obstacle_zs, min(current_z, goal_z))
    obstacle_zs[-1] = goal_z
    obstacle_zs[0] = current_z

    # z_level = current_z
    # ts, zs = [0.], [current_z]
    # for i, t, z in zip(range(len(cum_dists[1:])), cum_dists[1:], obstacle_zs[1:]):
    #     if (z > z_level + SMALL_NUMBER) and (z < max_map_height):
    #         ts.append(t)
    #         zs.append(z)
    #         z_level = z
    #     elif (z < z_level - SMALL_NUMBER):
    #         if i > 0:
    #             ts.append(t)
    #             # add with last z so we interpolate straight ahead to the end of the previous obstacle first
    #             zs.append(z_level)
    #         z_level = z
    # ts.append(cum_dists[-1])
    # zs.append(obstacle_zs[-1])

    ts, zs = interpolate_z_cpp(cum_dists, obstacle_zs, current_z, max_map_height)
    # NOTE: needs to use eps == SMALL_NUMBER to give the exactly same result!
    # assert np.all(np.abs(np.array(ts) - ts2) < 0.000001)
    # assert np.all(np.abs(np.array(zs) - zs2) < 0.000001)

    f = interp1d(ts, zs, kind='linear')
    interpolated_obstacle_zs = f(cum_dists)

    return interpolated_obstacle_zs


def repeat_plan_for_pause(plan_meter, time_planner_prepause: float, head_start: float):
    assert isinstance(plan_meter, np.ndarray), type(plan_meter)
    if time_planner_prepause < head_start:
        n_pause_steps = (head_start - time_planner_prepause) // TIME_STEP_TRAIN
        plan_meter = np.concatenate([np.repeat(plan_meter[:1], n_pause_steps, axis=0), plan_meter])
    return plan_meter


def get_vec_to_goal_vel(from_tf, to_tf, min_vel: float, max_vel: float):
    vec_to_goal = np.array(to_tf[:3]) - from_tf[:3]
    return normscale_vel(vec_to_goal / 100.0, min_vel, max_vel)


def get_max_angle_slerp_pct(from_tf, to_tf, dt: float):
    max_angle = dt * MAX_PLANNER_ANGULAR_VELOCITY
    angle = angle_shortest_path(from_tf[3:], to_tf[3:])
    slerp_pct = np.clip(max_angle / max(angle, SMALL_NUMBER), 0.0, 1.0)
    return slerp_pct


class EEPlanner:
    def __init__(self,
                 gripper_goal_tip,
                 head_start: float,
                 map,
                 robot_config: dict,
                 success_thres_dist: float,
                 success_thres_rot: float):
        self._head_start = head_start
        self._map = map
        self._robot_config = robot_config
        self._success_thres_dist = success_thres_dist
        self._success_thres_rot = success_thres_rot
        self.set_goal(gripper_goal_tip)

    def tip_to_gripper_goal(self, gripper_goal_tip):
        return tip_to_gripper_goal(gripper_goal_tip, self._robot_config["tip_to_gripper_offset"])

    def set_goal(self, gripper_goal_tip):
        self.gripper_goal_tip = gripper_goal_tip
        self.gripper_goal_wrist = self.tip_to_gripper_goal(gripper_goal_tip)

    def reset(self,
              robot_obs: RobotObs,
              is_analytic_env: bool,
              plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        raise NotImplementedError()

    def step(self, robot_obs: RobotObs, learned_vel_norm: float) -> Tuple[EEObs, float, List]:
        raise NotImplementedError()

    def generate_obs_step(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        raise NotImplementedError()

    def update_weights(self, robot_obs: RobotObs):
        """Function to update weights or similar in case the map changed (e.g. dynamic obstacles)"""
        raise NotImplementedError()


class EEPlannerFullPython(EEPlanner):
    def __init__(self,
                 gripper_goal_tip,
                 head_start: float,
                 map,
                 robot_config: dict,
                 success_thres_dist: float,
                 success_thres_rot: float):
        super(EEPlannerFullPython, self).__init__(gripper_goal_tip,
                                                  head_start,
                                                  map,
                                                  robot_config,
                                                  success_thres_dist=success_thres_dist,
                                                  success_thres_rot=success_thres_rot)

    @property
    def dt(self) -> float:
        if self._is_analytic_env:
            dt = TIME_STEP_TRAIN
        else:
            dt = rospy.get_time() - self._rostime_at_start - self._time_planner_prepause
        return dt

    def _update_time(self):
        dt = self.dt
        self._time_planner_prepause += dt
        if not self._is_analytic_env:
            if dt > 0.025:
                print(f"dt: {dt:.3f}, time_prepause: {self._time_planner_prepause:.2f}, time_planner: {self.time_planner:.2f}")
        return dt

    @property
    def time_planner(self):
        return max(self._time_planner_prepause - self._head_start, 0.0)

    def _in_start_pause(self, time: float = None):
        if time is None:
            time = self._time_planner_prepause
        return time < self._head_start

    def _is_done(self, gripper_tf):
        dist_to_goal = calc_euclidean_tf_dist(self.gripper_goal_wrist, gripper_tf)
        is_close = (dist_to_goal < self._success_thres_dist)
        if is_close:
            rot_distance = calc_rot_dist(gripper_tf, self.gripper_goal_wrist)
            is_close &= (rot_distance < self._success_thres_rot)
        return is_close

    def _reset(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        raise NotImplementedError()

    def reset(self,
              robot_obs: RobotObs,
              is_analytic_env: bool,
              plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        self._prev_plan = NextEEPlan(robot_obs.gripper_tf)
        self._initial_dist_to_gripper_goal = calc_euclidean_tf_dist(robot_obs.gripper_tf, self.gripper_goal_wrist)
        self._is_analytic_env = is_analytic_env
        self._time_planner_prepause = 0.0
        self.plan_horizon_meter = plan_horizon_meter

        # make sure we plan in the latest map
        self._map.update(dt=0.0)

        ee_obs, ee_plan = self._reset(robot_obs, self.plan_horizon_meter)

        self._rostime_at_start = 0.0 if is_analytic_env else rospy.get_time()

        return ee_obs, ee_plan

    def _cpp_step(self, time: float, dt: float, prev_plan: NextEEPlan, robot_obs: RobotObs, learned_vel_norm: float) -> NextEEPlan:
        raise NotImplementedError()

    def _update_plan_helper(self, robot_obs: RobotObs, dt: float):
        """Function to set additional state whenever we update the previous plan"""
        pass

    def step(self, robot_obs: RobotObs, learned_vel_norm: float) -> Tuple[EEObs, float, List]:
        last_dt = self._update_time()
        pause_gripper = self._in_start_pause()
        if pause_gripper:
            gripper_plan = self._prev_plan
        else:
            max_dt = 0.5
            last_dt_clipped = min(last_dt, max_dt)
            if last_dt > max_dt:
                print(f"dt {last_dt:.3f} > {max_dt}, clipping it for end-effector motion")
            gripper_plan = self._cpp_step(time=self.time_planner,
                                          dt=last_dt_clipped,
                                          prev_plan=self._prev_plan,
                                          robot_obs=robot_obs,
                                          learned_vel_norm=learned_vel_norm)
            self._update_plan_helper(robot_obs, dt=last_dt_clipped)
            self._prev_plan = gripper_plan

        reward = 0.0
        done = False
        return eeplan_to_eeobs(gripper_plan.next_gripper_tf, robot_obs, reward, done), last_dt

    def generate_obs_step_ee_obs(self, robot_obs: RobotObs) -> EEObs:
        pause_gripper = self._in_start_pause()
        gripper_plan = self._cpp_step(time=self.time_planner,
                                      dt=0.0 if pause_gripper else TIME_STEP_TRAIN,
                                      prev_plan=self._prev_plan,
                                      robot_obs=robot_obs,
                                      learned_vel_norm=-1)
        reward = 0.0
        done = self._is_done(robot_obs.gripper_tf)
        return eeplan_to_eeobs(gripper_plan.next_gripper_tf, robot_obs, reward, done)

    @staticmethod
    def get_min_max_vel(learned_vel_norm: float, dt: float):
        if learned_vel_norm >= 0.0:
            min_vel, max_vel = learned_vel_norm, learned_vel_norm
        else:
            min_vel, max_vel = MIN_PLANNER_VELOCITY, MAX_PLANNER_VELOCITY
        return dt * min_vel, dt * max_vel


class LinearPlannerFullPython(EEPlannerFullPython):
    def __init__(self,
                 gripper_goal_tip,
                 head_start: float,
                 map,
                 robot_config: dict,
                 success_thres_dist: float,
                 success_thres_rot: float,
                 max_vel: Optional[float] = None):
        super(LinearPlannerFullPython, self).__init__(gripper_goal_tip,
                                                      head_start,
                                                      map,
                                                      robot_config,
                                                      success_thres_dist=success_thres_dist,
                                                      success_thres_rot=success_thres_rot)
        self.max_vel = max_vel

    def _reset(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        key_rots = Rotation.from_quat([robot_obs.gripper_tf[3:], self.gripper_goal_wrist[3:]])
        self.slerp = Slerp([0, 1], key_rots)
        return self.generate_obs_step(robot_obs, plan_horizon_meter)

    def _cpp_step(self, time: float, dt: float, prev_plan: NextEEPlan, robot_obs: RobotObs, learned_vel_norm: float) -> NextEEPlan:
        min_vel, max_vel = self.get_min_max_vel(learned_vel_norm, dt)
        if self.max_vel is not None:
            max_vel = min(max_vel, self.max_vel)

        # new xy velocities based on distance to current Goal
        planned_gripper_vel = get_vec_to_goal_vel(from_tf=prev_plan.next_gripper_tf, to_tf=self.gripper_goal_wrist, min_vel=min_vel, max_vel=max_vel)
        planned_gripper_xyz = np.array(prev_plan.next_gripper_tf[:3]) + np.array(planned_gripper_vel)

         # new rotations: interpolated from start to goal based on distance achieved so far
        dist_to_goal_post = calc_euclidean_tf_dist(self.gripper_goal_wrist[:3], planned_gripper_xyz)
        slerp_pct = np.clip(1.0 - dist_to_goal_post / self._initial_dist_to_gripper_goal, 0.0, 1.0)
        planned_gripper_q = self.slerp([slerp_pct]).as_quat()[0]

        return NextEEPlan(planned_gripper_xyz.tolist() + planned_gripper_q.tolist())

    def generate_obs_step(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        ee_obs = self.generate_obs_step_ee_obs(robot_obs=robot_obs)
        plan_meter = interpolate_plan(start_meter=ee_obs.next_gripper_tf,
                                      goal_meter=self.gripper_goal_wrist,
                                      resolution=self._map.global_map_resolution)
        plan_meter = repeat_plan_for_pause(plan_meter, time_planner_prepause=self._time_planner_prepause, head_start=self._head_start)
        subgoal_idx = min(int(plan_horizon_meter / self._map.global_map_resolution), len(plan_meter))
        return ee_obs, plan_meter[:subgoal_idx]

    def update_weights(self, robot_obs: RobotObs):
        """Function to update weights or similar in case the map changed (e.g. dynamic obstacles)"""
        return


class GMMPlannerWrapper(EEPlannerFullPython):
    def __init__(self,
                 gripper_goal_tip,
                 head_start: float,
                 map,
                 robot_config: dict,
                 success_thres_dist: float,
                 success_thres_rot: float,
                 gmm_model_path: str,
                 interpolate_z: bool = True):
        assert os.path.exists(gmm_model_path), f"Path {gmm_model_path} doesn't exist"
        self._gmm_model_path = gmm_model_path
        self.next_z_subgoal = None
        self.interp_timestep = 18 * TIME_STEP_TRAIN
        self._gmm = GaussianMixtureModel(MAX_PLANNER_ANGULAR_VELOCITY)
        self._gmm.load_from_file(str(self._gmm_model_path))

        self.interpolate_z = interpolate_z

        super(GMMPlannerWrapper, self).__init__(gripper_goal_tip,
                                                head_start,
                                                map,
                                                robot_config,
                                                success_thres_dist=success_thres_dist,
                                                success_thres_rot=success_thres_rot)

    def set_goal(self, gripper_goal_tip):
        super(GMMPlannerWrapper, self).set_goal(gripper_goal_tip)
        self._gmm.adapt_model(self.gripper_goal_tip, [self._robot_config["gmm_base_offset"], 0., 0.])
        self.gmm_mus_ee, self.gmm_mus_base = [], []
        for i, mu in enumerate(self._gmm.get_mus()):
            if (i % 2 == 0):
                self.gmm_mus_ee.append(mu)
            else:
                self.gmm_mus_base.append(mu)

    @staticmethod
    def obj_origin_to_tip(obj_origin, gmm_model_path):
        assert os.path.exists(gmm_model_path), f"Path {gmm_model_path} doesn't exist"
        max_rot, gmm_base_offset = 0.1, 0.0
        gmm = GaussianMixtureModel(max_rot)
        gmm.load_from_file(str(gmm_model_path))
        gmm.adapt_model(IDENTITY_TF, [gmm_base_offset, 0., 0.])
        return gmm.obj_origin_to_tip(obj_origin)

    def _reset(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        self._gmm_time_offset = 0.0

        return self.generate_obs_step(robot_obs, plan_horizon_meter)

    def _update_plan_helper(self, robot_obs: RobotObs, dt: float):
        """'Hack' to not start gmm motion when far away from the first centroid"""
        thresh = 2.5
        msg = """This should check whether we are already past the first mu or not, otherwise could activate again 
        later. Will work like this as long as the distance between the mu's is never above 2.5"""
        assert calc_euclidean_tf_dist(self.gmm_mus_ee[0], self.gmm_mus_ee[-1]) < thresh, msg
        if calc_euclidean_tf_dist(self.gmm_mus_ee[0], self._prev_plan.next_gripper_tf) > thresh:
            self._gmm_time_offset += dt

    def _cpp_step(self, time: float, dt: float, prev_plan: NextEEPlan, robot_obs: RobotObs, learned_vel_norm: float) -> NextEEPlan:
        # NOTE: not multiplying by dt here, but rather in cpp. Also min_vel is 0 and not MIN_PLANNER_VELOCITY (not sure if needed for anything)
        # min_vel, max_vel = self.get_min_max_vel(learned_vel_norm, dt)
        if (learned_vel_norm >= 0.0):
            # forcing the min to be learned_vel_norm can lead to very shaky motions as it constantly wants to move further than the current attractor.
            # so always allow it to go slower. if that's how the motion is defined
            min_vel, max_vel = 0.0, learned_vel_norm
            # max_rot_vel = min(learned_vel_norm / MAX_PLANNER_VELOCITY, 1.) * MAX_PLANNER_ANGULAR_VELOCITY
            max_rot_vel = MAX_PLANNER_ANGULAR_VELOCITY
        else:
            min_vel, max_vel, max_rot_vel = 0.0, MAX_PLANNER_VELOCITY, MAX_PLANNER_ANGULAR_VELOCITY

        current_speed = robot_obs.gripper_velocities_world + robot_obs.base_velocity_world[:2] + 5 * [0.]
        assert len(current_speed) == 14, len(current_speed)
        # NOTE: not setting plan.next_base_tf (always identity_tf)
        prev_plan_next_base_tf = IDENTITY_TF
        next_gripper_tf = self._gmm.integrate_model(time,
                                                    dt,
                                                    prev_plan.next_gripper_tf,
                                                    prev_plan_next_base_tf,
                                                    current_speed,
                                                    min_vel,
                                                    max_vel,
                                                    max_rot_vel,
                                                    self._gmm_time_offset,
                                                    self._robot_config["tip_to_gripper_offset"])

        if self.next_z_subgoal is not None:
            completed_pct = np.clip(calc_euclidean_tf_dist(prev_plan.next_gripper_tf, next_gripper_tf) / (calc_euclidean_tf_dist(prev_plan.next_gripper_tf, self.next_z_subgoal) + SMALL_NUMBER), 0., 1.)
            next_gripper_tf[2] = prev_plan.next_gripper_tf[2] + np.clip(completed_pct * (self.next_z_subgoal[2] - prev_plan.next_gripper_tf[2]),
                                                                        -dt * max_vel,
                                                                        dt * max_vel)

        return NextEEPlan(next_gripper_tf)

    def _interpolate_gmm_plan(self, robot_obs: RobotObs, plan_horizon_meter: float):
        time_step = self.interp_timestep

        plan_meter = []
        prev_plan = self._prev_plan
        t = self.time_planner
        for i in range(int(np.ceil(plan_horizon_meter / (time_step * MAX_PLANNER_VELOCITY)))):
            dt = 0.0 if self._in_start_pause(t) else time_step
            plan = self._cpp_step(time=t, dt=dt, prev_plan=prev_plan, robot_obs=robot_obs, learned_vel_norm=-1)
            plan_meter.append(plan.next_gripper_tf)
            prev_plan = plan
            # NOTE: does not update the gmm_time_offset, so could slightly differ from the actual motion while far away from the first centroid
            t += time_step

            if self._is_done(plan.next_gripper_tf):
                break
        # plt.scatter(np.array(plan_meter)[:, 0], np.array(plan_meter)[:, 1]); plt.show();
        return plan_meter

    def generate_obs_step(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        ee_obs = self.generate_obs_step_ee_obs(robot_obs=robot_obs)
        plan_meter = self._interpolate_gmm_plan(robot_obs=robot_obs, plan_horizon_meter=plan_horizon_meter)

        # height interpolation to make aware of obstacles
        plan_meter = np.array(plan_meter)
        plan_meter_with_start_goal = np.concatenate([np.array(self._prev_plan.next_gripper_tf)[np.newaxis],
                                                     plan_meter,
                                                     np.array(self.gripper_goal_wrist)[np.newaxis]],
                                                    0)
        path_meters_dists = np.linalg.norm(np.diff(plan_meter_with_start_goal[:, :3], axis=0), axis=1)
        # make up for the lost point due to np.diff()
        path_meters_dists = np.concatenate([[0.0], path_meters_dists])
        path_meters_dists_cumsum = path_meters_dists.cumsum()

        if self.interpolate_z:
            path_pixels = self._map.meter_to_pixels(plan_meter_with_start_goal[:, :2])
            obstacle_zs = np.asarray(self._map._floorplan_img_filled)[path_pixels[:, 1], path_pixels[:, 0]]

            xy_dist_goal = np.linalg.norm(np.array(self.gripper_goal_wrist[:2]) - plan_meter_with_start_goal[:, :2], axis=-1)
            ignore_last = sum(xy_dist_goal < 0.10)

            # inflate the ee in xy by just taking the max height
            ofst = int(0.15 / self._map.global_map_resolution)
            obstacle_zs[:-ignore_last] = np.maximum(obstacle_zs[:-ignore_last], np.asarray(self._map._floorplan_img_filled)[path_pixels[:-ignore_last, 1] + ofst, path_pixels[:-ignore_last, 0] + ofst])
            obstacle_zs[:-ignore_last] = np.maximum(obstacle_zs[:-ignore_last], np.asarray(self._map._floorplan_img_filled)[path_pixels[:-ignore_last, 1] - ofst, path_pixels[:-ignore_last, 0] - ofst])

            obstacle_zs += PointToPointPlanner.HEIGHT_INFLATION
            path_meters_z = interpolate_z(path_meters_dists_cumsum, obstacle_zs, self._prev_plan.next_gripper_tf[2], self.gripper_goal_wrist[2], self._map.max_map_height)
            if len(path_meters_z) > 1:
                path_meters_z = path_meters_z[1:]
            plan_meter[:, 2] = path_meters_z[:len(plan_meter)]
            self.next_z_subgoal = plan_meter[min(1, len(plan_meter) - 1)]

        return ee_obs, plan_meter


class PointToPointPlanner(EEPlannerFullPython):
    OCCUPIED_COST = 10_000
    HEIGHT_INFLATION = 0.25
    PLANNER_RESOLUTION = 0.05

    def __init__(self,
                 gripper_goal_tip,
                 head_start: float,
                 map,
                 robot_config: dict,
                 success_thres_dist: float,
                 success_thres_rot: float,
                 use_fwd_orientation: bool,
                 eval: bool):
        super(PointToPointPlanner, self).__init__(gripper_goal_tip,
                                                  head_start,
                                                  map,
                                                  robot_config,
                                                  success_thres_dist=success_thres_dist,
                                                  success_thres_rot=success_thres_rot)
        self.use_fwd_orientation = use_fwd_orientation
        self.eval = eval

    @staticmethod
    def get_inflated_map_weights(map: "Map", ignore_below_height: float, inflation_radius_meter: float = None, floorplan_img=None):
        inflated_map = map.get_inflated_map(ignore_below_height=ignore_below_height, inflation_radius_meter=inflation_radius_meter, floorplan_img=floorplan_img)
        assert inflated_map.min() >= 0.0 and inflated_map.max() <= 1.0, (inflated_map.min(), inflated_map.max())
        inflated_map = resize_to_resolution(current_resolution=map.global_map_resolution, target_resolution=PointToPointPlanner.PLANNER_RESOLUTION, map=inflated_map)
        inflated_map = np.clip(inflated_map, 0.0, 1.0)

        # weights need to be at least 1 for pyastar -> set walls to high value, all other to low values
        weights = inflated_map.astype(np.float32).transpose([1, 0])
        return weights

    @staticmethod
    def map_to_weights(map, ignore_below_height: float, inflate_height: float = HEIGHT_INFLATION, height_preference: bool = True, inflation_radius_meter=None):
        ignore_below_height = max(ignore_below_height - inflate_height, 0.0)

        weights = PointToPointPlanner.get_inflated_map_weights(map, ignore_below_height=ignore_below_height, inflation_radius_meter=inflation_radius_meter)
        weights *= PointToPointPlanner.OCCUPIED_COST

        if (ignore_below_height > 0.0) and height_preference:
            # add a slight preference to not pass over small obstacles unless the path is clearly shorter
            height_weights = PointToPointPlanner.get_inflated_map_weights(map, ignore_below_height=0.0, inflation_radius_meter=inflation_radius_meter)
            height_weights *= 0.25
            weights = np.maximum(weights, height_weights)

        weights += 1
        return weights

    def generate_base_feasible_eeweights(self,
                                         base_tf,
                                         gripper_goal_wrist,
                                         max_base_collisions: float = 0.33,
                                         ignore_last_meters: float = 0.4):
        """
        Generate weights for the EE-plan that allow it to max. deviate as far as the robot arm length from a
        feasible plan for the base
        """
        ignore_last_cells = round(ignore_last_meters / PointToPointPlanner.PLANNER_RESOLUTION)
        base_path_meters, _, base_cost = self.calc_xyplan(self._map, ignore_below_height=0, start=base_tf[:2], goal=gripper_goal_wrist[:2])
        # NOTE: this is added to the empty_map at map.global_map_resolution, so also use this resolution to convert it to pixels
        base_path_pixels = self._map.meter_to_pixels(base_path_meters)

        nr_collisions = base_cost[:-ignore_last_cells].sum() / PointToPointPlanner.OCCUPIED_COST
        if nr_collisions > max_base_collisions:
            print(F'EE-PLANNER HAVING MORE COLLISIONS THAN POTENTIALLY DESIRED: {nr_collisions:.3f} collisions')

        # plot this path onto an empty map, then inflate it with the max. distance the EE is allowed to have from the base (i.e. ~ the arm length of the robot)
        empty_map = np.zeros((self._map._floorplan_img.size[1], self._map._floorplan_img.size[0]))
        empty_map[base_path_pixels[:, 1], base_path_pixels[:, 0]] = 1
        inflated_map = self.get_inflated_map_weights(self._map, ignore_below_height=0.0, floorplan_img=Image.fromarray(empty_map), inflation_radius_meter=self._robot_config['arm_range'])
        # inverse this map and add to the floorplan weights
        inversed_map = 10 * np.maximum(inflated_map.max() - inflated_map, 0)

        ee_weights = (self.map_to_weights(self._map,
                                          ignore_below_height=self._robot_config['z_max'] - self.HEIGHT_INFLATION,
                                          height_preference=self.eval)
                      + inversed_map)
        return ee_weights

    def _cpp_step(self, time: float, dt: float, prev_plan: NextEEPlan, robot_obs: RobotObs, learned_vel_norm: float) -> NextEEPlan:
        min_vel, max_vel = self.get_min_max_vel(learned_vel_norm, dt)
        # don't slow down only because we are close to a subgoal
        if len(self._last_global_plan_meters) > 1:
            min_vel = max_vel
            subgoal_idx = 1
        else:
            subgoal_idx = 0
        next_subgoal = self._last_global_plan_meters[subgoal_idx]

        # xyz
        planned_gripper_vel = get_vec_to_goal_vel(from_tf=prev_plan.next_gripper_tf, to_tf=next_subgoal, min_vel=min_vel, max_vel=max_vel)
        planned_gripper_xyz = np.array(prev_plan.next_gripper_tf[:3]) + np.array(planned_gripper_vel)

        # rotation: interpolation from current to next subgoal rotation
        max_slerp_pct = get_max_angle_slerp_pct(from_tf=prev_plan.next_gripper_tf, to_tf=next_subgoal, dt=dt)

        dist_to_subgoal = max(calc_euclidean_tf_dist(next_subgoal, prev_plan.next_gripper_tf), SMALL_NUMBER)
        completed_pct = np.clip(np.linalg.norm(planned_gripper_vel) / dist_to_subgoal, 0.0, 1.0)
        completed_pct = min(completed_pct, max_slerp_pct)

        # key_rots = Rotation.from_quat([prev_plan.next_gripper_tf[3:], next_subgoal[3:]])
        # slerp = Slerp([0., 1.], key_rots)
        # planned_gripper_q = slerp([completed_pct]).as_quat()[0].tolist()
        planned_gripper_q = slerp_single(prev_plan.next_gripper_tf[3:], next_subgoal[3:], completed_pct)

        return NextEEPlan(planned_gripper_xyz.tolist() + planned_gripper_q)

    def _reset(self, robot_obs: RobotObs, plan_horizon_meter: float) -> EEObs:
        # self._weights = self.map_to_weights(self._map, ignore_below_height=self._robot_config['z_max'])
        self._weights = self.generate_base_feasible_eeweights(base_tf=robot_obs.base_tf, gripper_goal_wrist=self.gripper_goal_wrist)

        self._last_global_plan_meters = None

        path_meters = self._calc_6Dplan(self.gripper_goal_wrist, robot_obs)
        path_pixels = path_meters.copy()
        path_pixels[:, :2] = self._map.meter_to_pixels(path_meters[:, :2])

        return self.generate_obs_step(robot_obs, plan_horizon_meter)

    @staticmethod
    def calc_xyplan(map: "Map", start, goal, ignore_below_height: float, weights: Optional[np.ndarray] = None):
        """Use _calc_plan() internally!"""
        if weights is None:
            weights = PointToPointPlanner.map_to_weights(map, ignore_below_height=ignore_below_height)
        else:
            assert ignore_below_height is None, "ignore_below_height will be ignored if providing weights manually"
        assert weights.dtype == np.float32

        if calc_euclidean_tf_dist(start, goal) < SQROOT_TWO * map.global_map_resolution:
            path_meters = np.array([goal])
            path_pixels = np.array([map.meter_to_pixels(np.array(goal))], dtype=int)
            cost = np.array([0.0])
        else:
            astar_start = map.meter_to_pixels(np.array(start), resolution=PointToPointPlanner.PLANNER_RESOLUTION)
            astar_goal = map.meter_to_pixels(np.array(goal), resolution=PointToPointPlanner.PLANNER_RESOLUTION)
            path_pixels = pyastar.astar_path(weights, astar_start, astar_goal, allow_diagonal=True, costfn='linf')

            # convert to robot locations (meters)
            path_meters = map.pixels_to_meter(path_pixels, resolution=PointToPointPlanner.PLANNER_RESOLUTION)
            # replace discretized last element with the actual goal
            path_meters[0] = start
            path_meters[-1] = goal

            cost = weights[path_pixels[:, 0], path_pixels[:, 1]]

        return path_meters, path_pixels, cost

    def _calc_6Dplan(self, gripper_goal_wrist, robot_obs: RobotObs):
        # NOTE: use last plan, not current position during training (cpp planner already does that, so python should as well for consistency)
        path_meters, _, cost = self.calc_xyplan(self._map, weights=self._weights, ignore_below_height=None,
                                                          start=self._prev_plan.next_gripper_tf[:2], goal=gripper_goal_wrist[:2])

        path_meters_dists = np.linalg.norm(np.diff(path_meters, axis=0), axis=1)
        # make up for the lost point due to np.diff()
        path_meters_dists = np.concatenate([[0.0], path_meters_dists])
        path_meters_dists_cumsum = path_meters_dists.cumsum()

        # find the z to interpolate to: either linear interpolation towards goal or height of the obstacles + inflation in the way
        path_pixels_map_resolution = self._map.meter_to_pixels(path_meters[:, :2], resolution=None)
        obstacle_zs = np.asarray(self._map._floorplan_img_filled)[path_pixels_map_resolution[:, 1], path_pixels_map_resolution[:, 0]]
        obstacle_zs += self.HEIGHT_INFLATION

        # tiny hack to not create motions that go super high if a goal is on a dynamic obstacle
        obstacle_zs[obstacle_zs > self._robot_config['z_max']] = 0.0

        # NOTE: in the real world we do not have access to height information. So should evade all obstacles anyway, unless the goal is above. interpolate_z() will do exactly that: ignore all obstacle_z that are above max_height, except for the goal z.
        max_height = 0.5 if self._map.update_from_global_costmap else self._map.max_map_height
        path_meters_z = interpolate_z(path_meters_dists_cumsum, obstacle_zs, self._prev_plan.next_gripper_tf[2], self.gripper_goal_wrist[2], max_height)

        # rotations (NOTE: not including the height distance in the slerp percentages atm)
        key_rots = Rotation.from_quat([self._prev_plan.next_gripper_tf[3:], self.gripper_goal_wrist[3:]])
        key_times = [0, path_meters_dists_cumsum[-1] + SMALL_NUMBER]
        slerp = Slerp(key_times, key_rots)
        subgoal_rots = slerp(path_meters_dists_cumsum)
        if self.use_fwd_orientation:
            subgoal_orientations = self._generate_fwd_orientations(subgoal_rots, path_meters)
        else:
            subgoal_orientations = subgoal_rots.as_quat()

        path_meters = np.concatenate([path_meters, path_meters_z[:, np.newaxis], subgoal_orientations], axis=-1)
        # replace final goal with exact goal because the others are discretized
        path_meters[-1] = gripper_goal_wrist
        return path_meters

    def generate_obs_step(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        path_meters = self._calc_6Dplan(self.gripper_goal_wrist, robot_obs)
        self._last_global_plan_meters = path_meters

        path_meters = repeat_plan_for_pause(path_meters, time_planner_prepause=self._time_planner_prepause, head_start=self._head_start)
        subgoal_idx = min(int(plan_horizon_meter / PointToPointPlanner.PLANNER_RESOLUTION), len(path_meters))
        return self.generate_obs_step_ee_obs(robot_obs=robot_obs), path_meters[:subgoal_idx]

    def update_weights(self, robot_obs: RobotObs):
        if self._map._world_type != 'sim':
            # Don't interpret as height map. Values are decreasing away from obstacles -> makes sure ee wants to pass through the middle of e.g. doors
            inflated_map = np.asarray(self._map._floorplan_img) / self._map.max_map_height
            assert inflated_map.min() >= 0.0 and inflated_map.max() <= 1.0, (inflated_map.min(), inflated_map.max())
            inflated_map = resize_to_resolution(current_resolution=self._map.global_map_resolution, target_resolution=PointToPointPlanner.PLANNER_RESOLUTION, map=inflated_map)
            inflated_map = np.clip(inflated_map, 0.0, 1.0)

            # weights need to be at least 1 for pyastar -> set walls to high value, all other to low values
            weights = inflated_map.astype(np.float32).transpose([1, 0])
            weights *= PointToPointPlanner.OCCUPIED_COST
            weights += 1
            self._weights = weights
        else:
            # NOTE: to make this less expensive for dyn obstacles: do not inflate at all?
            self._weights = self.map_to_weights(self._map,
                                                ignore_below_height=self._robot_config['z_max'] - self.HEIGHT_INFLATION,
                                                height_preference=False,
                                                inflation_radius_meter=None)
        # else:
        #     self._weights = self.generate_base_feasible_eeweights(base_tf=robot_obs.base_tf, gripper_goal_wrist=self.gripper_goal_wrist)

    def _generate_fwd_orientations(self, subgoal_rots, path_meters):
        """always face the yaw in the direction of the next subgoal, but slerp the roll and pitch angles as usual"""
        if len(path_meters) <= 12:
            # when we are close, just interpolate to the target orientation
            return subgoal_rots.as_quat()
        to_next_subgoal_vec = np.diff(path_meters, axis=0)
        yaws = np.arctan2(to_next_subgoal_vec[:, 1], to_next_subgoal_vec[:, 0])
        yaws = np.concatenate([yaws[:1], yaws], axis=0)

        # don't allow the next yaw change to be too big
        # NOTE: restricting the max angular velocity in c++ -> values here / projected onto map won't be the exactly correct rotations
        max_yaw_diff = np.pi / 16
        current_yaw = quaternion_to_yaw(self._prev_plan.next_gripper_tf[3:])
        for i in range(len(yaws)):
            yaw_prev = yaws[i - 1] if i else current_yaw
            yaw_diff = clamp_angle_minus_pi_pi(yaws[i] - yaw_prev)
            yaws[i] = yaw_prev + np.clip(yaw_diff, -max_yaw_diff, max_yaw_diff)

        subgoal_eulers = subgoal_rots.as_euler('xyz')
        subgoal_eulers[:, 2] = yaws
        subgoal_orientations = Rotation.from_euler('xyz', subgoal_eulers).as_quat()
        return subgoal_orientations

    def plot_global_plan(self, map: "Map" = None, path_meters: list = None):
        """NOTE: this function plots the plan in the resolution of self._map, independent of what resolution it was calculated in"""
        if path_meters is None:
            assert self._last_global_plan_meters is not None, "Plan never calculated yet"
            path_meters = self._last_global_plan_meters
        if map is None:
            map = self._map
        path_pixels = self._map.meter_to_pixels(np.array(path_meters)[:, :2])
        f, ax = map.plot_floorplan()
        ax.scatter(path_pixels[:, 0], path_pixels[:, 1])
        for c, point in zip(['blue', 'red'], [path_pixels[0], path_pixels[-1]]):
            ax.scatter(point[0], point[1], color=c, marker='X')
        return f, ax

    def plot_plan_onto_weights(self, weights: np.ndarray = None, path_meters: list = None):
        if weights is None:
            weights = self._weights
        if path_meters is None:
            assert self._last_global_plan_meters is not None, "Plan never calculated yet"
            path_meters = self._last_global_plan_meters
        path_pixels = self._map.meter_to_pixels(np.array(path_meters)[:, :2], resolution=PointToPointPlanner.PLANNER_RESOLUTION)

        f, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(weights.transpose([1, 0]))
        ax.scatter(path_pixels[:, 0], path_pixels[:, 1])
        for c, point in zip(['blue', 'red'], [path_pixels[0], path_pixels[-1]]):
            ax.scatter(point[0], point[1], color=c, marker='X')
        return f, ax

    # def plot_base_weights(self, max_value: float = 20):
    #     assert self._last_global_plan_meters is not None, "Plan never calculated yet"
    #     path_meters = self._last_global_plan_meters
    #     path_pixels = self._map.meter_to_pixels(np.array(path_meters)[:, :2], resolution=self.PLANNER_RESOLUTION)
    #
    #     f, ax = plt.subplots(1, 2, figsize=(12, 6))
    #     ax[0].imshow(np.asarray(self._map._floorplan_img).squeeze())
    #     ax[1].imshow(np.clip(self._weights.transpose([1, 0]), 0, max_value))
    #     ax[1].scatter(path_pixels[:, 0], path_pixels[:, 1])
    #     for c, point in zip(['blue', 'red'], [path_pixels[0], path_pixels[-1]]):
    #         ax[1].scatter(point[0], point[1], color=c, marker='X')
    #
    #     ax[0].axis('off')
    #     ax[1].axis('off')
    #     plt.tight_layout()
    #
    #     from pathlib import Path
    #     p = Path(__file__).parent.parent.parent / "paper_n2" / "figures"
    #     plt.savefig(p / "astar_weights")


class SplinePlanner(EEPlannerFullPython):
    def __init__(self,
                 gripper_goal_tip,
                 head_start: float,
                 map,
                 robot_config: dict,
                 success_thres_dist: float,
                 success_thres_rot: float,
                 waypoints_wrist: list):
        super(SplinePlanner, self).__init__(gripper_goal_tip,
                                            head_start,
                                            map,
                                            robot_config,
                                            success_thres_dist=success_thres_dist,
                                            success_thres_rot=success_thres_rot)
        self.waypoints_wrist = waypoints_wrist

    def set_goal(self, gripper_goal_tip):
        # super().set_goal(gripper_goal_tip=gripper_goal_tip)
        # if hasattr(self, "path_meters"):
        #     all_waypoints = np.stack([self.path_meters[0]] + self.waypoints_wrist + [self.gripper_goal_wrist], 0)
        #     self._calc_path_meters(all_waypoints)
        if hasattr(self, "gripper_goal_tip"):
            # if it's already initialised ignore any further user inputs / modifications
            return
        else:
            super().set_goal(gripper_goal_tip=gripper_goal_tip)


    def _calc_path_meters(self, all_waypoints):
        # interpolate xyz
        diff_dists = np.linalg.norm(np.diff(all_waypoints[:, :3], axis=0), axis=1)
        # make up for the lost point due to np.diff()
        diff_dists = np.concatenate([[0.0], diff_dists])
        t = diff_dists.cumsum()
        t_to_interpolate_to = np.arange(0, t[-1], self._map.global_map_resolution)
        t_to_interpolate_to = np.append(t_to_interpolate_to, t[-1])
        xyz_plan = []
        for dim in range(3):
            f = interp1d(t, all_waypoints[:, dim], kind='cubic')
            xyz_plan.append(f(t_to_interpolate_to))
        # ensure interpolation adheres to height limits
        # NOTE: interpolated xy might still go outside of the map
        xyz_plan[2] = np.clip(xyz_plan[2], self._robot_config['z_min'], self._robot_config['z_max'])
        xyz_plan = np.stack(xyz_plan, axis=-1)

        # slerp rotations proportional to distance
        xyz_plan_dists = np.linalg.norm(np.diff(xyz_plan, axis=0), axis=1)
        # make up for the lost point due to np.diff()
        xyz_plan_dists = np.concatenate([[0.0], xyz_plan_dists])
        xyz_plan_dists_cumsum = xyz_plan_dists.cumsum()

        key_rots = Rotation.from_quat([all_waypoints[0][3:], all_waypoints[-1][3:]])
        key_times = [0, xyz_plan_dists_cumsum[-1] + SMALL_NUMBER]
        slerp = Slerp(key_times, key_rots)
        subgoal_rots = slerp(xyz_plan_dists_cumsum)
        subgoal_orientations = subgoal_rots.as_quat()

        # put together
        self.path_meters = np.concatenate([xyz_plan, subgoal_orientations], axis=-1)
        self.current_subgoal = self.path_meters[0]
        self.path_meters = self.path_meters[1:]

    def _reset(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        all_waypoints = np.stack([robot_obs.gripper_tf] + self.waypoints_wrist + [self.gripper_goal_wrist], 0)
        self._calc_path_meters(all_waypoints)

        return self.generate_obs_step(robot_obs, plan_horizon_meter)

    def _cpp_step(self, time: float, dt: float, prev_plan: NextEEPlan, robot_obs: RobotObs, learned_vel_norm: float) -> NextEEPlan:
        min_vel, max_vel = self.get_min_max_vel(learned_vel_norm, dt)

        # don't slow down only because we are close to a subgoal
        if len(self.path_meters) > 1:
            min_vel = max_vel

        # xyz velocity
        planned_gripper_vel = get_vec_to_goal_vel(from_tf=prev_plan.next_gripper_tf, to_tf=self.current_subgoal, min_vel=min_vel, max_vel=max_vel)
        next_gripper_position = (np.array(prev_plan.next_gripper_tf[:3]) + planned_gripper_vel).tolist()
        next_gripper_position[2] = np.clip(next_gripper_position[2], self._robot_config['z_min'], self._robot_config['z_max'])

        # angular velocity
        slerp_pct = get_max_angle_slerp_pct(from_tf=prev_plan.next_gripper_tf, to_tf=self.current_subgoal, dt=dt)

        # key_rots = Rotation.from_quat([prev_plan.next_gripper_tf[3:], self.current_subgoal[3:]])
        # slerp = Slerp([0, 1.0], key_rots)
        # planned_q = slerp([slerp_pct]).as_quat()[0].tolist()
        planned_q = slerp_single(prev_plan.next_gripper_tf[3:], self.current_subgoal[3:], slerp_pct)

        return NextEEPlan(next_gripper_position + planned_q)

    def _subgoal_achieved(self, current_gripper_tf):
        # TODO: should this also consider if we've achieved the rotation of the previous subgoal?
        return calc_euclidean_tf_dist(self.current_subgoal, current_gripper_tf) < 0.01

    def generate_obs_step(self, robot_obs: RobotObs, plan_horizon_meter: float) -> Tuple[EEObs, EEPlan]:
        if self._subgoal_achieved(self._prev_plan.next_gripper_tf):
            self.current_subgoal = self.path_meters[0]
            if len(self.path_meters) > 1:
                self.path_meters = self.path_meters[1:]
        ee_obs = self.generate_obs_step_ee_obs(robot_obs=robot_obs)
        path_meters = repeat_plan_for_pause(self.path_meters, time_planner_prepause=self._time_planner_prepause, head_start=self._head_start)
        subgoal_idx = min(int(plan_horizon_meter / self._map.global_map_resolution), len(path_meters))
        return ee_obs, path_meters[:subgoal_idx]

    def update_weights(self, robot_obs: RobotObs):
        """Function to update weights or similar in case the map changed (e.g. dynamic obstacles)"""
        return
