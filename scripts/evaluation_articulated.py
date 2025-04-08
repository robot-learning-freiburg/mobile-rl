from pathlib import Path
import time
import numpy as np
import rospy
import wandb
from matplotlib import pyplot as plt
from typing import List, Tuple
import copy
import subprocess

from ocs2_msgs.msg import mpc_target_trajectories, mpc_state, mpc_input, mpc_observation, mpc_flattened_controller

plt.style.use('seaborn')
from modulation.utils import setup_config_wandb, env_creator, launch_ros, create_env
from modulation.envs.tasks import GripperActions, publish_move_obstacles_enabled
from modulation.planning.rrt import PlanningEpisode
from modulation.envs.env_utils import quaternion_to_yaw, calc_euclidean_tf_dist, calc_rot_dist
from modulation.articulated.articulated_utils import BaseCollisionMarker, get_ocs2_msg, set_joints_to_articulated_values, \
    ArticulatedEpisode, publish_sdf, start_mpc_mrt_node, get_grippertf_world, \
    load_velocity_controllers, switch_to_position_controllers
from modulation.envs.eeplanner import NextEEPlan, TIME_STEP_TRAIN
from modulation.evaluation import get_task_prefix


DT = 0.1
PLANNER_VELOCITY = 0.1


def get_ee_plan(env, continuing_episode: bool, max_plan_len=None, start_gripper_tf: bool = None):
    ee_planner = env.unwrapped._ee_planner
    ee_planner._is_analytic_env = True
    ee_planner._rostime_at_start = 0.0
    ee_planner._time_planner_prepause = 0.0

    initial_robot_obs = env.get_robot_obs()
    if continuing_episode:
        if start_gripper_tf is None:
            start_gripper_tf = get_grippertf_world(env)
        initial_robot_obs.gripper_tf = start_gripper_tf
        ee_planner._prev_plan = NextEEPlan(start_gripper_tf)
    ee_plan = [initial_robot_obs.gripper_tf]
    eeobs = None

    while True:
        if eeobs:
            initial_robot_obs.gripper_tf = eeobs.next_gripper_tf
        _, _, _ = ee_planner.step(initial_robot_obs, learned_vel_norm=PLANNER_VELOCITY)
        eeobs, _ = ee_planner.generate_obs_step(initial_robot_obs, plan_horizon_meter=0.5)
        ee_plan.append(eeobs.next_gripper_tf)
        if ee_planner._is_done(eeobs.next_gripper_tf):
            break
        elif (max_plan_len is not None) and (len(ee_plan) > max_plan_len):
            return ee_plan
        if len(ee_plan) > 10_000:
            env.publish_marker(ee_planner.gripper_goal_wrist, 0, "gripper_goal", "green", 1.0, frame_id="world")
            if hasattr(ee_planner, "_gmm"):
                mus = ee_planner._gmm.get_mus()
                env.publish_markers(mus, 0, "mus", "blue", 1.0, frame_id="world")
            env.publish_markers(ee_plan[::10], 0, "gripper_goal", "cyan", 1.0, frame_id="world")
            assert False, len(ee_plan)
    return ee_plan


def publish_markers(env, episode, current_desired_ee, marker_id: int):
    env.publish_marker(episode.base_tfs[-1], marker_id, "base_tf", "red" if episode.base_collisions[-1] else "orange", 0.5, frame_id="world")
    success = (episode.dists_to_motion[-1] < 0.1) and (episode.rot_dists_to_motion[-1] < 0.05)
    env.publish_marker(episode.gripper_tfs[-1], marker_id, "gripper_tf", "green" if success else "red", 0.5, frame_id="world")
    env.publish_marker(current_desired_ee, marker_id, "gripper_goal", "cyan", 0.5, frame_id="world")
    BaseCollisionMarker.publish_basecollision(env, episode.base_tfs[-1])


def send_eeplan_to_mrt(ee_plan, dt: float, env, episode=None, is_dynamic_task=False):
    if (episode is not None) and (not is_dynamic_task):
        episode.add_ee_plan(ee_plan)

    pub = rospy.Publisher('/mobile_manipulator_mpc_target', mpc_target_trajectories, queue_size=25)

    # construct msg with complete motion
    def _build_mpc_msg(ee_plan, dt):
        msg = mpc_target_trajectories()
        t = get_ocs2_msg().time + dt
        for i, x in enumerate(ee_plan):
            # if (i % 5 == 0) or (i == len(ee_plan) - 1):
            goal = mpc_state()
            goal.value = x
            msg.stateTrajectory.append(goal)
            inpt = mpc_input()
            inpt.value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            msg.inputTrajectory.append(inpt)
            msg.timeTrajectory.append(t + (i + 1) * dt)
        return msg

    # publish message
    ocs2_msg = get_ocs2_msg()
    last_t = ocs2_msg.time
    rate = rospy.Rate(1. / TIME_STEP_TRAIN)
    marker_id = 0
    max_dynamic_steps = 10_000

    msg = _build_mpc_msg(ee_plan, dt)
    tt, st = copy.deepcopy(msg.timeTrajectory), copy.deepcopy(msg.stateTrajectory)
    current_desired_ee = st[0].value

    if is_dynamic_task:
        map_changed = env.map.update(dt=0.0)
        if map_changed:
            env.unwrapped._ee_planner.update_weights(None)
        publish_sdf(env.map)
        publish_move_obstacles_enabled(True)

    while len(msg.timeTrajectory) and (ocs2_msg.time <= msg.timeTrajectory[-1]):
        rospy.loginfo_throttle(5, f"current time: {ocs2_msg.time:.2f}, final time: {msg.timeTrajectory[-1]:.2f}")
        pub.publish(msg)
        ocs2_msg = get_ocs2_msg()

        if is_dynamic_task and (ocs2_msg.time - last_t > TIME_STEP_TRAIN):
            print(f"dt: {ocs2_msg.time - last_t:.4f}")
            map_changed = env.map.update(dt=ocs2_msg.time - last_t)
            if map_changed:
                env.unwrapped._ee_planner.update_weights(None)
            # don't do the head-start thing, as it screws up the get_ee_plan() logic
            env.unwrapped._ee_planner._head_start = 0.0
            ee_plan = get_ee_plan(env, continuing_episode=True, max_plan_len=25, start_gripper_tf=ee_plan[1])
            msg = _build_mpc_msg(ee_plan, dt)
            current_desired_ee = ee_plan[0]
            publish_sdf(env.map)
            episode.add_ee_plan([current_desired_ee])
            assert len(episode.ee_plan) < max_dynamic_steps, f"Ran {max_dynamic_steps} steps without reaching the goal"
            # env.publish_markers(ee_plan, 9999, 'eee', 'pink', 0.5, frame_id="world")
        else:
            while len(tt) and (tt[0] < ocs2_msg.time):
                tt.pop(0)
                current_desired_ee = st.pop(0).value

        if episode is not None:
            violating_joint_limit = set_joints_to_articulated_values(env, ocs2_msg=ocs2_msg)
            robot_obs = env.get_robot_obs()
            local_map = env.get_local_map(robot_obs.base_tf, ee_plan, use_ground_truth=True)
            base_collision = env.check_collision(local_map)
            env.unwrapped._publish_local_map(local_map, robot_obs)
            # NOTE: robot_obs.gripper_tf somehow has a height offset to the one shown by mpc in rviz
            gripper_tf = get_grippertf_world(env)
            episode.add_step(gripper_tf=gripper_tf, base_tf=robot_obs.base_tf, base_collision=base_collision, violating_joint_limit=violating_joint_limit)

            if is_dynamic_task and len(ee_plan) < 50:
                ee_dist = calc_euclidean_tf_dist(episode.gripper_tfs[-1], ee_plan[-1])
                ee_rot_dist = calc_rot_dist(episode.gripper_tfs[-1], ee_plan[-1])
                # TODO: can this get stuck in an infinite loop for the dynamic envs?
                if (ee_dist < env.unwrapped._ee_planner._success_thres_dist) and (ee_rot_dist < env.unwrapped._ee_planner._success_thres_rot):
                    return

            # early termination of the episode if getting stuck
            if (calc_euclidean_tf_dist(episode.gripper_tfs[-1], current_desired_ee) > 1.0) or (episode.nr_joint_limit_violations > 50) or (episode.nr_base_collisions > 50):
                return

            if base_collision or (ocs2_msg.time - last_t > 0.25):
                publish_markers(env, episode, current_desired_ee, marker_id)
                marker_id += 1
                last_t = ocs2_msg.time

        rate.sleep()

    print("goal reached")


def rollout_goal(env, episode, dt: float = DT, mpc_mrt_node=None):
    ee_plan = get_ee_plan(env, continuing_episode=len(episode) > 0)

    # make sure we start at the beginning of the plan
    if len(episode) == 0:
        # start mrt_ros_interface with our initial state
        i = 0
        while True:
            try:
                if mpc_mrt_node is not None:
                    mpc_mrt_node.shutdown()
                    time.sleep(2.0)
                # mrt_node = start_mrt_node(env.get_robot_obs(), ee_plan, env)
                mpc_mrt_node = start_mpc_mrt_node(env.get_robot_obs(), ee_plan, env)
                time.sleep(1)
                get_ocs2_msg(timeout=5)
                time.sleep(1)
                get_ocs2_msg(timeout=5)
                break
            except:
                i += 1
            if i > 10:
                assert False, "Could not get ocs2 msg from node"

    # somehow doesn't necessarily set the same initial joint values and can have somewhat significant deviations in the initial ee-pose
    # so publish the initial pose as a goal again
    # TODO: this can mean that it can start from siginificantly different joint-values than ours
    gripper_tf = get_grippertf_world(env)
    if (calc_euclidean_tf_dist(ee_plan[0], gripper_tf) > 0.03) or calc_rot_dist(ee_plan[0], gripper_tf) > 0.04:
        send_eeplan_to_mrt(ee_plan[:1], episode=None, dt=dt, env=env)

    # start actual rollout
    send_eeplan_to_mrt(ee_plan, episode=episode, dt=dt, env=env, is_dynamic_task='dyn' in env.loggingname)
    return mpc_mrt_node


def main(raw_args=None):
    # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py
    # https://github.com/ray-project/ray/blob/master/rllib/rollout.py

    main_path = Path(__file__).parent.absolute()
    run, wandb_config = setup_config_wandb(main_path, sync_tensorboard=False, allow_init=True, no_ckpt_endig=True, framework='ray', raw_args=raw_args)
    assert wandb_config.algo == "articulated", wandb_config.algo

    assert wandb_config.local_map_resolution == 0.025, wandb_config.local_map_resolution
    assert wandb_config.global_map_resolution == 0.025, wandb_config.global_map_resolution
    assert not wandb_config.learn_torso, wandb_config.learn_torso

    mpc_mrt_node = None
    vel_controller_spawner = None

    for world_type in wandb_config.eval_execs:
        wandb_config.update({'world_type': world_type}, allow_val_change=True)
        for global_step, task in enumerate(wandb_config.eval_tasks):
            fwd_orientations = [False, True] if (task in ['picknplace', 'bookstorepnp']) else [False]
            for o in fwd_orientations:
                wandb_config.update({'use_fwd_orientation': o}, allow_val_change=True)
                _ = subprocess.run(["rosnode", "kill", "/mobile_manipulator_mpc_node"])
                launch_ros(main_path=main_path, config=wandb_config, task=task, no_gui=True, always_relaunch=world_type == "sim")
                rospy.init_node('kinematic_feasibility_py', anonymous=False)

                rospy.set_param("/ocs2_world_type", world_type)
                rospy.set_param("/ocs2_robotname", wandb_config.env)
                if world_type == "gazebo":
                    vel_controller_spawner = load_velocity_controllers(wandb_config.env)
                    assert len(wandb_config.eval_execs) == 1, "Don't think this will run well for multiple tasks in parallel"
                    assert len(wandb_config.eval_tasks) == 1, "Don't think this will run well for multiple tasks in parallel"

                rospy.set_param("/articulated_collision_mu", wandb_config.articulated_collision_mu)
                rospy.set_param("/articulated_collision_delta", wandb_config.articulated_collision_delta)
                rospy.set_param("/articulated_jv_mu", wandb_config.articulated_jv_mu)
                rospy.set_param("/articulated_jv_delta", wandb_config.articulated_jv_delta)
                rospy.set_param("/articulated_time_horizon", wandb_config.articulated_time_horizon)
                rospy.set_param("/articulated_min_step", wandb_config.articulated_min_step)

                env = create_env(wandb_config, task=task, node_handle="train_env", eval=True)
                # start_mpc_node(robot_name=env.env_name)

                rospy.loginfo(f"Evaluating on task {env.loggingname}.")
                name_prefix = get_task_prefix(env, prefix='')
                episodes = []

                for i in range(wandb_config.nr_evaluations):
                    if wandb_config.eval_seed is not None:
                        env.seed(wandb_config.eval_seed + i)

                    # ensure ee plans the whole motion in advance
                    # env.unwrapped.plan_horizon_meter = 15

                    if (world_type == "gazebo") and (env.env_name == "pr2"):
                        switch_to_position_controllers()
                    obs = env.reset()
                    publish_move_obstacles_enabled(False)

                    env.map.publish_floorplan_rviz()
                    publish_sdf(env.map)

                    episode = ArticulatedEpisode()

                    mpc_mrt_node = rollout_goal(env, episode=episode, mpc_mrt_node=mpc_mrt_node)

                    if hasattr(env, 'goals'):
                        while episode.success(0.1, 0.05) and (env.current_goal < len(env.goals) - 1):
                            set_joints_to_articulated_values(env)

                            end_action = env.goals[env.current_goal].end_action
                            if end_action == GripperActions.GRASP:
                                env.grasp()
                            elif end_action == GripperActions.OPEN:
                                env.env.open_gripper(wait_for_result=False)

                            new = env.goals[env.current_goal + 1]
                            ee_planner = new.ee_fn(gripper_goal_tip=new.gripper_goal_tip,
                                                   head_start=new.head_start,
                                                   map=env.env.map,
                                                   robot_config=env.env.robot_config,
                                                   success_thres_dist=new.success_thres_dist,
                                                   success_thres_rot=new.success_thres_rot)
                            obs = env.env.set_ee_planner(ee_planner=ee_planner)
                            env.env.current_goal += 1

                            rollout_goal(env, episode=episode)

                    # env._episode_cleanup()
                    episodes.append(episode)

                    if mpc_mrt_node is not None:
                        mpc_mrt_node.shutdown()
                        mpc_mrt_node = None
                        time.sleep(1)

                    # TODO: goal_reached might be incoccrect if stopped due to base_collisions before having done all subgoals
                    rospy.logwarn(f"""\n\n\n\n{name_prefix} ep {i}: 
                                      successD0.1RD0.05: {np.sum([e.success(0.1, 0.05) for e in episodes])}/{i + 1}, 
                                      successD0.05RD0.05: {np.sum([e.success(0.05, 0.05) for e in episodes])}/{i + 1}, 
                                      goal_reached: {np.sum([e.goal_reached for e in episodes])}/{i+1}, 
                                      no base collisions: {np.sum([e.nr_base_collisions == 0 for e in episodes])}/{i+1},
                                      no joint limit violations: {np.sum([e.nr_joint_limit_violations == 0 for e in episodes])}/{i+1},
                                      episode success: {episode.success(0.1, 0.05)},
                                      base_collisions: {episode.nr_base_collisions},
                                      joint_limit_violations: {episode.nr_joint_limit_violations}, 
                                      max_dist_to_motion: {np.max(episode.dists_to_motion):.2f}, 
                                      max_rot_dist_to_motion: {np.max(episode.rot_dists_to_motion):.2f}\n\n\n\n""")
                    # plt.close(); plt.plot(episode.dists_to_motion), plt.plot(episode.rot_dists_to_motion); plt.show();
                    # for _ in range(2):
                    #     PlanningEpisode.delete_markers(["/train_env/gripper_goal_visualizer"], are_marker_array=[False], frame_id="world")
                    #     time.sleep(0.05)

                metrics = {"successD0.1RD0.05": np.mean([e.success(0.1, 0.05) for e in episodes]),
                           "successD0.05RD0.05": np.mean([e.success(0.05, 0.05) for e in episodes]),
                           "base_collisions0": np.mean([e.nr_base_collisions == 0 for e in episodes]),
                           "base_collisions": np.mean([e.nr_base_collisions for e in episodes]),
                           "joint_limit_violations0": np.mean([e.nr_joint_limit_violations == 0 for e in episodes]),
                           "success_motion_avgdist": np.nanmean([np.mean(e.dists_to_motion) if e.success(0.1, 0.05) else np.nan for e in episodes]),
                           "success_motion_maxdist": np.nanmean([np.max(e.dists_to_motion) if e.success(0.1, 0.05) else np.nan for e in episodes]),
                           "success_motion_avgrotdist": np.nanmean([np.mean(e.rot_dists_to_motion) if e.success(0.1, 0.05) else np.nan for e in episodes]),
                           "success_motion_maxrotdist": np.nanmean([np.mean(e.rot_dists_to_motion) if e.success(0.1, 0.05) else np.nan for e in episodes]),
                           'global_step':           global_step,
                           'timesteps_total':       global_step}
                logmetrics = {(f'{name_prefix}/{k}' if (k not in ('global_step', 'timesteps_total')) else k): v for k, v in metrics.items()}
                wandb.log(logmetrics)
    print("Done")

    if vel_controller_spawner is not None:
        # stop the velocity controllers, which also restarts the position controllers to do the env reset etc
        vel_controller_spawner.shutdown()
        vel_controller_spawner = None



if __name__ == '__main__':
    main()
