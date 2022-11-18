from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Optional

import json
from pathlib import Path
import numpy as np
import rospy
import time
import torch
import wandb
from matplotlib import pyplot as plt

from modulation.envs.env_utils import calc_disc_return, quaternion_to_yaw, SMALL_NUMBER, calc_euclidean_tf_dist, calc_rot_dist
from modulation.utils import episode_is_success, env_creator


@dataclass
class Episode:
    snapshot: Optional[Tuple]
    initial_obs: np.ndarray
    actions: list = field(default_factory=list)
    unscaled_actions: list = field(default_factory=list)
    # obs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    infos: list = field(default_factory=list)
    _gripper_dists_to_motion = None

    def add_step(self, action, obs, reward, done, info, unscaled_action):
        has_nan = np.any(np.isnan(obs)) if not isinstance(obs, tuple) else sum([np.isnan(o).sum() for o in obs])
        assert not has_nan, "Nan found in obs"
        assert reward <= 0.001, reward

        self.actions.append(action)
        self.unscaled_actions.append(unscaled_action)
        self.rewards.append(reward)
        self.dones.append(done)
        # self.obs.append(obs)
        self.infos.append(info)

        if done:
            self._gripper_dists_to_motion = self.gripper_dists_to_motion

    def __len__(self):
        return len(self.rewards)

    @property
    def goal_reached(self):
        return self.infos[-1].get('ee_done', False)

    @property
    def goal_reached_no_collision(self):
        return self.goal_reached and self.nr_base_collisions == 0

    @property
    def nr_base_collisions(self):
        return self.infos[-1].get('nr_base_collisions', 0)

    @property
    def nr_kin_failures(self):
        return self.infos[-1].get('nr_kin_failures', 0)

    @property
    def nr_selfcollisions(self):
        return np.sum([info.get('selfcollision', 0) for info in self.infos])

    @property
    def total_nr_jumps(self):
        return np.sum([info.get('jumps_vel_limit', 0) for info in self.infos])

    @property
    def steps_above_joint_vel_limit(self):
        return np.sum([info.get('above_joint_vel_limit', 0) for info in self.infos])

    @property
    def steps_above_3x_joint_vel_limit(self):
        return np.sum([info.get('above_3x_joint_vel_limit', 0) for info in self.infos])

    @property
    def vel_normed_avg_deviation(self):
        return np.sum([info.get('vel_normed_avg_deviation', 0) for info in self.infos])

    @property
    def gripper_dists_to_achieved(self) -> np.ndarray:
        return np.array([info.get('gripper_dist_to_achieved', 0) for info in self.infos])

    @property
    def gripper_rot_dists_to_achieved(self) -> np.ndarray:
        return np.array([info.get('gripper_rot_dist_to_achieved', 0) for info in self.infos])

    @staticmethod
    def calc_gripper_dists_to_motion(gripper_tfs_achieved: np.ndarray, gripper_tfs_desired: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dists, rot_dists = [], []
        for tf in gripper_tfs_achieved:
            dist_to_motion = calc_euclidean_tf_dist(tf[np.newaxis], gripper_tfs_desired).min()
            rot_dist_to_motion = calc_rot_dist(tf[np.newaxis], gripper_tfs_desired).min()
            dists.append(dist_to_motion)
            rot_dists.append(rot_dist_to_motion)
        return np.array(dists), np.array(rot_dists)

    @property
    def gripper_dists_to_motion(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._gripper_dists_to_motion is not None:
            return self._gripper_dists_to_motion
        gripper_tfs_achieved = np.array([info.get('gripper_tf_achieved', 0.0) for info in self.infos])
        gripper_tfs_desired = np.array([info.get('gripper_tf_desired', 0.0) for info in self.infos])
        return self.calc_gripper_dists_to_motion(gripper_tfs_achieved=gripper_tfs_achieved,
                                                 gripper_tfs_desired=gripper_tfs_desired)

    def gripper_dists_to_motion_below_thresh(self, max_dist: float, max_rot_dist: float) -> bool:
        dists, rot_dists = self.gripper_dists_to_motion
        return (dists.max() < max_dist) and (rot_dists.max() < max_rot_dist)

    @property
    def success(self):
        return episode_is_success(nr_kin_fails=self.nr_kin_failures, nr_collisions=self.nr_base_collisions, goal_reached=self.goal_reached)

    @property
    def success_nojumps(self):
        return self.success and self.steps_above_joint_vel_limit == 0

    @property
    def success_nojumps3x(self):
        return self.success and self.steps_above_3x_joint_vel_limit == 0

    def deviation_success(self, max_dist: float, max_rot_dist: float) -> bool:
        return (self.nr_base_collisions == 0) and self.goal_reached and self.gripper_dists_to_motion_below_thresh(max_dist=max_dist, max_rot_dist=max_rot_dist)

    def get_disc_return(self, gamma: float):
        return calc_disc_return(self.rewards, gamma=gamma)

    @property
    def total_reward(self):
        return np.sum(self.rewards)

    @property
    def max_close_steps_reached(self):
        return self.infos[-1].get('max_close_steps', False)

    def reset_env_to_start(self, env):
        assert self.snapshot is not None
        env.load_snapshot(self.snapshot)
        return self.initial_obs


def get_next_global_step(debug: bool, increment: int, trainer=None) -> int:
    if debug:
        # won't log the values as step 0 was already logged
        return 0
    else:
        global_step = max(wandb.run.step, wandb.run.summary.get('timesteps_total', 0))

        api = wandb.Api()
        r = api.run(wandb.run.path)
        hist = r.history(samples=10, keys=['timesteps_total'], pandas=True)
        if len(hist):
            global_step = max(global_step, max(hist['timesteps_total'].values))
            global_step = max(global_step, r.summary['timesteps_total'])

        if trainer is not None:
            global_step = max(global_step, trainer.get_state()['timesteps_total'])
        assert global_step is not None, global_step
        return global_step + increment


def get_metric_file_path():
    return Path(wandb.run.dir) / "metric"


def download_wandb(file, max_tries: int = 10, root: str = '/tmp', replace: bool = True):
    for i in range(max_tries):
        try:
            return file.download(root=root, replace=replace)
        except Exception as e:
            print(f"{i} Failed to download file, trying again until {max_tries}")
            fail = str(e)
            time.sleep(5)
    else:
        raise RuntimeError(fail)

def compute_ray_action_fast(policy, obs, explore: bool):
    """NOTE: faster than policy.compute_action(), but will be wrong if using any obs or action normalisation"""
    # torch.as_tensor(obs[0], dtype=torch.float32)
    if isinstance(obs, tuple):
        obs = (torch.unsqueeze(torch.as_tensor(obs[0], dtype=torch.float32, device=policy.device), 0),
               torch.unsqueeze(torch.as_tensor(obs[1], device=policy.device), 0))
    else:
        obs = torch.unsqueeze(torch.as_tensor(obs, dtype=torch.float32, device=policy.device), 0)
    dist_class = policy.dist_class
    dist_inputs = policy.model.get_policy_output(obs)
    assert not explore, explore
    action_dist = dist_class(dist_inputs, policy.model)
    action = action_dist.deterministic_sample()
    return action.cpu().numpy()[0]


def rollout(env, policy, deterministic: bool, max_len: int, replay_episode: Episode = None):
    policy.model.eval()

    with torch.no_grad():
        if replay_episode is not None:
            obs = replay_episode.reset_env_to_start(env)
        else:
            obs = env.reset()
        done = False
        done_players = defaultdict(bool)
        episode = Episode(None, obs)

        while not done:
            action = compute_ray_action_fast(policy, obs, explore=not deterministic)
            # obs_flat = flatten_obs(obs, use_map_obs=env.unwrapped._use_map_obs or getattr(env, '_obstacle_spacing', None))
            # action, *_ = policy.compute_single_action(obs_flat, explore=not deterministic, clip_actions=policy.config['clip_actions'])

            obs, reward, done, info = env.step(action)

            if info.get('max_close_steps', None):
                print('max_close_steps reached')

            if len(episode) and len(episode) % 10_000 == 0:
                rospy.logwarn(f"{len(episode)} steps already!. Continuing until a max. of {max_len}")
            if len(episode) > max_len:
                dist_to_goal = calc_euclidean_tf_dist(env.get_robot_obs().gripper_tf, env.unwrapped._ee_planner.gripper_goal_wrist)
                assert len(episode) < max_len, f"EPISODE OF {len(episode)} STEPS! dist_to_desired: {np.round(info['dist_to_desired'], 3)}, dist_to_goal: {np.round(dist_to_goal, 3)}, gripper_tf: {np.round(env.get_robot_obs().gripper_tf, 3)}, base_tf: {np.round(env.get_robot_obs().base_tf, 3)}"
                # info['max_close_steps'] = True
                # done = True
            episode.add_step(action=action, obs=obs, reward=reward, done=done, info=info, unscaled_action=env.unwrapped._convert_policy_to_env_actions(action))

            if env.vis_env and done:
                print(f"final dist_to_desired: {info['dist_to_desired']:.2f}, rot_dist_to_desired: {info['rot_dist_to_desired']:.2f}")
                print(f"max dist_to_desired: {np.max([i['dist_to_desired'] for i in episode.infos]):.3f}, rot_dist_to_desired: {np.max([i['rot_dist_to_desired'] for i in episode.infos]):.3f}")
                print(f"max dist_to_motion: {np.max(episode.gripper_dists_to_motion[0]):.3f}, rot_dist_to_motion: {np.max(episode.gripper_dists_to_motion[1]):.3f}")
                # plt.close(); plt.plot([i['dist_to_desired'] for i in episode.infos]); plt.plot([i['rot_dist_to_desired'] for i in episode.infos])
                # plt.plot(episode.gripper_dists_to_motion[0]); plt.plot(episode.gripper_dists_to_motion[1]);

    return episode


def get_task_prefix(env, prefix: str) -> str:
    prefix = f"{prefix + '_' if prefix else ''}"
    prefix += env.loggingname
    if env.get_world() != 'sim':
        prefix += f'_ts{env.rate.sleep_dur.to_sec()}'
    return prefix


def evaluation_rollout(policy, env, num_eval_episodes: int, global_step: Optional[int], eval_seed: int, verbose: bool = True,
                       name_prefix: str = '', deterministic: bool = True, debug: bool = False):
    name_prefix = get_task_prefix(env, name_prefix)
    gamma = policy.gamma if hasattr(policy, "gamma") else policy.config["gamma"]

    episodes = []
    with torch.no_grad():
        for i in range(num_eval_episodes):
            t, t_ros = time.time(), rospy.get_time()
            if eval_seed is not None:
                env.seed(eval_seed + i)

            max_len = 20_000 if env.is_analytical_world() else 100_000
            episode = rollout(env=env, policy=policy, deterministic=deterministic, max_len=max_len, replay_episode=None)
            # rollout(env=env, policy=policy, deterministic=False, max_len=100_000, replay_episode=episode).nr_kin_failures
            episodes.append(episode)

            if (verbose > 1) or (env.get_world() != "sim"):
                rospy.loginfo(f"{name_prefix}: Eval ep {i}: {(time.time() - t) / 60:.2f} min, {(rospy.get_time() - t_ros) / 60:.2f} ros-min, {len(episode)} steps. "
                              f"{sum([e.success for e in episodes])}/{i + 1} success, {sum([e.success_nojumps for e in episodes])}/{i + 1} success_nojumps, {sum([e.deviation_success(0.1, 0.05) for e in episodes])}/{i + 1} motionD0.1RD0.05, "
                              f"{sum(np.array([e.nr_base_collisions for e in episodes]) == 0)}/{i + 1} no collisions, {sum(np.array([e.nr_kin_failures for e in episodes]) == 0)}/{i + 1} no ik fail, "
                              f"max_close_steps_reached: {sum([e.max_close_steps_reached for e in episodes])}/{i + 1}. "
                              f"Ik failures: {episode.nr_kin_failures}, base coll: {episode.nr_base_collisions}, steps_w_jump: {episode.steps_above_joint_vel_limit}.")

    log_dict = {}
    if env.learn_vel_norm:
        log_dict[f'{name_prefix}/vel_norm'] = np.mean([np.mean([s[0] for s in e.unscaled_actions]) for e in episodes])

    if global_step is None:
        if wandb.run.mode in ["dryrun", "offline", "disabled"]:
            global_step = 0
        else:
            global_step = get_next_global_step(debug=debug, increment=5)

    fails_per_episode = np.array([e.nr_kin_failures for e in episodes])
    metrics = {f'return_undisc':        np.mean([e.total_reward for e in episodes]),
               f'return_disc':          np.mean([e.get_disc_return(gamma) for e in episodes]),
               f'epoch_len':            np.mean([len(e) for e in episodes]),
               # f'ik_b{ik_fail_thresh}': np.mean(fails_per_episode <= ik_fail_thresh),
               f'ik_b11':               np.mean(fails_per_episode < 11),
               f'ik_zero_fail':         np.mean(fails_per_episode == 0),
               f'ik_fails':             np.mean(fails_per_episode),
               f'selfcollisions':       np.mean([e.nr_selfcollisions > 0 for e in episodes]),
               f'goal_reached':         np.mean([e.goal_reached for e in episodes]),
               f'goal_reached_nocollision': np.mean([e.goal_reached_no_collision for e in episodes]),
               f'base_collisions':      np.mean([e.nr_base_collisions > 0 for e in episodes]),
               f'success':              np.mean([e.success for e in episodes]),
               f'success_nojumps':      np.mean([e.success_nojumps for e in episodes]),
               f'success_nojumps3x':    np.mean([e.success_nojumps3x for e in episodes]),
               f'success_motionD0.1RD0.05': np.mean([e.deviation_success(0.1, 0.05) for e in episodes]),
               f'success_motionD0.05RD0.05': np.mean([e.deviation_success(0.05, 0.05) for e in episodes]),
               f'success_motion_avgdist': np.nanmean([e.gripper_dists_to_motion[0].mean() if e.success else np.nan for e in episodes]),
               f'success_motion_maxdist': np.nanmean([e.gripper_dists_to_motion[0].max() if e.success else np.nan for e in episodes]),
               f'success_motion_avgrotdist': np.nanmean([e.gripper_dists_to_motion[1].mean() if e.success else np.nan for e in episodes]),
               f'nr_jumps':             np.mean([e.total_nr_jumps for e in episodes]),
               f'vel_normed_avg_deviation': np.mean([e.vel_normed_avg_deviation for e in episodes]),
               f'steps_above_joint_vel_limit': np.mean([e.steps_above_joint_vel_limit for e in episodes]),
               f'steps_above_3x_joint_vel_limit': np.mean([e.steps_above_3x_joint_vel_limit for e in episodes]),
               f'max_close_steps_reached': np.mean([e.max_close_steps_reached for e in episodes]),
               'global_step':           global_step,
               'timesteps_total':       global_step}
    if not env.is_analytical_world():
        # TODO: should this be distance to current robot_obs or previous robot_obs, as we've just sent the command?
        metrics['gripper_dist_to_achieved_avg'] = np.mean([np.mean(e.gripper_dists_to_achieved) for e in episodes])
        metrics['gripper_dist_to_achieved_max'] = np.mean([np.max(e.gripper_dists_to_achieved) for e in episodes])
        metrics['gripper_rot_dist_to_achieved_avg'] = np.mean([np.mean(e.gripper_rot_dists_to_achieved) for e in episodes])
        metrics['gripper_rot_dist_to_achieved_max'] = np.mean([np.max(e.gripper_rot_dists_to_achieved) for e in episodes])

    rospy.loginfo("---------------------------------------")
    rospy.loginfo(f"T {global_step}, {name_prefix:} evaluation over {num_eval_episodes:.0f} episodes: "
                  f"Avg. return (undisc) {metrics[f'return_undisc']:.2f}, (disc) {metrics[f'return_disc']:.2f}, Avg failures {metrics[f'ik_fails']:.2f}, "
                  f"Avg. base coll.: {metrics[f'base_collisions']:.2f}, Avg success: {metrics['success']:.2f}p")
    rospy.loginfo("---------------------------------------")

    logmetrics = {(f'{name_prefix}/{k}' if (k not in ('global_step', 'timesteps_total')) else k): v for k, v in metrics.items()}
    log_dict.update(logmetrics)
    wandb.log(log_dict)

    plt.close('all')
    return metrics, name_prefix, episodes


def evaluate_on_task(wandb_config, policy, eval_env_config, task: str, world_type: str, global_step: Optional[int], eval_seed: int, debug: bool = False):
    eval_env_config = eval_env_config.copy()
    eval_env_config['task'] = task
    eval_env_config['world_type'] = world_type
    env = env_creator(eval_env_config)

    rospy.loginfo(f"Evaluating on task {env.loggingname} with {world_type} execution.")

    metrics, name_prefix, episodes = evaluation_rollout(policy,
                                                        env,
                                                        wandb_config["nr_evaluations"],
                                                        global_step=global_step,
                                                        verbose=2,
                                                        debug=debug,
                                                        eval_seed=eval_seed)
    env.clear()
    return metrics, episodes
