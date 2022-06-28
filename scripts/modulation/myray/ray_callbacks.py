from typing import Optional, Dict

import numpy as np
from ray.rllib import BaseEnv, RolloutWorker, Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import AgentID, PolicyID

from modulation.envs.env_utils import calc_disc_return
from modulation.utils import episode_is_success
from modulation.envs.eeplanner import PointToPointPlanner

def get_player_policy_names(values):
    if len(values) == 1 and 'default_policy' in values:
        # normal env
        return ['agent0']
    else:
        # adversarial env
        return [p.split('_')[0] for p in values if p.startswith('player')]


def set_early_termination_dones_false(*, postprocessed_batch: SampleBatch) -> None:
    """Continue to bootstrap after early terminations, but not after actually reaching the goal"""
    goal_reached = np.array([info['ee_done'] for info in postprocessed_batch[SampleBatch.INFOS]])
    postprocessed_batch[SampleBatch.DONES] = np.logical_and(postprocessed_batch[SampleBatch.DONES], goal_reached).astype(postprocessed_batch[SampleBatch.DONES].dtype)


class TrainCallback(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:
        for m in ('vel_normed_avg_deviation', 'above_joint_vel_limit', 'above_3x_joint_vel_limit', 'vel_norms', 'dist_to_desired', 'rot_dist_to_desired'):
            episode.user_data[m] = []

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        if hasattr(base_env, 'vector_env'):
            env = base_env.vector_env.envs[0].unwrapped
        else:
            env = base_env.get_unwrapped()[0]

        policy_names = get_player_policy_names(sorted(episode._policies.keys()))
        if (len(policy_names) == 1) or not episode.last_done_for(policy_names[0]):
            info = episode.last_info_for(policy_names[0])
            for m in ('vel_normed_avg_deviation', 'above_joint_vel_limit', 'above_3x_joint_vel_limit', 'dist_to_desired', 'rot_dist_to_desired'):
                episode.user_data[m].append(info[m])
            episode.user_data[m].append(info[m])

            if env.learn_vel_norm:
                action = episode.last_action_for(policy_names[0])
                # it seems the above returns the unclipped actions
                action_clipped = np.clip(action, -1, 1)
                if action.sum() != 0.0:
                    vel_norm, base_actions, joint_value_deltas = env.unwrapped._convert_policy_to_env_actions(action_clipped)
                    episode.user_data['vel_norms'].append(vel_norm)

            maxlen = 20_000
            episode_len = len(episode.user_data[m])
            if episode_len and episode_len % 1000 == 0:
                print(F"EPISODE AT {episode_len} STEPS. CONTINUING TILL {maxlen}")
            assert episode_len < maxlen, f"Episode reached {episode_len} steps. This should not be possible"

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        policy_names = get_player_policy_names(sorted(policies.keys()))
        for p, pol in zip(policy_names, sorted(policies.keys())):
            last_info = episode.last_info_for(p)
            if last_info is not None:
                pname = p + '_' if p != 'agent0' else ''
                episode.custom_metrics[f"{pname}goal_reached"] = last_info['ee_done']
                episode.custom_metrics[f"{pname}ik_fails"] = last_info['nr_kin_failures']
                episode.custom_metrics[f"{pname}ik_zero_fail"] = (last_info['nr_kin_failures'] == 0)
                episode.custom_metrics[f"{pname}ik_b100"] = (last_info['nr_kin_failures'] < 100)
                episode.custom_metrics[f"{pname}base_collisions"] = last_info['nr_base_collisions']
                episode.custom_metrics[f"{pname}success"] = episode_is_success(nr_kin_fails=last_info['nr_kin_failures'],
                                                                               nr_collisions=last_info['nr_base_collisions'],
                                                                               goal_reached=last_info['ee_done'])
                episode.custom_metrics[f"{pname}vel_normed_avg_deviation"] = np.mean(episode.user_data['vel_normed_avg_deviation'])
                episode.custom_metrics[f"{pname}steps_above_joint_vel_limit"] = np.sum(episode.user_data['above_joint_vel_limit'])
                episode.custom_metrics[f"{pname}steps_above_3x_joint_vel_limit"] = np.sum(episode.user_data['above_3x_joint_vel_limit'])
                episode.custom_metrics[f"{pname}dist_to_desired"] = np.mean(episode.user_data['dist_to_desired'])
                episode.custom_metrics[f"{pname}rot_dist_to_desired"] = np.mean(episode.user_data['rot_dist_to_desired'])
                episode.custom_metrics[f"{pname}success_nojumps"] = episode.custom_metrics[f"{pname}success"] and (episode.custom_metrics[f"{pname}steps_above_joint_vel_limit"] == 0)
                episode.custom_metrics[f"{pname}success_nojumps3x"] = episode.custom_metrics[f"{pname}success"] and (episode.custom_metrics[f"{pname}steps_above_3x_joint_vel_limit"] == 0)
                episode.custom_metrics[f"{pname}max_close_steps_reached"] = float(last_info.get('max_close_steps', False))

                rs = np.array(episode._agent_reward_history[p])
                if len(policies) == 1:
                    # can only do this sanity check for single agent envs as o/w they might not take an action at every env step
                    assert len(rs) == episode.length, (episode.length, rs)

                if episode.user_data.get('vel_norms', None):
                    episode.custom_metrics[f'{pname}vel_norm'] = np.mean(episode.user_data['vel_norms'])

                disc_return = calc_disc_return(rewards=rs, gamma=policies[pol].config['gamma'])
                episode.custom_metrics[f"{pname}episode_disc_return"] = disc_return


    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            episode (MultiAgentEpisode): Episode object.
            agent_id (str): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches (dict): Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        if not policies[policy_id].config['no_done_at_end']:
            set_early_termination_dones_false(postprocessed_batch=postprocessed_batch)


class EvalCallback(TrainCallback):
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        TrainCallback.on_episode_end(self, worker=worker, base_env=base_env, policies=policies, episode=episode,
                                     env_index=env_index, **kwargs)

        assert policies.policy_config['evaluation_config']['env_config']['eval'], "Don't want to have this applied to the train env!"
        eval_seed = policies.policy_config['evaluation_config']['env_config']['eval_seed']
        nr_evaluations_per_eval = policies.policy_config['evaluation_num_episodes']

        if (not hasattr(self, "n_eval_episodes")) or (self.n_eval_episodes >= nr_evaluations_per_eval):
            self.n_eval_episodes = 0

        s = eval_seed + self.n_eval_episodes
        base_env.get_unwrapped()[episode.env_id].seed(s)

        self.n_eval_episodes += 1
