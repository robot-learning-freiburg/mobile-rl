import numpy as np
import rospy
import torch
from matplotlib import pyplot as plt


def calc_dist_to_sol(p):
    """NOTE: requires that planned_gripper_... is the plan that gripper_... tried to achieve"""
    return np.sqrt((p["gripper_x"] - p["planned_gripper_x"]) ** 2 +
                   (p["gripper_y"] - p["planned_gripper_y"]) ** 2 +
                   (p["gripper_z"] - p["planned_gripper_z"]) ** 2)


def move_straight(env, agent=None, gripper_goal=None, action=None, start_pose_distribution="fixed",
                  show_base=True, show_actual_gripper=True, show_planned_gripper=True):
    """Helper for quick testing. If it actually moves straight depends on config.strategy"""
    if agent is not None:
        rospy.loginfo("Using agent actions")
    elif action is None:
        action = np.zeros(env.action_space.shape)
        rospy.loginfo(f"Using action {action}")
    if gripper_goal is None:
        gripper_goal = (4, 0, 0.5, 0, 0, 0)

    with torch.no_grad():
        obs = env.reset(start_pose_distribution=start_pose_distribution, gripper_goal_distribution='rnd',
                        success_thres_dist=0.025, success_thres_rot=0.05, gripper_goal=gripper_goal, gmm_model_path="")
        # env.parse_obs(obs)

        done_return, i, actions = 0, 0, []
        while not done_return:
            if agent is not None:
                action = agent.predict(obs, deterministic=True)[0]
            # pathPoint = env.visualize()
            obs, reward, done_return, info = env.step(np.array(action))
            env.parse_obs(obs)
            actions.append(action)
            i += 1
            # if i > 1000:
            #     break
    pathPoint = env.visualize()

    shw = False
    if shw:
        f = env.plot_pathPoints([pathPoint], show_base=show_base, show_actual_gripper=show_actual_gripper,
                                show_planned_gripper=show_planned_gripper)
        plt.show()
        plt.close(f)

        plt.plot(np.diff([p["base_x"] for p in pathPoint]), label='base_vel_x')
        plt.plot(np.diff([p["gripper_x"] for p in pathPoint]), label='gripper_vel_x')
        plt.plot(np.diff([p["planned_gripper_x"] for p in pathPoint]), label='planned_gripper_vel_x')
        plt.plot([0.002 * p["ik_fail"] for p in pathPoint], label='ik_fail')
        plt.legend(); plt.show(); plt.close();

        # plt.plot(np.diff([p.gripper_z for p in pathPoint]), label='gripper_vel_z')
        # plt.plot(np.diff([p.planned_gripper_z for p in pathPoint]), label='planned_gripper_vel_z')
        # plt.plot([0.002 * p.ik_fail for p in pathPoint], label='ik_fail')
        # plt.legend(); plt.show(); plt.close();
        #
        # plt.plot([p.gripper_z for p in pathPoint], label='gripper_z')
        # plt.plot([p.planned_gripper_z for p in pathPoint], label='planned_gripper_z')
        # plt.plot([0.002 * p.ik_fail for p in pathPoint], label='ik_fail')
        # plt.legend(); plt.show(); plt.close();

        plt.plot(np.diff([p["gripper_x"] for p in pathPoint]) - np.diff([p["base_x"] for p in pathPoint]), label='gripper diff(x) - base diff(x)')
        plt.legend(); plt.show(); plt.close();

        plt.plot([p["gripper_rel_x"] for p in pathPoint], label='gripper_rel_x')
        plt.plot([p["gripper_rel_y"] for p in pathPoint], label='gripper_rel_y')
        plt.plot([p["gripper_rel_z"] for p in pathPoint], label='gripper_rel_z')
        plt.legend(); plt.show(); plt.close();

        plt.plot([p["planned_base_x"] - p.base_x for p in pathPoint], label='base x: planned - current')
        plt.plot([p["planned_gripper_x"] - p.gripper_x for p in pathPoint], label='gripper x: planned - current')
        plt.plot([p["planned_gripper_y"] - p.gripper_y for p in pathPoint], label='gripper y: planned - current')
        plt.plot([p["planned_gripper_z"] - p.gripper_z for p in pathPoint], label='gripper z: planned - current')
        plt.plot([0.2 * p["ik_fail"] for p in pathPoint], label='ik_fail')
        plt.legend(); plt.show(); plt.close();

        dists = [calc_dist_to_sol(p) for p in pathPoint]
        plt.plot(dists, label='dist from desired gripper')
        plt.plot([0.2 * p["ik_fail"] for p in pathPoint], label='ik_fail')
        plt.legend(); plt.show(); plt.close();

        plt.plot([p["desired_base_x"] for p in pathPoint], label='base x: desired')
        plt.plot([p["planned_base_x"] for p in pathPoint], label='base x: planned')
        plt.plot([p["base_x"] for p in pathPoint], label='base x: achieved')
        plt.plot([p["desired_base_x"] - p["base_x"] for p in pathPoint], label='base x: desired - achieved')
        plt.legend(); plt.show(); plt.close();

        plt.plot([p["desired_base_y"] for p in pathPoint], label='base y: desired')
        plt.plot([p["planned_base_y"] for p in pathPoint], label='base y: planned')
        plt.plot([p["base_y"] for p in pathPoint], label='base y: achieved')
        plt.plot([p["desired_base_y"] - p["base_y"] for p in pathPoint], label='base y: desired - achieved')
        plt.legend(); plt.show(); plt.close();

        plt.plot([p["desired_base_rot"] for p in pathPoint], label='base rot: desired')
        plt.plot([p["base_rot"] for p in pathPoint], label='base rot: achieved')
        plt.plot([p["desired_base_rot"] - p["base_rot"] for p in pathPoint], label='base rot: desired - achieved')
        plt.legend(); plt.show(); plt.close();

        # plt.plot([p.torso_desired for p in pathPoint], label='torso: desired')
        # plt.plot([p.torso_actual for p in pathPoint], label='torso: achieved')
        # plt.plot([p.torso_desired - p.torso_actual for p in pathPoint], label='torso: desired - achieved')
        # plt.legend(); plt.show(); plt.close();

        plt.plot([p["base_cmd_angular_z"] for p in pathPoint], label='base cmd: angular z')
        plt.plot([p["base_cmd_linear_x"] for p in pathPoint], label='base cmd: linear x')
        plt.plot([p["base_cmd_linear_y"] for p in pathPoint], label='base cmd: linear y')
        plt.plot([0.2 * p["ik_fail"] for p in pathPoint], label='ik_fail')
        plt.legend(); plt.show(); plt.close();

        plt.plot([p["collision"] for p in pathPoint], label='collisions')
        plt.legend(); plt.show(); plt.close();

        actions = np.array(actions)
        for i, n in enumerate(env.action_names):
            plt.plot(actions[:, i], label=n)
        plt.title('actions'); plt.legend(); plt.show(); plt.close();

    rospy.loginfo(f"N collisions detected: {sum([p['collision'] for p in pathPoint])}")
    rospy.loginfo(f"length of the episode: {len(pathPoint)}")