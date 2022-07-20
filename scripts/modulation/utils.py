import argparse
import copy
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rostopic
import sys
import torch
import wandb
import yaml
from gym import spaces, Wrapper
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from supersuit.generic_wrappers.frame_skip import frame_skip_gym
from supersuit.utils.frame_skip import check_transform_frameskip

from modulation import __version__
from modulation.dotdict import DotDict
from modulation.envs import ALL_TASKS
from modulation.envs.combined_env import CombinedEnv
from modulation.envs.env_utils import calc_euclidean_tf_dist
from modulation.envs.robotenv import RobotEnv


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    all_tasks = ALL_TASKS.keys()

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_best_defaults', type=str2bool, nargs='?', const=True, default=True, help="Replace default values with those from configs/best_defaults.yaml.")
    parser.add_argument('--seed', type=int, default=-1, help="Set to a value >= to use deterministic seed")
    parser.add_argument('--total_steps', type=int, default=1_000_000, help='Total number of action/observation steps to take over all episodes')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    #################################################
    # ALGORITHMS
    #################################################
    parser.add_argument('--algo', type=str.lower, default='sac', choices=['sac', 'bi2rrt', 'moveit', 'articulated'])
    parser.add_argument('--gamma', type=float, default=0.99, help='discount')
    parser.add_argument('--lr_start', type=float, default=0.0001)
    parser.add_argument('--lr_end', type=float, default=1e-6, help="Final / min learning rate. -1 to not decay")
    # parser.add_argument('--lr_gamma', type=float, default=0.999, help='adam decay')
    parser.add_argument('--tau', type=float, default=0.001, help='target value moving average speed')
    parser.add_argument('--explore_noise_type', type=str.lower, default='egreedy', choices=['normal', 'ou', 'egreedy', ''], help='Type of exploration noise')
    parser.add_argument('--explore_noise', type=float, default=0.0, help='')
    parser.add_argument('--explore_noise_final_scale', type=float, default=1.0, help='Set to 0.0 to anneal noise linearly to 0, 1 to not anneal at all. Only in ray.')
    parser.add_argument('--nstep', type=int, default=1, help='Use nstep returns. Currently only correct TD3. Requires a specific stable-baselines branch.')
    parser.add_argument('--fcnet_hiddens', nargs='+', type=int, default=[512, 512, 256], help='Fully connected layers. Ray only.')
    parser.add_argument('--mapencoder', type=str.lower, default='localdouble', choices=['local', 'localdouble'], help='Map encoder structure. Only in ray.')
    #################################################
    # SAC
    #################################################
    parser.add_argument('--ent_coef', default="auto", help="Entropy coefficient. 'auto' to learn it.")
    parser.add_argument('--initial_alpha', type=float, default=1.0, help="Initial alpha value. Only in ray atm.")
    parser.add_argument('--target_ent', default="auto", help="Target entropy. 'auto' to learn it. Only in ray atm.")
    parser.add_argument('--ent_lr', type=float, default=None, help="Alpha learning rate. Set to None to use lr_start. Set to 0 if ent_coef != auto.")
    parser.add_argument('--no_done_at_end', type=str2bool, nargs='?', const=True, default=True, help="Continue bootstrapping the values when episode is done. Only in ray.")
    parser.add_argument('--training_intensity', type=float, default=1, help='Training intensity. Only in ray.')
    parser.add_argument('--prioritizedrp', type=str2bool, nargs='?', const=True, default=False, help='Whether to use prioritized replay. Only in ray.')
    #################################################
    # Planning baselines
    #################################################
    parser.add_argument('--planner_max_iterations_time', type=int, default=200, help="Max. time for the planner.")
    parser.add_argument('--bi2rrt_extend_step_factor', type=float, default=0.4, help='extend_step_factor, original value is 0.4.')
    parser.add_argument('--moveit_max_waypoint_distance', type=float, default=0.2, help='Maximum waypoint distance.')
    parser.add_argument('--moveit_num_planning_attempts', type=int, default=10, help='num_planning_attempts.')
    parser.add_argument('--moveit_planner', type=str.lower, choices=['rrtstar', 'rrtconnect'], default='rrtconnect', help='Which planner to use.')
    parser.add_argument('--moveit_range', type=float, default=-1, help='range parameter for planners (max size of motions steps). -1 to use default value.')
    parser.add_argument('--moveit_orientation_constraint', type=str2bool, default=False, help='moveit_orientation_constraint.')
    #################################################
    # Articulated baseline
    #################################################
    parser.add_argument('--articulated_collision_mu', type=float, default=500, help='Collision constraint mu.')
    parser.add_argument('--articulated_collision_delta', type=float, default=0.001, help='Collision constraint delta.')
    parser.add_argument('--articulated_jv_mu', type=float, default=0.01, help='Joint value limit constraint mu.')
    parser.add_argument('--articulated_jv_delta', type=float, default=0.001, help='Joint value limit constraint delta.')
    parser.add_argument('--articulated_time_horizon', type=float, default=4.0, help='mpc time horizon.')
    parser.add_argument('--articulated_min_step', type=float, default=0.01, help='mpc min step size.')
    #################################################
    # Env
    #################################################
    parser.add_argument('--env', type=str.lower, default='pr2', choices=['pr2', 'tiago', 'hsr'], help='')
    parser.add_argument('--use_map_obs', type=str2bool, nargs='?', const=True, default=True, help='Observe a local obstacle map')
    parser.add_argument('--global_map_resolution', type=float, default=0.025, help='Resolution to use for global map')
    parser.add_argument('--local_map_resolution', type=float, default=0.025, help='Resolution to use for local map')
    parser.add_argument('--overlay_plan', type=str2bool, nargs='?', const=True, default=False, help='Whether to overlay the ee plan onto the local map')
    parser.add_argument('--concat_plan', type=str2bool, nargs='?', const=True, default=False, help='Whether to concat main points of the ee plan to the robot state')
    parser.add_argument('--concat_prev_action', type=str2bool, nargs='?', const=True, default=True, help='Whether to concat the previous action to the robot state')
    parser.add_argument('--collision_penalty', type=float, default=10, help='Penalty per base collision. Only used if use_map_obs==True.')
    parser.add_argument('--task', type=str.lower, default='simpleobstacle', choices=all_tasks, help='Train on a specific task env. Might override some other choices.')
    parser.add_argument('--obstacle_config', type=str.lower, default='rnd', choices=['none', 'inpath', 'rnd', 'dyn'], help='Obstacle configuration for ObstacleConfigMap. Ignored for all other tasks')
    parser.add_argument('--time_step', type=float, default=0.02, help='Time steps at which the RL agent makes decisions during actual execution. NOTE: time_step for training is hardcoded in robot_env.cpp.')
    parser.add_argument('--world_type', type=str, default="sim", choices=["sim", "gazebo", "world"], help="What kind of movement execution and where to get updated values from. Sim: analytical environemt, don't call controllers, gazebo: gazebo simulator, world: real world")
    parser.add_argument('--ik_fail_thresh', type=int, default=20, help='number of failures after which on it is considered as failed (i.e. failed: failures > ik_fail_thresh)')
    parser.add_argument('--ik_fail_thresh_eval', type=int, default=50, help='different eval threshold to make comparable across settings and investigate if it can recover from failures')
    parser.add_argument('--penalty_scaling', type=float, default=0.0, help='by how much to scale the penalties to incentivise minimal modulation')
    parser.add_argument('--acceleration_penalty', type=float, default=0.0, help='incentivise a small _change_ in base actions')
    parser.add_argument('--learn_vel_norm', type=float, default=-1, help="Learn the norm of the next EE-motion. Value is the factor weighting the loss for this. -1 to not learn it.")
    parser.add_argument('--vis_env', type=str2bool, nargs='?', const=True, default=False, help='Whether to publish markers to rviz')
    parser.add_argument('--transition_noise_base', type=float, default=0.0, help='Std of Gaussian noise applied to the next base transform during training, as percentage of the max velocity (0.01 for 1%)')
    parser.add_argument('--simpleobstacle_spacing', type=float, default=1.75, help='Space between obstacles in the simpleobstacle task in meters.')
    parser.add_argument('--simpleobstacle_offsetstd', type=float, default=1.75/4., help='Offset std for v3.')
    parser.add_argument('--use_fwd_orientation', type=str2bool, nargs='?', const=True, default=False, help='For the PointToPoint planner whether to rotate the ee to always look forward')
    parser.add_argument('--frame_skip', nargs='+', type=int, default=0, help='Whether to repeat the same action for this number of frames. Set to 0 during evaluation. Pass in a tuple to randomly chose a skip from a range.')
    parser.add_argument('--frame_skip_observe', type=str2bool, nargs='?', const=True, default=True, help='Whether to observe the drawn frame_skip')
    parser.add_argument('--frame_skip_curriculum', type=int, default=0, help='Whether to anneal frame_skip either towards 1 (scalar frame_skip) or the lower value of the range over this many steps. 0 or None for no schedule.')
    parser.add_argument('--iksolver', type=str.lower, default='bioik', choices=['default', 'bioik'], help="Which ik solver to use")
    parser.add_argument('--selfcollision_as_failure', type=str2bool, nargs='?', const=True, default=False, help='Whether to count self-collisions as failure and penalize them additionally or not. Note that if in self-collision, we keep the previous joint values that are not in self-collision.')
    parser.add_argument('--bioik_center_joints_weight', type=float, default=0.0, help='bioik.')
    parser.add_argument('--bioik_avoid_joint_limits_weight', type=float, default=1.0, help='bioik.')
    parser.add_argument('--bioik_regularization_weight', type=float, default=1.0, help='bioik.')
    parser.add_argument('--bioik_regularization_type', type=str.lower, default='mindispl', choices=['reg', 'mindispl', 'mymindispl', 'vellimit', 'abovevellimit'], help='bioik.')
    parser.add_argument('--learn_torso', type=str2bool, nargs='?', const=True, default=False, help='Whether to learn the torso joint with RL.')
    parser.add_argument('--learn_joint_values', type=str2bool, nargs='?', const=True, default=False, help='Whether to learn the arm joint velocities with RL.')
    #################################################
    # IK Slack
    #################################################
    parser.add_argument('--ikslack_dist', type=float, default=0.1, help='Allowed slack for the ik solution')
    parser.add_argument('--ikslack_rot_dist', type=float, default=0.05, help='Allowed slack for the ik solution')
    parser.add_argument('--ikslack_sol_dist_reward', type=str.lower, default='l2', choices=['ik_fail', 'l2'], help="'slack': penalise distance to perfect ik solution, 'l2': always use l2 norm instead of binary ik_failure penalty")
    parser.add_argument('--ikslack_penalty_multiplier', type=float, default=1.0, help='Multiplier for the ikslack penalty')
    #################################################
    # Eval
    #################################################
    parser.add_argument('--nr_evaluations', type=int, default=50, help='Nr of runs for the evaluation')
    parser.add_argument('--evaluation_frequency', type=int, default=75000, help='In nr of steps')
    parser.add_argument('--evaluation_only', type=str2bool, nargs='?', const=True, default=False, help='If True only model will be loaded and evaluated no training')
    parser.add_argument('--eval_execs', nargs='+', default=['sim'], choices=['sim', 'gazebo', 'world'], help='Eval execs to run')
    parser.add_argument('--eval_tasks', nargs='+', default=['rndstartrndgoal', 'houseexpo', 'picknplace', 'door', 'drawer'], choices=all_tasks, help='Eval tasks to run')
    parser.add_argument('--eval_seed', type=int, default=999, help='Seed used for evaluations (re-set ot this seed at each new evaluation). None to use new random seeds every evaluation.')
    parser.add_argument('--exec_action_clip', type=float, default=None, help='Value to clip base actions at. Applied to [-1, 1] range.')
    parser.add_argument('--exec_action_scaling', type=float, default=None, help='Factor scaling down the action ranges for the base. Applied to [-1, 1] range.')
    parser.add_argument('--exec_acceleration_clip', type=float, default=None, help='Clip base actions if difference to previous larger than this. Applied to [-1, 1] range.')
    parser.add_argument('--execute_style', type=str.lower, default="track",  choices=['direct', "track"], help='How to execute the rl actions.')
    parser.add_argument('--perception_style', type=str.lower, default="base",  choices=['base', "all", "none"], help='How to execute the rl actions.')
    #################################################
    # Ray Deployment
    #################################################
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--num_cpus_per_worker', type=float, default=1, help='Number of cpus per worker')
    parser.add_argument('--num_envs_per_worker', type=int, default=1, help='Number of cpus per worker')
    parser.add_argument('--num_gpus', type=float, default=0, help='Number of gpus for the trainer')
    parser.add_argument('--num_gpus_per_worker', type=float, default=0, help='Number of gpus per worker')
    parser.add_argument('--ray_verbosity', type=int, default=1, help='Ray tune verbosity')
    # #################################################
    # wandbstuff
    #################################################
    parser.add_argument('--resume_id', type=str, default=None, help='wandb id to resume')
    parser.add_argument('--resume_model_name', type=str, default='last_model.zip', help='If specifying a resume_id, which model to restore')
    parser.add_argument('--model_file', type=str, default=False, help='Restore the model and config saved in /scripts/model_checkpoints/${model_file}')
    parser.add_argument('--name', type=str, default="", help='wandb display name for this run')
    parser.add_argument('--name_suffix', type=str, default="", help='suffix for the wandb name')
    parser.add_argument('--group', type=str, default=None, help='wandb group')
    parser.add_argument('--use_name_as_group', type=str2bool, nargs='?', const=True, default=True, help='use the name as group')
    parser.add_argument('--project_name', type=str, default='mobile_rl', help='wandb project name')
    parser.add_argument('-d', '--dry_run', type=str2bool, nargs='?', const=True, default=False, help='whether not to log this run to wandb')
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help='log gradients to wandb, potentially extra verbosity')
    return parser


def create_run_name(args) -> str:
    parser = get_parser()

    n = args['name']
    if not n:
        n = []
        for k, v in sorted(args.items()):
            if (k in parser.parse_known_args()[0]) and (v != parser.get_default(k)) \
                and (not k.startswith('bioik')) \
                and not (args['algo'] != 'moveit' and 'moveit' in k) \
                and not (args['algo'] != 'articulated' and 'articulated' in k) \
                    and (k not in ['env', 'seed', 'load_best_defaults', 'name_suffix', 'name', 'evaluation_only',
                                   'vis_env', 'resume_id', 'eval_tasks', 'eval_execs', 'total_steps',
                                   'init_controllers', 'group',
                                   'num_workers', 'num_cpus_per_worker', 'num_envs_per_worker',
                                   'num_gpus', 'num_gpus_per_worker', 'ray_verbosity']):
                if type(v) == str:
                    s = str(v)
                elif type(v) == bool:
                    s = f'{k}:T' if v else f'{k}:F'
                elif type(v) == float:
                    s = f'{k}:{round(v, 7)}'
                elif type(v) == list:
                    # somehow wandb is not syncing checkpoints if the name includes []
                    s = f"{k}:{','.join([str(vv) for vv in v])}"
                else:
                    s = f'{k}:{v}'
                n.append(s)
        n.append(f"{args['bioik_regularization_type']}{args['bioik_regularization_weight']},{args['bioik_center_joints_weight']},{args['bioik_avoid_joint_limits_weight']}")
        n = '_'.join(n)
    return '_'.join([j for j in [args['env'], n, args['name_suffix']] if j])


def parse_args(config_path, framework: str, using_wandb: bool = False, raw_args=None):
    assert framework in ['ray', 'sbl']
    parser = get_parser()
    args = parser.parse_args(raw_args)
    args = vars(args)

    # user-specified command-line arguments
    if raw_args is None:
        raw_args = sys.argv
    cl_args = [k.replace('-', '').replace(" ", "=").split('=')[0] for k in raw_args]

    if args.pop('load_best_defaults'):
        with open(config_path / 'configs' / f'best_defaults_{framework}.yaml') as f:
            best_defaults = yaml.safe_load(f)
        # replace with best_default value unless something else was specified through command line
        for k, v in best_defaults[args['env']].items():
            if k not in cl_args:
                args[k] = v

    # consistency checks for certain arguments
    if args['resume_id'] or args['model_file']:
        assert args['evaluation_only'], "Continuing to train not supported atm (replay buffer doesn't get saved)"
    if args['evaluation_only']:
        if not (args['resume_id'] or args['model_file'] or (args['algo'] == 'unmodulated')):
            print("Evaluation only but no model to load specified! Evaluating a randomly initialised agent.")
    if args['algo'] == 'articulated':
        args['frame_skip'] = 0
        args['learn_torso'] = False
    if isinstance(args['frame_skip'], list) and len(args['frame_skip']) == 1:
        args['frame_skip'] = args['frame_skip'][0]
    elif isinstance(args['frame_skip'], list):
        args['frame_skip'] = tuple(args['frame_skip'])
    if args['acceleration_penalty']:
        assert args['concat_prev_action'], "Not fulfilling markov property if agent doesn't observe previous action"
    if args["learn_joint_values"]:
        args["learn_torso"] = True
        args['frame_skip'] = 0
        args['frame_skip_curriculum'] = 0

    # do we need to initialise controllers for the specified tasks?
    tasks_that_need_controllers = [k for k, v in ALL_TASKS.items() if v.requires_simulator()]
    task_needs_gazebo = len(set([args['task']] + args['eval_tasks']).intersection(set(tasks_that_need_controllers))) > 0
    world_type_needs_controllers = set([args['world_type']] + args['eval_execs']) != {'sim'}
    args['init_controllers'] = task_needs_gazebo or world_type_needs_controllers

    if args['init_controllers']:
        print('Initialising controllers')

    group = args.pop('group')
    use_name_as_group = args.pop('use_name_as_group')
    run_name = create_run_name(args)

    if use_name_as_group:
        assert not group, "Don't specify a group and use_name_as_group"
        rname = run_name[:99] if len(run_name) > 99 else run_name
        group = rname + f'_v{__version__}'
    args['group'] = group

    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['logpath'] = f'{config_path}/logs'
    os.makedirs(args['logpath'], exist_ok=True)
    args['version'] = __version__

    if not using_wandb:
        if args['seed'] == -1:
            args['seed'] = random.randint(10, 1000)
        set_seed(args['seed'])

    print(f"Log path: {args['logpath']}")

    return run_name, group, args, cl_args


def launch_ros(main_path: Path, config, task: str, no_gui: bool = False, always_relaunch: bool = None, algo="rl", pure_analytical=None):
    if not always_relaunch:
        try:
            # check if ros is already running, if yes, assume it was started manually
            rostopic.get_topic_class('/rosout')
            return
        except:
            # kill any existing ros stuff
            _ = subprocess.run(["rosnode", "kill", "-a"])
            _ = subprocess.run(["killall", "-9", "gzserver", "gzclient"])

    script = str(main_path / "modulation" / "handle_launchfiles.py")

    if pure_analytical is None:
        pure_analytical = "no" if config["init_controllers"] else "yes"
    args = ["--env", config['env'], "--algo", algo, "--pure_analytical", pure_analytical, "--task", task]
    if config.iksolver == "bioik":
        args += ["--bioik"]
    if config.debug and not no_gui:
        args += ["--gui"]

    # remove any references to the conda env, so that roslaunch really uses python2 in everything it starts
    # NOTE: assumes the conda env is called "modulation_rl"
    my_env = os.environ.copy()
    conda_path = Path(shutil.which("conda"))
    my_env["PATH"] = my_env["PATH"].replace(str(conda_path.parent.parent / "envs" / "modulation_rl" / "bin") + ":", "")
    my_env["PATH"] = my_env["PATH"].replace(str(conda_path.parent) + ":", "")

    out = subprocess.run(["python2", script] + args, env=my_env)
    assert out.returncode == 0, out


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_wandb_group(wandb_run_id,  project_name='mobile_rl'):
    api = wandb.Api()
    run = api.run(("%s/%s/%s" % (api.default_entity, project_name, wandb_run_id)))
    return run.config.get('group', '')


def setup_config_wandb(config_path, framework: str, sync_tensorboard=False, allow_init: bool = True,
                       no_ckpt_endig: bool = False, raw_args=None):
    run_name, group, args, cl_args = parse_args(config_path, using_wandb=True, framework=framework, raw_args=raw_args)

    if args['nstep'] != 1:
        assert args['algo'] == 'td3', "Not correctly implemented for SAC yet"

    if args['dry_run']:
        os.environ['WANDB_MODE'] = 'dryrun'

    if no_ckpt_endig:
        args['resume_model_name'] = args['resume_model_name'].replace('.zip', '')

    if args['algo'] == 'articulated':
        args['frame_skip'] = 0
        args['learn_torso'] = False

    common_args = {'project': args.pop('project_name'),
                   'dir': args['logpath'],
                   'sync_tensorboard': sync_tensorboard}
    if args['resume_id']:
        assert not args['dry_run']
        common_args['group'] = get_wandb_group(args['resume_id'], project_name=common_args['project'])
        run = wandb.init(id=args['resume_id'],
                         resume=args['resume_id'],
                         **common_args)
    elif args['model_file']:
        model_file = config_path / 'model_checkpoints' / args['model_file']
        args['model_file'] = str(model_file)
        print(f"RESTORING MODEL from {model_file}")

        with open(model_file.parent / 'config.yaml', "rb") as f:
            raw_params = yaml.safe_load(f)
        params = {k: v['value'] for k, v in raw_params.items() if k not in ['_wandb', 'wandb_version']}

        params['model_file'] = False
        params['resume_id'] = None
        params['resume_model_name'] = None

        common_args['group'] = group

        run = wandb.init(config=params, **common_args)
        if args['evaluation_only']:
            wandb.config.update({"evaluation_only": True}, allow_val_change=True)
    else:
        if allow_init:
            # delete all past wandb runs. Mostly relevant for running sweeps in docker which might fill up the space o/w
            delete_dir(os.path.join(common_args['dir'], 'wandb'))
            common_args['group'] = group
            args['version'] = f'v{__version__}'
            run = wandb.init(config=args, name=run_name, **common_args)
        else:
            raise ValueError("Not allowed to initialise a new run. Sepcify restore_model=True or a resume_id")

    if args['resume_id'] or args['model_file']:
        # update an alternative dict placeholder so we don't change the logged values which it was trained with
        config = DotDict(copy.deepcopy(dict(wandb.config)))

        for k, v in args.items():
            # allow to override loaded config with command line args
            if k in cl_args:
                config[k] = v
            # backwards compatibility if a config value didn't exist before
            if k not in wandb.config.keys():
                print(f"Key {k} not found in config. Setting to {v}")
                config[k] = args[k]
        # always update these values
        for k in ['init_controllers', 'device', 'num_workers', 'num_cpus_per_worker', 'num_envs_per_worker', 'num_gpus',
                  'num_gpus_per_worker', 'logpath', 'simpleobstacle_spacing', 'obstacle_config']:
            config[k] = args[k]
    else:
        config = wandb.config

    # Set seeds
    # NOTE: if changing the args wandb will not get the change in sweeps as they don't work over the command line!!!
    if config.seed == -1:
        wandb.config.update({"seed": random.randint(10, 1000)}, allow_val_change=True)
        config['seed'] = wandb.config.seed

    set_seed(config.seed)

    return run, config


def delete_dir(dirname: str):
    try:
        print(f"Deleting dir {dirname}")
        shutil.rmtree(dirname)
    except Exception as e:
        print(f"Failed to delete dir {dirname}: {e}")


def traced(func, ignoredirs=None):
    """
    Decorates func such that its execution is traced, but filters out any
    Python code outside of the system prefix.
    https://drake.mit.edu/python_bindings.html#debugging-with-the-python-bindings
    """
    import functools
    import sys
    import trace
    if ignoredirs is None:
        ignoredirs = ["/usr", sys.prefix]
    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return tracer.runfunc(func, *args, **kwargs)

    return wrapped


def wrap_in_task(env, task: str, wrap_kwargs: dict, **env_kwargs):
    if isinstance(env, CombinedEnv):
        combined_env = env
    else:
        combined_env = env.unwrapped
    assert isinstance(combined_env, CombinedEnv), combined_env

    task_fn = ALL_TASKS[task.lower()]
    task_env = task_fn(combined_env, **env_kwargs)
    if task_env.requires_simulator():
        assert env._robot.get_init_controllers(), "We need gazebo to spawn objects etc"

    if env.ikslack_dist or env.ikslack_rot_dist:
        task_env = MaxCloseStepsWrapper(task_env, max_close_steps=50 if env.is_analytical_world() else 500)

    if wrap_kwargs['frame_skip']:
        task_env = TaskSettableEnvSkip(task_env, wrap_kwargs['frame_skip'], observe_frame_skip=wrap_kwargs['frame_skip_observe'], gamma=wrap_kwargs['gamma'])

    return task_env


def create_env(config,
               task: str,
               node_handle: str,
               eval: bool) -> CombinedEnv:
    print(f"Creating {config['env']}")

    robot_env = RobotEnv(env=config["env"],
                         node_handle_name=node_handle,
                         penalty_scaling=config["penalty_scaling"],
                         acceleration_penalty=config["acceleration_penalty"],
                         time_step_world=config["time_step"],
                         seed=config["seed"],
                         world_type=config["world_type"],
                         init_controllers=config["init_controllers"],
                         vis_env=config["vis_env"],
                         transition_noise_base=config["transition_noise_base"],
                         ikslack_dist=config["ikslack_dist"],
                         ikslack_rot_dist=config["ikslack_rot_dist"],
                         ikslack_sol_dist_reward=config["ikslack_sol_dist_reward"],
                         ikslack_penalty_multiplier=config["ikslack_penalty_multiplier"],
                         selfcollision_as_failure=config["selfcollision_as_failure"],
                         iksolver=config["iksolver"],
                         bioik_center_joints_weight=config["bioik_center_joints_weight"],
                         bioik_avoid_joint_limits_weight=config["bioik_avoid_joint_limits_weight"],
                         bioik_regularization_weight=config["bioik_regularization_weight"],
                         bioik_regularization_type=config["bioik_regularization_type"],
                         learn_torso=config['learn_torso'],
                         exec_action_clip=config.get('exec_action_clip', None),
                         exec_action_scaling=config.get('exec_action_scaling', None),
                         exec_acceleration_clip=config.get('exec_acceleration_clip', None),
                         execute_style=config.get('execute_style', "direct"),
                         perception_style=config.get('perception_style', "base"),
                         fake_gazebo=config.get('fake_gazebo', False))
    env = CombinedEnv(robot_env=robot_env,
                      ik_fail_thresh=config["ik_fail_thresh"],
                      learn_vel_norm_penalty=config["learn_vel_norm"],
                      collision_penalty=config['collision_penalty'],
                      use_map_obs=config["use_map_obs"],
                      overlay_plan=config['overlay_plan'],
                      concat_plan=config['concat_plan'],
                      concat_prev_action=config['concat_prev_action'],
                      global_map_resolution=config['global_map_resolution'],
                      local_map_resolution=config['global_map_resolution'],
                      learn_joint_values=config['learn_joint_values'])
    if task == 'simpleobstacle':
        env_kwargs = {'obstacle_spacing': config['simpleobstacle_spacing'],
                      'offset_std': config['simpleobstacle_offsetstd']}
    elif task in ['picknplace', 'door', 'drawer']:
        env_kwargs = {'obstacle_configuration': config['obstacle_config']}
    else:
        env_kwargs = {}
    if task not in ['door', 'drawer', 'roomdoor', 'doorhall', 'drawerhall', 'roomdoorhall', 'aisdoor', 'aisroomdoor', 'aisdrawer']:
        env_kwargs['use_fwd_orientation'] = config['use_fwd_orientation']
    env_kwargs['eval'] = eval

    assert not (eval and (config['frame_skip'] > 1)), f"Really want to use frame skip {config['frame_skip']} for evaluation?"

    wrap_kwargs = {"frame_skip": config['frame_skip'],
                   "frame_skip_observe": config["frame_skip_observe"],
                   "gamma": config.get('gamma') or config.get('player_gamma')}

    return wrap_in_task(env=env, task=task, wrap_kwargs=wrap_kwargs, **env_kwargs)


def mytimer(fn, n=100):
    import time
    t = time.time()
    for _ in range(n):
        fn()
    print(f"{(time.time() - t) / n:.5f}s on average")


def episode_is_success(nr_kin_fails: int, nr_collisions: int, goal_reached: bool) -> bool:
    return (nr_kin_fails == 0) and (nr_collisions == 0) and goal_reached


def env_creator(ray_env_config: dict):
    """Allows to construct a different eval env by defining 'task': eval_task in 'evaluation_config'"""
    # time.sleep(random.uniform(0.0, 0.5))
    env = create_env(ray_env_config,
                     task=ray_env_config['task'],
                     node_handle=ray_env_config["node_handle"],
                     eval=ray_env_config['eval'])
    return env


class TaskSettableEnvSkip(frame_skip_gym):
    def __init__(self, env, num_frames, observe_frame_skip: bool, gamma: float):
        super(TaskSettableEnvSkip, self).__init__(env, num_frames)
        self.observe_frame_skip = observe_frame_skip
        self.gamma = gamma

        if self.observe_frame_skip:
            if isinstance(self.observation_space, spaces.Box):
                self.observation_space = spaces.Box(shape=[self.observation_space.shape[0] + 1],
                                                    low=self.observation_space.low[0],
                                                    high=self.observation_space.high[0],
                                                    dtype=self.observation_space.dtype)
            elif len(self.observation_space) == 2:
                assert len(self.observation_space[0].shape) == 1, self.observation_space[0].shape
                self.observation_space = spaces.Tuple([spaces.Box(shape=[self.observation_space[0].shape[0] + 1],
                                                                  low=self.observation_space[0].low[0],
                                                                  high=self.observation_space[0].high[0],
                                                                  dtype=self.observation_space[0].dtype),
                                                       self.observation_space[1]])
            else:
                raise NotImplementedError(self.observation_space)

    def _draw_num_frames(self) -> int:
        return int(self.np_random.randint(self.num_frames[0], self.num_frames[1] + 1))

    @staticmethod
    def _attach_frameskip_to_obs(obs, frame_skip: int):
        normalized = frame_skip / 10.
        if isinstance(obs, (np.ndarray, list)):
            return np.append(obs, normalized).astype(np.float32)
        elif len(obs) == 2:
            assert isinstance(obs[0], list), type(obs[0])
            return (obs[0] + [normalized], obs[1])
        else:
            raise NotImplementedError(len(obs))

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.next_frame_skip = self._draw_num_frames()
        if self.observe_frame_skip:
            obs = self._attach_frameskip_to_obs(obs, self.next_frame_skip)
        return obs

    def step(self, action):
        total_reward = 0.0
        for t in range(self.next_frame_skip):
            # don't just do super() as o/w we'll do the frame_skip twice
            obs, rew, done, info = super(frame_skip_gym, self).step(action)
            total_reward += (self.gamma ** t) * rew
            if done:
                break

        info['frame_skip'] = self.next_frame_skip

        self.next_frame_skip = self._draw_num_frames()
        if self.observe_frame_skip:
            obs = self._attach_frameskip_to_obs(obs, self.next_frame_skip)
        return obs, total_reward, done, info

    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [self._draw_num_frames() for _ in range(n_tasks)]

    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.num_frames

    def set_task(self, num_frames):
        """Implement this to set the task (curriculum level) for this env."""
        self.num_frames = check_transform_frameskip(num_frames)


def frame_skip_curriculum_fn(train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext):
    """Function returning a possibly new task to set `task_settable_env` to.
    Args:
        train_results (dict): The train results returned by Trainer.train().
        task_settable_env (TaskSettableEnv): A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx (EnvContext): The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.
    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    assert not env_ctx['eval']
    max_frames = env_ctx['frame_skip'] if isinstance(env_ctx['frame_skip'], (float, int)) else env_ctx['frame_skip'][-1]
    timesteps_total_current = train_results['timesteps_total']
    timesteps_total_max = env_ctx['frame_skip_curriculum']
    pct_training_complete = min(timesteps_total_current / timesteps_total_max, 1.0)

    new_num_frames = max_frames - int(pct_training_complete * (max_frames - 1))
    if not isinstance(env_ctx['frame_skip'], (float, int)):
        new_num_frames = (env_ctx['frame_skip'][0], max(new_num_frames, env_ctx['frame_skip'][0]))

    return new_num_frames


class MaxCloseStepsWrapper(Wrapper):
    def __init__(self, env, max_close_steps: int):
        super(MaxCloseStepsWrapper, self).__init__(env)
        self._max_close_steps = max_close_steps
        self._elapsed_close_steps = None

    def step(self, action, **kwargs):
        # TODO: might not work correctly on GMM, as there it might be close for quite longer than 50 steps when the ee in only rotating in the end
        assert self._elapsed_close_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action, **kwargs)
        dist_to_goal = calc_euclidean_tf_dist(self.env.get_robot_obs().gripper_tf, self.env.unwrapped._ee_planner.gripper_goal_wrist)
        self._elapsed_close_steps += (dist_to_goal < info['dist_to_desired'] + 0.001)
        if self._elapsed_close_steps >= self._max_close_steps:
            print(f"REACHED MAX_CLOSE_STEPS with dist_to_goal {dist_to_goal:.4f}, dist_to_desired: {info['dist_to_desired']:.3f}")
            info['max_close_steps'] = not done
            # reward -= 0
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_close_steps = 0
        return self.env.reset(**kwargs)
