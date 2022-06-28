import os
import pickle
import random
import shutil
from typing import Callable, Dict, Tuple, Optional
import copy
from pathlib import Path
import wandb
from ray.rllib.utils.typing import TrainerConfigDict
from ray.tune import Trainable
from ray.tune.function_runner import FunctionRunner
from ray.tune.integration.wandb import _set_api_key, _clean_log, _is_allowed_type
from ray.tune.logger import Logger, NoopLogger
from ray.tune.utils import flatten_dict

from modulation.evaluation import evaluate_on_task, download_wandb
from modulation.myray.custom_sac import SACTrainerCustom
from modulation.utils import create_run_name

RAY_METADATA_ENDING = '_trainableState'


def wandb_handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.pop("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if not _is_allowed_type(v):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update


class MyWandbTrainableMixin:
    _wandb = wandb

    def __init__(self, config: Dict, *args, **kwargs):
        if not isinstance(self, Trainable):
            raise ValueError(
                "The `WandbTrainableMixin` can only be used as a mixin "
                "for `tune.Trainable` classes. Please make sure your "
                "class inherits from both. For example: "
                "`class YourTrainable(WandbTrainableMixin)`.")

        # super().__init__(config, *args, **kwargs)

        _config = config.copy()

        if _config['dry_run']:
            os.environ['WANDB_MODE'] = 'dryrun'

        try:
            wandb_config = _config.pop("wandb").copy()
        except KeyError:
            raise ValueError(
                "Wandb mixin specified but no configuration has been passed. "
                "Make sure to include a `wandb` key in your `config` dict "
                "containing at least a `project` specification.")

        api_key_file = wandb_config.pop("api_key_file", None)
        if api_key_file:
            api_key_file = os.path.expanduser(api_key_file)

        _set_api_key(api_key_file, wandb_config.pop("api_key", None))

        # Fill trial ID and name
        assert not _config['resume_id'], "IF RESTORING ASSIGN THIS THE SAME ID AS THE RESTORED CKPT IS COMING FROM (THOUGH MAYBE EASIER NOT TO USE RAY FOR EVALUATION AT ALL AND JUST WRITE MY OWN FUNCTION)"
        trial_id = self.trial_id if self.trial_id != 'default' else f"debug{random.randint(0, 99999)}"
        trial_name = self.trial_name

        # Project name for Wandb
        try:
            wandb_project = wandb_config.pop("project")
        except KeyError:
            raise ValueError(
                "You need to specify a `project` in your wandb `config` dict.")

        # Grouping
        if isinstance(self, FunctionRunner):
            default_group = self._name
        else:
            default_group = type(self).__name__
        wandb_group = wandb_config.pop("group", default_group)

        # remove unpickleable items!
        _config = _clean_log(_config)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=trial_name,
            resume=True,
            reinit=True,
            allow_val_change=False,
            group=wandb_group,
            project=wandb_project,
            config=_config,
            sync_tensorboard=False)
        wandb_init_kwargs.update(wandb_config)

        self.wandb = self._wandb.init(**wandb_init_kwargs)
        # self._wandb.tensorboard.patch(tensorboardX=True)

    def stop(self):
        self._wandb.join()
        if hasattr(super(), "stop"):
            super().stop()


def _log_results(trainer, result: dict):
    trainer.callbacks.on_train_result(trainer=trainer, result=result)
    result = _clean_log(result)
    processed_result, config_update = wandb_handle_result(result)

    # NOTE: can lead to port errors when running many envs in parallel, e.g. during hyperpara search
    # don't log too many images for SAC which seems to have much smaller train_iterations
    # if isinstance(trainer, PPOTrainer) or (random.random() < 0.15):
    #     task = trainer.config['env_config']['task']
    #     if (task in ['houseexpo', 'hadversarial']):
    #         # NOTE: has to pickle the env to get it here from the worker. But just executing the plot function on the worker instead
    #         # fails within plt.canvas with 'object is readonly'
    #         # get access to the env in a hacky way
    #         def get_env(worker):
    #             return worker.env
    #         first_worker = trainer.workers.local_worker()
    #         if first_worker and first_worker.env:
    #             env = first_worker.apply(get_env)
    #         else:
    #             first_worker = trainer.workers.remote_workers()[0]
    #             env = ray.get(first_worker.apply.remote(get_env))
    #
    #         f, ax = env.plot_trajectory()
    #         processed_result['map_traj'] = wandb.Image(f)
    #         plt.close('all')

    wandb.log(processed_result)


def _init(self, parent_class, config, env, logger_creator):
    config = config.copy()

    nonwandb_config = config.copy()
    wandb_mixin_config = nonwandb_config.pop('wandb')
    # create the run name from the selected parameters (in case of hyperparameter search)
    args = copy.deepcopy(config)
    args.update(config['env_config'])
    args.update(wandb_mixin_config['original_config'])
    if args.get('lr_schedule'):
        if isinstance(args['lr_schedule'][1][1], (float, int)):
            s = args['lr_schedule']
        else:
            s = args['lr_schedule'][0]
        args['lr_start'] = s[0][1]
        args['lr_end'] = s[1][1]
    run_name = create_run_name(args)
    wandb_mixin_config['name'] = run_name

    # move the original config we got from argparse into the mixin config to get it stored to the cloud
    wandb_original_config = wandb_mixin_config.pop('original_config')
    wandb_original_config['wandb'] = wandb_mixin_config
    # pass ray_config as well so we can differentiate hyperparameter searches by the drawn values
    wandb_original_config['ray'] = nonwandb_config.copy()

    assert isinstance(self, parent_class)
    parent_class.__init__(self, nonwandb_config, env, logger_creator)

    if args.get('lr_schedule'):
        wandb_original_config['ray']['lr_end'] = s[1][1]
        wandb_original_config['ray']['lr_step'] = s[1][0]
    MyWandbTrainableMixin.__init__(self, config=wandb_original_config)

    with open(os.path.join(self.logdir, "ray_config.pkl"), 'wb') as f:
        pickle.dump(config, f)
        wandb.save(f.name)


def _save_ckpt(self, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-{}".format(self.iteration))
    pickle.dump(self.__getstate__(), open(checkpoint_path, "wb"))
    pickle.dump(self.get_state(), open(checkpoint_path + RAY_METADATA_ENDING, "wb"))

    wandb.save(checkpoint_path)
    wandb.save(checkpoint_path + RAY_METADATA_ENDING)
    return checkpoint_path


class WandbSACTrainer(MyWandbTrainableMixin, SACTrainerCustom):
    def __init__(self,
                 config: TrainerConfigDict = None,
                 env: str = None,
                 logger_creator: Callable[[], Logger] = None):
        _init(self, SACTrainerCustom, config, env, logger_creator)

    def log_result(self, result):
        return _log_results(self, result)

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        return _save_ckpt(self, checkpoint_dir)


def filter_wandb_files(files, name: str):
    return [f for f in files if f.name == name][0]


def restore_ckpt_files(resume_model_name: Optional[str], model_file: Optional[str], target_dir: str, run_path=None):
    # entity = os.environ['WANDB_ENTITY']
    # run_path = f"{entity}/{wandb_config.project_name}/{wandb_config.resume_id}"

    if resume_model_name:
        assert not model_file, model_file
        if run_path is None:
            run_path = wandb.run.path
        api = wandb.Api()
        api_run = api.run(run_path)
        print("Querying wandb stored files")
        files = list(api_run.files())
        ckpt_files = [f for f in files if f.name.startswith('checkpoint') and not f.name.endswith('trainableState')]

        resume_model_name = resume_model_name.replace('.zip', '')
        if resume_model_name == 'last_model':
            steps = [int(f.name.replace('checkpoint-', '')) for f in ckpt_files]
            resume_model_name = f'checkpoint-{max(steps)}'

        # model_file_path = wandb.restore(model_file.name, run_path=run_path, root=target_dir, replace=True)
        # metadata_file_path = wandb.restore(metadata_file.name, run_path=run_path, root=target_dir, replace=True)
        model_file = download_wandb(filter_wandb_files(files, resume_model_name), root=target_dir, replace=True)
        # model_file = wandb.restore(resume_model_name, root=target_dir, replace=True)
        # load and rename to what ray expects (storing under a different name to ensure I won't conflict with any hyperparamsearch functionality
        metadata_file = download_wandb(filter_wandb_files(files, resume_model_name + RAY_METADATA_ENDING), root=target_dir, replace=True)
        # metadata_file = wandb.restore(resume_model_name + RAY_METADATA_ENDING, root=target_dir, replace=True)

        ray_config_file = download_wandb(filter_wandb_files(files, 'ray_config.pkl'), root=target_dir, replace=True)
        # ray_config_file = wandb.restore('ray_config.pkl', root=target_dir, replace=True)
        print(f"Restored file {model_file.name}")
        model_file_name = model_file.name
        metadata_file_name = metadata_file.name
        ray_config_file_name = ray_config_file.name
    else:
        assert model_file, model_file
        model_file_name = model_file
        metadata_file_name = model_file + RAY_METADATA_ENDING
        ray_config_file_name = str(Path(model_file).parent / 'ray_config.pkl')

    new_metadata_filename = metadata_file_name.replace(RAY_METADATA_ENDING, '.tune_metadata')
    shutil.copy(metadata_file_name, new_metadata_filename)

    with open(new_metadata_filename, "rb") as f:
        metadata = pickle.load(f)
    if metadata.get('saved_as_dict', None) is None:
        metadata['saved_as_dict'] = False
        with open(new_metadata_filename, "wb") as f:
            pickle.dump(metadata, f)

    with open(ray_config_file_name, 'rb') as f:
        ray_config = pickle.load(f)

    return model_file_name, ray_config


def get_ckpt_weights(model_path, policy='default_policy'):
    model_path = Path(model_path)

    from ray import cloudpickle
    extra_data = cloudpickle.load(open(str(model_path), "rb"))
    objs = cloudpickle.loads(extra_data['worker'])
    # weights = objs['state']
    weights = objs['state'][policy]['weights']
    # global_timestep = objs['state']['default_policy']['global_timestep']
    return {policy: weights}
