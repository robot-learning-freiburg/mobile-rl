import sys
import wandb

def read_run_robotname(wandb_run_id, entity='wazzup', project_name='mobile_rl'):
    api = wandb.Api()
    run = api.run(("%s/%s/%s" % (entity, project_name, wandb_run_id)))
    return run.config['env']

if __name__ == '__main__':
    robot = read_run_robotname(sys.argv[1])
    sys.stdout.write(robot)
    sys.exit(0)