#!/usr/bin/env bash

cmd="python -u scripts/main_ray.py --num_gpus 0.24 --num_workers 20 --num_cpus_per_worker 0.1 --num_envs_per_worker 1 --num_gpus_per_worker 0 --ray_verbosity 2 \
     --training_intensity 0.1 --load_best_defaults"

nseeds=1
startseed=40
envs=("pr2")
export GPU=0
export IMG="11.1-latest"
export TASK="train"
export SUFFIX=""
for ROBOT in "${envs[@]}"; do
  for ((i=$startseed;i<$startseed+$nseeds;i++)); do
    export CMD="${cmd} --seed ${i} --env ${ROBOT}"
    export ROBOT=${ROBOT}
    bash run_experiment.sh
  done
done
sleep 2 && dll


#############
# Evaluation
#############
tasks=("picknplace" "door" "drawer" "roomdoor" "spline" "bookstorepnp" "simpleobstacle" "dynobstacle" "picknplacedyn")
cmd="python -u scripts/evaluation_ray.py --evaluation_only --eval_execs sim gazebo"
# wandb run id
ids=("8397c_00000")

####### RESUME
export IMG="11.1-latest"
for id in "${ids[@]}"; do
  for task in "${tasks[@]}"; do
    export GPU=0
    export ROBOT=`python get_run_robotname.py ${id}`
    export CMD="${cmd} --env ${ROBOT} --resume_id ${id} --eval_tasks ${task}"
    export TASK=$task
    export SUFFIX=$id
    bash run_experiment.sh || return 0
  done
done
dll
