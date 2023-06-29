# N2M2:  Learning Navigation for Arbitrary Mobile Manipulation Motions in Unseen and Dynamic Environments

Repository providing the source code for the paper "N2M2:  Learning Navigation for Arbitrary Mobile Manipulation Motions in Unseen and Dynamic Environments", see the [project website](http://mobile-rl.cs.uni-freiburg.de/).  
Please cite the paper as follows:

    @article{honerkamp2021learning,
        title={N$^2$M$^2$: Learning Navigation for Arbitrary Mobile Manipulation Motions in Unseen and Dynamic Environments},
        journal={IEEE Transactions on Robotics}, 
        author={Daniel Honerkamp and Tim Welschehold and Abhinav Valada},
        year={2023},
        doi={10.1109/TRO.2023.3284346}
    }

Note: the current version represents the code used for the paper, except for the proprietary components outlined below. We may release a cleaner, more modular version of this code in the future.

## Proprietary Components
### HSR
The HSR environment relies on packages that are part of the proprietory HSR simulator. If you have an HSR account with Toyota,
please follow these steps to use the environment. Without these, you will still be able to build the HSR classes of this project,
but may not be able to run them.

### Range sensor integration
In Gazebo we equip the robots with three range sensors in the back to provide limited vision in this area. The integration of these observations into the costmap relies on the
on a proprietary node from the TIAGo robot. We are not able to provide this package. But all experiments in Gazebo can nonetheless be run without incorporation of these range sensors.
But note that quantitative results in Gazebo may differ from the paper as a result.


## Docker
Easiest way to get started is to pull the docker image 
	
	docker pull dhonerkamp/mobile_rl:11.1-latest

The current implementation relies on the Weights And Biases library for logging.
So create a python3 environment with wandb (free account required) and run `wandb login` to login to your account.
Alternatively first run `wandb disabled` in your shell to run it without an account and without logging any results 
(evaluations still get printed to stdout).

The following commands also assume that you have docker installed, and to use a CUDA-capable GPUs also the nvidia-docker driver. To only use the CPU remove the `--gpus` flag. 

The commands to train and evaluate the models in the paper can be found in `docker_run.sh`.

To run the container interactively, use the following commands: 

    docker run -it --gpus all --network host --rm dhonerkamp/mobile_rl:11.1-latest bash
    cd catkin_ws_[pr2/hsr/tiago]
    source devel/setup.bash
    conda activate

then proceed to the "Run" section below.

To visualize the runs, start rviz _outside_ of the docker container with the following command (the `--network host` flag means that the ROS messages are available on the host machine):
```
rviz -d src/modulation_rl/rviz/rviz_config[_tiago_hsr].config
```

Unfortunately running the GUIs inside docker is more evolved. So for visualizing Gazebo, it is often more convenient to follow the local installation section below.
Alternatively, instead of the docker command, it is possible to use [rocker](https://github.com/osrf/rocker) as follows:
```
pip install rocker
rocker --nvidia --x11 --pull dhonerkamp/mobile_rl:11.1-latest bash
```
followed by the commands in the Run section below.

## Evaluating on your own task
Tasks are implemented as environment wrappers in `scripts/modulation/envs/tasks.py`.

To construct a new task add a new wrapper to `tasks.py`, add it to ALL_TASKS in `scripts/modulation/envs/__init__.py`.
It can the be used by passing the task name to the `--tasks` flag.

## Local installation
For development or qualitative inspection of the behaviours in rviz or gazebo it can be easier to install the setup locally.
The following illustrates the main steps to do this for the PR2. 
As the different robots come with different ROS dependencies, please use the `Dockerfile` as the full guide to install them.

The repository consists of two main parts: a small C++ component connected to python through bindings and the RL agents written in python3.

As not all ROS packages work with python3, the setup relies on running the robot-specific packages in a python2 environment
and our package in a python3 environment.
The environment was tested for Ubunto 18.04 and ROS melodic.

### Install
Install the appropriate version for your system (full install recommended): http://wiki.ros.org/ROS/Installation

Install the corresponding catkin package for python bindings
        
    sudo apt install ros-[version]-pybind11-catkin
        
Install moveit and the pr2
    
    sudo apt-get install ros-[version]-pr2-simulator
    sudo apt-get install ros-[version]-moveit
    sudo apt-get install ros-[version]-moveit-pr2
    ros-melodic-pr2-common
    ros-melodic-pr2-navigation
   
Create a catkin workspace (ideally a separate one for each robot)

    mkdir ~/catkin_ws
    cd catkin_ws

Install further ros packages:

    yes | rosinstall src /opt/ros/melodic modulation.rosinstall

Fork the repo and clone into `./src`
    
    cd src
    git clone [url] src/modulation_rl

Create a python environment. We recommend using conda, which requires to first install Anaconda or Miniconda. Then do

    conda env create -f src/modulation_rl/environment.yml
    conda activate modulation_rl

Configure the workspace it to use your environment's python3 (adjust path according to your version)

    catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so
    
Build the workspace
    
    catkin build
    
Each new build of the ROS / C++package requires a
    
    source devel/setup.bash
    
To be able to visualise install rviz

    http://wiki.ros.org/rviz/UserGuide
    
For more details and how to install the ROS requirements for the other robots please check the `Dockerfile`. It also contains further details to packages you might need to build from source.


### Run
- If you have a weights and biases account, log into the account. Otherwise first run `wandb offline` in your shell, which will turn off logging.

- Training (set workers etc. as needed, this is the command used in the paper):

        python src/modulation_rl/scripts/main_ray.py --load_best_defaults --num_gpus 0.24 --num_workers 20 
        --num_cpus_per_worker 0.1 --num_envs_per_worker 1 --num_gpus_per_worker 0 --ray_verbosity 2
        --training_intensity 0.1 --load_best_defaults --vis_env

- Evaluation of a pretrained checkpoint (pick one of [pr2/hsr/tiago]):

        python src/modulation_rl/scripts/evaluation_ray.py --evaluation_only --eval_execs sim gazebo --env [pr2/hsr/tiago] --model_file=[pr2/hsr/tiago]/checkpoint-1000 --eval_tasks picknplace --vis_env 

- Evaluation of a wandb logged run (adjust [pr2/hsr/tiago] and ${wandb_run_id}):

        python src/modulation_rl/scripts/evaluation_ray.py --evaluation_only --eval_execs sim gazebo --env [pr2/hsr/tiago] --resume_id ${wandb_run_id} --eval_tasks picknplace --vis_env

5. [Only to visualise] start rviz:

        rviz -d src/modulation_rl/rviz/rviz_config[_tiago_hsr].config

Useful command line flags:
- `--load_best_defaults`: use the robot specific best default arguments
- `--vis_env`: publish the visualisation markers to display in rviz
- `--task`: what task to evaluate on
- `-d`: dry-run -> do not upload to wandb

## Troubleshooting
- Library conflicts: error message either around `cv2` or `libgcc_s.so.1 must be installed for pthread_cancel to work`:
    Solution: rename cv2 installed by ROS: 
    https://stackoverflow.com/questions/48039563/import-error-ros-python3-opencv

