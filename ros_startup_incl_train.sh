#!/usr/bin/env bash
if [ -z ${ROBOT+x} ]; then echo "no robot model provided. Set the ROBOT env var." && exit 1; fi
if [ -z ${CMD+x} ]; then echo "Set the CMD env var to the command to use (python scripts/main.py --option ...)." && exit 1; fi

# start ROS
cd "catkin_ws_${ROBOT}" || echo "Could not cd in catkin_ws. Already in there?"
source /opt/ros/melodic/setup.bash
source devel/setup.bash

# start python. NOTE: conda run buffers all stdout -.-: https://github.com/conda/conda/issues/9412
echo "Starting command ${CMD}"
cd src/modulation_rl && source activate base && ${CMD}
