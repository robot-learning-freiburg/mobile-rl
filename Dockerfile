ARG MY_CUDA_VERSION
FROM dhonerkamp/ros_torch:${MY_CUDA_VERSION}
ENV HOME=/root \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=melodic
ARG MY_CUDA_VERSION
WORKDIR $HOME

#####################
# CONDA DEPENDENCIES FROM ENV.YAML
#####################
# only want to get the environment.yml at this point, so as not to recreate everytime some code changes
COPY environment${MY_CUDA_VERSION}.yml src/modulation_rl/

RUN conda env update -n ${ENV_NAME} -f src/modulation_rl/environment${MY_CUDA_VERSION}.yml \
    && conda clean -afy


######################
## CREATE CATKIN WORKSPACE WITH PYTHON3
######################
RUN apt-get update \
    && apt-get install -y python-catkin-tools python3-dev python3-numpy gdb ros-melodic-costmap-2d ros-melodic-people-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $HOME/catkin_ws_pr2
RUN catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so

WORKDIR $HOME/catkin_ws_tiago
RUN catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so

WORKDIR $HOME/catkin_ws_hsr
RUN catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so

######################
## Tiago
######################
WORKDIR $HOME/catkin_ws_tiago
# install tiago sdk following http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/TiagoSimulation
# https://hub.docker.com/r/jacknlliu/tiago-ros/dockerfile
RUN wget -O tiago_public.rosinstall https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/kinetic-devel/tiago_public-melodic.rosinstall \
    && yes | rosinstall src /opt/ros/melodic tiago_public.rosinstall \
    && rosdep update --rosdistro $ROS_DISTRO \
    && apt-get update \
    && rosdep install -y -r -q --from-paths src --ignore-src --rosdistro $ROS_DISTRO --skip-keys="opencv2 opencv2-nonfree pal_laser_filters speed_limit_node sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup opencv3 hsrb_moveit_plugins hsrb_moveit_config hsrb_description hsrc_description" \
    # && apt-get install -y ros-melodic-base-local-planner ros-melodic-people-msgs ros-melodic-roslint ros-melodic-four-wheel-steering-controller ros-melodic-twist-mux \
    && rm -rf /var/lib/apt/lists/*

COPY modulation.rosinstall .
#NOTE: requires to have sensor-to-cloud as a submodule (git submodule add git@aisgit.informatik.uni-freiburg.de:rl/kuka-motion-control.git)
#COPY ros_pkgs/sensor-to-cloud src/sensor-to-cloud
RUN rm -rf src/navigation_layers && yes | rosinstall src /opt/ros/$ROS_DISTRO modulation.rosinstall
RUN catkin config --blacklist tiago_pcl_tutorial # combined_robot_hw_tests # force_torque_sensor_controller mode_state_controller
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build || exit"

# update the moveit configs with the global joint
COPY gazebo_world/tiago/robot/modified_tiago_pal-gripper.srdf src/tiago_moveit_config/config/srdf/tiago_pal-gripper.srdf
COPY gazebo_world/tiago/robot/modified_gripper.urdf.xacro src/pal_gripper/pal_gripper_description/urdf/gripper.urdf.xacro
COPY gazebo_world/tiago/robot/modified_wsg_gripper.urdf.xacro src/pal_wsg_gripper/pal_wsg_gripper_description/urdf/gripper.urdf.xacro

######################
## HSR
######################
WORKDIR $HOME/catkin_ws_hsr

# NOTE: the HSR classes can be built without any additional dependencies. But running the robot requires access to the proprietary HSR packages.
# If you have access, please install the HSR dependencies as outlined in the official manual

COPY modulation.rosinstall .
#NOTE: requires to have sensor-to-cloud as a submodule (git submodule add git@aisgit.informatik.uni-freiburg.de:rl/kuka-motion-control.git)
#COPY ros_pkgs/sensor-to-cloud src/sensor-to-cloud

RUN yes | rosinstall src /opt/ros/$ROS_DISTRO modulation.rosinstall
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build || exit"


######################
## PR2
######################
WORKDIR $HOME/catkin_ws_pr2
RUN apt-get update && apt-get install -y \
    ros-melodic-pr2-simulator \
    ros-melodic-moveit-pr2 \
    ros-melodic-pr2-common \
    ros-melodic-pr2-navigation \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --single-branch --depth 1 --branch melodic-devel https://github.com/PR2/pr2_mechanism.git src/pr2_mechanism

COPY modulation.rosinstall .
#NOTE: requires to have sensor-to-cloud as a submodule (git submodule add git@aisgit.informatik.uni-freiburg.de:rl/kuka-motion-control.git)
#COPY ros_pkgs/sensor-to-cloud src/sensor-to-cloud
RUN yes | rosinstall src /opt/ros/$ROS_DISTRO modulation.rosinstall
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build || exit"

#####################
# COPY FILES AND BUILD OUR ROS PACKAGE -> don't use conda python, but the original!
#####################
WORKDIR $HOME/catkin_ws_hsr
# build our package: copy only files required for compilation use caching whenever possible
COPY include/ src/modulation_rl/include/
COPY src/ src/modulation_rl/src/
COPY CMakeLists.txt package.xml src/modulation_rl/
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build modulation_rl || exit"

WORKDIR $HOME/catkin_ws_tiago
# build our package: copy only files required for compilation use caching whenever possible
COPY include/ src/modulation_rl/include/
COPY src/ src/modulation_rl/src/
COPY CMakeLists.txt package.xml src/modulation_rl/
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build modulation_rl || exit"

WORKDIR $HOME/catkin_ws_pr2
# build our package: copy only files required for compilation use caching whenever possible
COPY include/ src/modulation_rl/include/
COPY src/ src/modulation_rl/src/
COPY CMakeLists.txt package.xml src/modulation_rl/
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build modulation_rl || exit"

######################
## COPY FILES
######################
WORKDIR $HOME
# copy our object models into gazebo
COPY gazebo_models/ $HOME/.gazebo/models/
# copy our launch files and configs
COPY gazebo_world catkin_ws_hsr/src/modulation_rl/gazebo_world/
COPY gazebo_world catkin_ws_tiago/src/modulation_rl/gazebo_world/
COPY gazebo_world catkin_ws_pr2/src/modulation_rl/gazebo_world/

# gmm models
COPY GMM_models/ catkin_ws_hsr/src/modulation_rl/GMM_models
COPY GMM_models/ catkin_ws_tiago/src/modulation_rl/GMM_models
COPY GMM_models/ catkin_ws_pr2/src/modulation_rl/GMM_models/

# launch helper
COPY ros_startup_incl_train.sh ./

# git so commit gets logged on wandb
COPY rviz/rviz_config.rviz catkin_ws_hsr/src/modulation_rl/rviz/rviz_config.rviz
COPY rviz/rviz_config.rviz catkin_ws_tiago/src/modulation_rl/rviz/rviz_config.rviz
COPY rviz/rviz_config.rviz catkin_ws_pr2/src/modulation_rl/rviz/rviz_config.rviz
COPY .git/ catkin_ws_hsr/src/modulation_rl/.git/
COPY .git/ catkin_ws_tiago/src/modulation_rl/.git/
COPY .git/ catkin_ws_pr2/src/modulation_rl/.git/

# prevent potential erros with wandb
#RUN mv /opt/ros/melodic/lib/python2.7/dist-packages/cv2.so /opt/ros/melodic/lib/python2.7/dist-packages/cv2_renamed.so

######################
## RUN TRAINING
######################
# ensure we take python2 to run the ros startup stuff
# our task world includes assets from the publicly available pal gazebo worlds
ENV PATH=/usr/bin/:$PATH \
    PYTHONPATH=/usr/bin/:$PYTHONPATH \
    PYTHONUNBUFFERED=1 \
    ROSCONSOLE_FORMAT='[${severity} ${node}] [${time}]: ${message}' \
    GAZEBO_MODEL_PATH=${HOME}/catkin_ws_tiago/src/modulation_rl/gazebo_models:${HOME}/catkin_ws_tiago/src/pal_gazebo_worlds/models:${GAZEBO_MODEL_PATH} \
    GAZEBO_RESOURCE_PATH=${HOME}/catkin_ws_tiago/src/modulation_rl/gazebo_world/worlds:${GAZEBO_RESOURCE_PATH} \
    NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
    NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
COPY gazebo_world/rosconsole.config /opt/ros/melodic/share/ros/config/rosconsole.config

# copy all remaining files
COPY scripts/ catkin_ws_hsr/src/modulation_rl/scripts
COPY scripts/ catkin_ws_tiago/src/modulation_rl/scripts
COPY scripts/ catkin_ws_pr2/src/modulation_rl/scripts

CMD bash ros_startup_incl_train.sh "python scripts/main.py --load_best_defaults"
