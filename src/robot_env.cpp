#include <geometry_msgs/Twist.h>
#include "modulation_rl/robot_env.h"
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <moveit_msgs/RobotState.h>
//#include <moveit/move_group_interface/move_group_interface.h>
#include <pybind11/pybind11.h>
#include <std_srvs/Empty.h>
#include <boost/bind.hpp>
#include <utility>
namespace py = pybind11;

using namespace std;

RobotEnv::RobotEnv(uint32_t seed,
                   const string &node_handle_name,
                   bool init_controllers,
                   const string &world_type,
                   RoboConf robot_conf,
                   double bioik_center_joints_weight,
                   double bioik_avoid_joint_limits_weight,
                   double bioik_regularization_weight,
                   string bioik_regularization_type) :
        // ros::init will simply ignore new calls to it if creating multiple envs I believe
        ROSCommonNode2(0, NULL, "robot_env"),
        nh_{new ros::NodeHandle(node_handle_name)},
        robot_config_{std::move(robot_conf)},
        init_controllers_{init_controllers},
        bioik_center_joints_weight_{bioik_center_joints_weight},
        bioik_avoid_joint_limits_weight_{bioik_avoid_joint_limits_weight},
        bioik_regularization_weight_{bioik_regularization_weight},
        bioik_regularization_type_{std::move(bioik_regularization_type)}{

    setRng(seed);
    base_cmd_pub_ = nh_->advertise<geometry_msgs::Twist>(robot_config_.base_cmd_topic, 1);
    robstate_visualizer_ = nh_->advertise<moveit_msgs::DisplayRobotState>("robot_state_visualizer", 50);
    gripper_visualizer_ = nh_->advertise<visualization_msgs::Marker>("gripper_goal_visualizer", 500);
    traj_visualizer_ = nh_->advertise<moveit_msgs::DisplayTrajectory>("traj_visualizer", 1);

    // Load Robot config from moveit movegroup (must be running)
    robot_model_loader::RobotModelLoaderPtr robot_model_loader;
    robot_model_loader.reset(new robot_model_loader::RobotModelLoader("robot_description"));

    robot_model::RobotModelPtr kinematic_model = robot_model_loader->getModel();
    kinematic_state_.reset(new robot_state::RobotState(kinematic_model));
    kinematic_state_->setToDefaultValues();
    if (!kinematic_model->hasJointModelGroup("whole_body")){
      for (auto name : kinematic_model->getJointModelGroupNames()){
        py::print(name);
      }
      throw runtime_error("Doesnt have whole_body group");
    }
    trajectory_.reset(new robot_trajectory::RobotTrajectory(kinematic_model, "whole_body"));

    joint_model_group_ = kinematic_model->getJointModelGroup(robot_config_.joint_model_group_name);
    ik_joint_model_group_ = kinematic_model->getJointModelGroup(robot_config_.ik_joint_model_group_name);
    string param_name = "/robot_description_kinematics/" + robot_config_.joint_model_group_name + "/kinematics_solver";
    ros::param::get(param_name, ik_solver_name_);

    for (const auto& joint_name : ik_joint_model_group_->getVariableNames()){
        auto& bounds = kinematic_model->getVariableBounds(joint_name);
        double max_vel = std::max(bounds.max_velocity_, 0.0);
        bioik_max_velocities_.push_back(max_vel);
        cout<<joint_name << ": " << max_vel << endl;
    }

    base_link_name_ = "base_footprint";

    planning_scene_.reset(new planning_scene::PlanningScene(kinematic_model));
    ROS_INFO("Planning frame: %s", planning_scene_->getPlanningFrame().c_str());
    updatePlanningScene();

    // add a floor to the planning scene, so we don't draw gripper poses that hit the floor
    moveit_msgs::CollisionObject floor_object;
    floor_object.header.frame_id = "base_footprint";

    shape_msgs::Plane plane;
    plane.coef = {{0, 0, 1, 0}};
    floor_object.planes.push_back(plane);

    geometry_msgs::Pose pose;
    pose.orientation.w = 1.0;
    floor_object.plane_poses.push_back(pose);

    moveit_msgs::PlanningScene planning_scene_msg;
    planning_scene_msg.world.collision_objects.push_back(floor_object);
    planning_scene_msg.is_diff = true;
    planning_scene_->setPlanningSceneDiffMsg(planning_scene_msg);

    // moveit initialises all joints to 0. If we use gazebo or similar, first update all values to match the simulator
    if (world_type == "gazebo") {
        kinematic_state_.reset(new robot_state::RobotState(planning_scene_->getCurrentState()));
    } else {
        // in the analytical env set non-default joints to the hardcoded values that we know we will use in the simulator later
        for (auto const &x : robot_config_.initial_joint_values) {
            kinematic_state_->setJointPositions(x.first, &x.second);
        }
    }

    setWorld(world_type);

    if (init_controllers_) {
        // https://readthedocs.org/projects/moveit/downloads/pdf/latest/
        // https://ros-planning.github.io/moveit_tutorials/doc/planning_scene_monitor/planning_scene_monitor_tutorial.html
        spinner_ = new ros::AsyncSpinner(2);
        spinner_->start();

         planning_scene_monitor_.reset(new planning_scene_monitor::PlanningSceneMonitor(robot_model_loader));
         // planning_scene_monitor_->startSceneMonitor("/planning_scene");
         planning_scene_monitor_->startStateMonitor("joint_states");
    }
}

void RobotEnv::updatePlanningScene(){
    moveit_msgs::GetPlanningScene scene_srv1;
    scene_srv1.request.components.components = 2;  // moveit_msgs::PlanningSceneComponents::ROBOT_STATE;
    ros::ServiceClient client_get_scene = nh_->serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
    if (!client_get_scene.call(scene_srv1)) {
        ROS_WARN("Failed to call service /get_planning_scene");
    }
    planning_scene_->setPlanningSceneDiffMsg(scene_srv1.response.scene);
}

RobotObs RobotEnv::reset(const map<string, double> &initial_joint_values, bool do_close_gripper) {
    ROS_INFO_COND(!world_->isAnalytical(), "Reseting environment");

    setJointValuesFromDict(initial_joint_values, true);

    if (init_controllers_) {
        do_close_gripper ? closeGripper(0.0, false) : openGripper(0.08, false);
    }
    // Clear the visualizations
    visualization_msgs::Marker marker;
    marker.header.frame_id = robot_config_.frame_id;
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::DELETEALL;
    gripper_visualizer_.publish(marker);

    world_->awaitTfs();

    trajectory_->clear();
    display_msg_.trajectory.clear();
    robot_state::robotStateToRobotStateMsg(*kinematic_state_, display_msg_.trajectory_start);
    publishRobotState();

    return getRobotObs(false, false);
}

void RobotEnv::setWorldJointPython(vector<double> &base_tf_list) {
    tf::Transform base_tf = myutils::listToTf(base_tf_list);
    setWorldJoint(base_tf);
    world_->setModelState(robot_config_.name, base_tf, base_cmd_pub_);
}

void RobotEnv::setWorldJoint(tf::Transform &transform){
    kinematic_state_->setVariablePosition("world_joint/x", transform.getOrigin().x());
    kinematic_state_->setVariablePosition("world_joint/y", transform.getOrigin().y());
    kinematic_state_->setVariablePosition("world_joint/theta", myutils::qToRpy(transform.getRotation()).z());
}

bool RobotEnv::checkSelfCollision(){
    collision_detection::CollisionRequest collision_request;
    collision_request.group_name = robot_config_.joint_model_group_name;
    collision_detection::CollisionResult collision_result;

    planning_scene_->getCurrentStateNonConst().update();
    robot_state::RobotState state_copy(*kinematic_state_);
    planning_scene_->checkCollision(collision_request, collision_result, state_copy);
    // collision_result.clear();
    return collision_result.collision;
}

RobotObs RobotEnv::step(const vector<double>& base_translation_relative_list,
                        const double base_rotation_relative,
                        const vector<double> &desired_gripper_tf_list,
                        const RobotObs &prev_robot_obs,
                        const double dt_exec,
                        const bool execute_cmd,
                        const bool learn_torso,
                        const double delta_torso,
                        const string &execute_style,
                        const string &perception_style,
                        const vector<double> &joint_values_action) {
    // base
    tf::Vector3 base_translation_relative = myutils::listToVector3(base_translation_relative_list);

    tf::Transform base_tf = getGlobalLinkTransform(base_link_name_);
    tf::Transform desired_base_tf = calcDesiredBaseTf(base_tf, base_translation_relative, base_rotation_relative, dt_exec);
    // NOTE: do this irrespective of the world, so findIk() searches for a solution relative to the next desired base_tf
    setWorldJoint(desired_base_tf);

    // arm
    // correct for potentially different ee orientation (e.g. HSR)
    tf::Transform desired_gripper_tf_world_orientation = myutils::listToTf(desired_gripper_tf_list);
    tf::Transform desired_gripper_tf = myutils::tipToGripperGoal(desired_gripper_tf_world_orientation,
                                                               tf::Vector3(0., 0., 0.),
                                                               robot_config_.gripper_to_base_rot_offset);
    // tf::Transform desired_gripper_tf_rel = desired_base_tf.inverse() * desired_gripper_tf;
    // if (world_->is_analytical()){
    //    desired_gripper_tf_rel = desired_base_tf.inverse() * desired_gripper_tf;
    //} else {
    //    desired_gripper_tf_rel = robot_state_.base_tf.inverse() * desired_gripper_tf;
    //}

    // learned torso
    if (learn_torso){
        double new_torso_position = kinematic_state_->getVariablePosition(robot_config_.torso_joint_name) + dt_exec * delta_torso;
        kinematic_state_->setVariablePosition(robot_config_.torso_joint_name, new_torso_position);
        kinematic_state_->enforcePositionBounds(kinematic_state_->getJointModel(robot_config_.torso_joint_name));
    }

    bool in_selfcollision = false;
    bool found_ik = false;
    if (!joint_values_action.empty()){
        if (!learn_torso){
            throw runtime_error("use only with learn_torso, so the joint_model_group_name only refers to the arm.");
        }
        vector<double> joint_values_pre;
        kinematic_state_->copyJointGroupPositions(ik_joint_model_group_, joint_values_pre);
        if (joint_values_action.size() != joint_values_pre.size()){
            throw runtime_error("Wrong size of joint_values_action. Is " + std::to_string(joint_values_action.size()) + " but should be " + std::to_string(joint_values_pre.size()));
        }

        kinematic_state_->setJointGroupPositions(robot_config_.ik_joint_model_group_name, joint_values_action);

        in_selfcollision = checkSelfCollision();
        if (in_selfcollision) {
            kinematic_state_->setJointGroupPositions(robot_config_.ik_joint_model_group_name, joint_values_pre);
        }
        found_ik = !in_selfcollision;
    } else {
        found_ik = findIk(desired_gripper_tf, dt_exec, in_selfcollision);
    }

    if (!world_->isAnalytical() && execute_cmd) {
        sendArmCommandCurrent(dt_exec);
        if ((execute_style == "direct") || (robot_config_.name == "tiago")){
            sendBaseCommand(base_translation_relative_list, base_rotation_relative);
        } else if (execute_style == "track"){
            // NOTE: Seems to work more stable if we calculate this for a fixed timestep, instead of using the desired_base_tf calculated above -> could make a little more efficient
            tf::Transform des = calcDesiredBaseTf(base_tf, base_translation_relative, base_rotation_relative, 1.0);
            // publishMarker(myutils::tfToList(des), 0, "des", "orange", 1.0, "arrow", {0.1, 0.025, 0.025}, "map");

            tf::Vector3 trans;
            double rot;
            calcDesiredBaseCommand(world_->getBaseTransformWorld(), des, 1.0, trans, rot);
            sendBaseCommand(myutils::vector3ToList(trans), rot);
        } else {
            throw runtime_error("Unknown execute_style");
        }

        // update state to what we actually achieve. Without execution we'll always be at the next base transform
        if (perception_style == "base") {
            // v0: update base, which will also change the end-effector tf that we return in robot_obs, but do so with the desired joint values
            tf::Transform latest_base = world_->getBaseTransformWorld();
            setWorldJoint(latest_base);
        } else if (perception_style == "all"){
            // v1: update everything updatePlanningScene() is very slow -> use state monitor
            planning_scene_monitor_->getStateMonitor()->setToCurrentState(*kinematic_state_);
            // TODO: this seems to return the base in the odom frame, not the map frame that we (sometimes) use in real world
            tf::Transform latest_base = world_->getBaseTransformWorld();
            setWorldJoint(latest_base);
            // myutils::printT(latest_base, "latest_base1");
            // myutils::printT(getGlobalLinkTransform(base_link_name_), "latest_base2");
        } else if (perception_style == "none"){
            // do nothing, pretend we achieve desired
        } else {
            throw runtime_error("Unknown perception_style");
        }
        // v2: update base_tf, but do not update the gripper_tf -> best to change in getRobotObs(): update base after getting the gripper_tf
    }

    robot_state::RobotState state_copy(*kinematic_state_);
    trajectory_->addSuffixWayPoint(state_copy, dt_exec);

    return getRobotObsWithVelocities(!found_ik, prev_robot_obs, in_selfcollision);
}

void RobotEnv::sendBaseCommand(const vector<double> &base_translation_relative_list, const double base_rotation_relative){
    tf::Vector3 base_translation_relative = myutils::listToVector3(base_translation_relative_list);
    // base command, in m/s and rad/s
    geometry_msgs::Twist base_cmd_rel;
    base_cmd_rel.linear.x = base_translation_relative.x();
    base_cmd_rel.linear.y = base_translation_relative.y();
    base_cmd_rel.linear.z = base_translation_relative.z();
    base_cmd_rel.angular.z = base_rotation_relative;
    base_cmd_pub_.publish(base_cmd_rel);
}

void RobotEnv::sendArmCommandCurrent(double exec_duration) {
    vector<double> timestamps{exec_duration};

    vector<double> joint_values;
    kinematic_state_->copyJointGroupPositions(joint_model_group_, joint_values);
    vector<vector<double>> joint_values_vec;
    joint_values_vec.push_back(joint_values);

    return sendArmCommands(joint_values_vec, timestamps);
}

// directly sending the desired joint values can fail, so actually plan & execute for the resets
//bool RobotEnv::resetArm(const double planning_time){
//    moveit::planning_interface::MoveGroupInterface move_group(robot_config_.joint_model_group_name);
//    move_group.setGoalJointTolerance(0.05);
//    move_group.setPlanningTime(planning_time);
//
//    // robot_state::RobotState start_state(*move_group.getCurrentState());
//    // move_group.setStartState(start_state);
//
//    vector<double> joint_values;
//    kinematic_state_->copyJointGroupPositions(joint_model_group_, joint_values);
//    move_group.setJointValueTarget(joint_values);
//
//    moveit::planning_interface::MoveItErrorCode error_code = move_group.move();
//    return error_code.val == error_code.SUCCESS;
//}

bool RobotEnv::checkArmValues(){
    // check if the reset of the arm and torso succeeded: compare desired joint values to world joint values
    // NOTE: PR2 can have phase differences in the joint values for which the action client doesn't get that it actually succeeded
    vector<double> joint_values_world, joint_values_desired;
    kinematic_state_->copyJointGroupPositions(joint_model_group_, joint_values_desired);

    joint_values_world = getJointValuesWorld(robot_config_.joint_model_group_name);

    vector<string> joint_names = joint_model_group_->getVariableNames();
    for (int i = 0; i < joint_values_desired.size(); i++) {
        double phase_diff = abs(myutils::trueModulo(joint_values_desired[i], 2.0 * M_PI) - myutils::trueModulo(joint_values_world[i], 2.0 * M_PI));
        if (phase_diff >= 0.03){
             py::print(joint_names[i], "phase diff:", phase_diff, "jv-desired:", joint_values_desired[i], "jv-world:", joint_values_world[i], "fmod1:", myutils::trueModulo(joint_values_desired[i], 2.0 * M_PI), "fmod2:", myutils::trueModulo(joint_values_world[i], 2.0 * M_PI));
            return false;
        }
    }
    return true;
}


void RobotEnv::setJointValuesFromDict(const map<string, double> &joint_values, bool set_in_world) {
    kinematic_state_->setVariablePositions(joint_values);
    if (!set_in_world){
        return;
    }

    // This might sometimes fail. So continue sampling a few random poses
    if (!world_->isAnalytical()) {
        // Arm: if not the analytical env, we actually execute it to reset.
        bool success = false;
        int trials = 0, max_trials = 10;
        while ((!success) && (trials < max_trials)) {
            // arm: use controllers
            sendArmCommandCurrent(4.0);
            getArmSuccess();
            success = checkArmValues();
            // success = resetArm(10);
            if (!success) {
                py::print("resetArm failed at iteration", trials);
            }
            trials++;
        }
        if (!success) {
            throw runtime_error("Could not set start pose after 50 trials!!!");
        }

        // Base
        tf::Transform base_tf = getGlobalLinkTransform(base_link_name_);
        if (world_->getName() == "world") {
            tf::Transform base_tf_world = world_->getBaseTransformWorld();
            if (!myutils::tfAlmostEqual(base_tf, base_tf_world)) {
                throw runtime_error("Initial base tf not matching");
            }
        } else {
            world_->setModelState(robot_config_.name, base_tf, base_cmd_pub_);
        }
    }
}

tf::Transform RobotEnv::getGlobalLinkTransform(const string &link_name){
    const Eigen::Affine3d &pose = kinematic_state_->getGlobalLinkTransform(link_name);
    tf::Transform transform;
    tf::transformEigenToTF(pose, transform);
    return transform;
}

void RobotEnv::setWorld(const string &world_type) {
    if (world_type == "gazebo") {
        world_ = new GazeboWorld(robot_config_.global_link_transform, base_link_name_);
    } else if (world_type == "world") {
        world_ = new RealWorld(robot_config_.global_link_transform, base_link_name_, 0.2, 1.0);
    } else if (world_type == "sim") {
        world_ = new SimWorld(robot_config_.global_link_transform, base_link_name_);
    } else {
        throw runtime_error("Unknown real_execution value");
    }
    if ((!world_->isAnalytical()) && !init_controllers_) {
        throw runtime_error("must have initialised controllers to use real_execution_");
    }
}

bool RobotEnv::findIkDefault(const tf::Transform &desired_gripper_tf_world, const robot_state::GroupStateValidityCallbackFn &callback_fn) {
    Eigen::Isometry3d state;
    tf::poseTFToEigen(desired_gripper_tf_world, state);
    const Eigen::Isometry3d &desired_gripper_eigen = state;

    // kinematics::KinematicsQueryOptions ik_options;
    // ik_options.return_approximate_solution = true;
    bool success = kinematic_state_->setFromIK(ik_joint_model_group_,
                                               desired_gripper_eigen,
                                               robot_config_.kinematics_solver_timeout,
                                               callback_fn);

    return success;
}

bool RobotEnv::findIkBioIk(const tf::Transform &desired_gripper_tf_world, const double dt, const robot_state::GroupStateValidityCallbackFn &callback_fn) {
    // NOTE: cannot pack this into the valididity callback fn as it doesn't have access to the target pose
    // double dist_desired_to_goal = (desired_gripper_tf_world.getOrigin() - gripper_goal_wrist_.getOrigin()).length();

    bio_ik::BioIKKinematicsQueryOptions ik_options;
    getBioIkOptions(ik_options, dt);

    const tf2::Vector3 origin(desired_gripper_tf_world.getOrigin().x(), desired_gripper_tf_world.getOrigin().y(),
                              desired_gripper_tf_world.getOrigin().z());
    const tf2::Quaternion rot(desired_gripper_tf_world.getRotation().x(),
                              desired_gripper_tf_world.getRotation().y(),
                              desired_gripper_tf_world.getRotation().z(),
                              desired_gripper_tf_world.getRotation().w());
    auto *pose_goal = new bio_ik::PoseGoal(robot_config_.global_link_transform, origin, rot);
    ik_options.goals.emplace_back(pose_goal);
    bool success = kinematic_state_->setFromIK(ik_joint_model_group_,
                                               Eigen::Isometry3d(),
                                               robot_config_.kinematics_solver_timeout,
                                               callback_fn,
                                               ik_options);
    return success;
}


bool RobotEnv::findIk(const tf::Transform &desired_gripper_tf_world, const double dt, bool &in_selfcollision) {
    vector<double> joint_values_pre;
    kinematic_state_->copyJointGroupPositions(ik_joint_model_group_, joint_values_pre);

    robot_state::GroupStateValidityCallbackFn callback_fn = boost::bind(&validity_fun::validityCallbackFnSelfCollision, planning_scene_, &in_selfcollision, kinematic_state_, _2, _3);

    bool success;
    if (ik_solver_name_ == "bio_ik/BioIKKinematicsPlugin"){
        success = findIkBioIk(desired_gripper_tf_world, dt, callback_fn);
    } else {
        success = findIkDefault(desired_gripper_tf_world, callback_fn);
    }
    // kinematic_state_ -> enforceBounds(ik_joint_model_group_);
    // myutils::printArrayDouble(joint_values_pre, "joint_values_pre");

    // in case of a collision keep the current position
    // NOTE: I believe with return_approx_solution success will only ever be false if the validity callback function returns false
    if (in_selfcollision) {
        kinematic_state_->setJointGroupPositions(robot_config_.ik_joint_model_group_name, joint_values_pre);
    }
    return success;
}

RobotObs RobotEnv::getRobotObsWithVelocities(bool ik_fail, const RobotObs &prev_robot_obs, bool in_selfcollision){
    auto robot_obs = getRobotObs(ik_fail, in_selfcollision);
    robot_obs.setVelocities(prev_robot_obs);
    return robot_obs;
}

RobotObs RobotEnv::getRobotObs(bool ik_fail, bool in_selfcollision){
    tf::Transform base_tf = getGlobalLinkTransform(base_link_name_);
    tf::Transform gripper_tf = getGlobalLinkTransform(robot_config_.global_link_transform);
    tf::Transform gripper_tf_oriented = myutils::gripperToTipGoal(gripper_tf, tf::Vector3(0., 0., 0.), robot_config_.gripper_to_base_rot_offset);
    tf::Transform relative_gripper_tf_oriented = base_tf.inverse() * gripper_tf_oriented;

    tf::Transform gripper_tf_achieved_oriented;
    if (world_->isAnalytical()){
        gripper_tf_achieved_oriented = gripper_tf_oriented;
    } else {
        tf::Transform gripper_tf_achieved = world_->getGripperTransformWorld();
        gripper_tf_achieved_oriented = myutils::gripperToTipGoal(gripper_tf_achieved, tf::Vector3(0., 0., 0.), robot_config_.gripper_to_base_rot_offset);
    }

    vector<double> joint_values;
    kinematic_state_->copyJointGroupPositions(joint_model_group_, joint_values);

    vector<double> link_positions;
    for (const auto& joint : joint_model_group_->getLinkModelNames()){
        tf::Transform joint_pose = getGlobalLinkTransform(joint);
        tf::Transform joint_pose_relative = base_tf.inverse() * joint_pose;
        link_positions.push_back(joint_pose_relative.getOrigin().x());
        link_positions.push_back(joint_pose_relative.getOrigin().y());
        link_positions.push_back(joint_pose_relative.getOrigin().z());
    }

    return RobotObs{.base_tf = myutils::tfToList(base_tf),
                    .gripper_tf = myutils::tfToList(gripper_tf_oriented),
                    .relative_gripper_tf = myutils::tfToList(relative_gripper_tf_oriented),
                    .joint_values = joint_values,
                    .ik_fail = ik_fail,
                    .in_selfcollision = in_selfcollision,
                    .link_positions = link_positions,
                    .base_velocity_world = vector<double>{0., 0., 0.},
                    .base_rotation_velocity_world = 0.,
                    .gripper_velocities_world = vector<double>{0., 0., 0., 0., 0., 0., 0.},
                    .gripper_tf_achieved = myutils::tfToList(gripper_tf_achieved_oriented)};
}

vector<double> RobotEnv::drawJointValues(const double z_min, const double z_max) {
        bool invalid = true;
        int i = 0;
        while (invalid) {
            kinematic_state_->setToRandomPositions(joint_model_group_, *rng_);

            invalid = checkSelfCollision();
            ROS_INFO_COND(invalid, "set_start_pose: drawn pose in self-collision, trying again");

            if (!invalid && ((z_min != 0.0) || (z_max != 0.0))) {
                if (z_min >= z_max){
                    throw runtime_error("z_min must be smaller than z_max");
                }
                tf::Transform temp_tf = getGlobalLinkTransform(robot_config_.global_link_transform);
                invalid |= (temp_tf.getOrigin().z() < z_min);
                invalid |= (temp_tf.getOrigin().z() > z_max);
                ROS_INFO_COND(invalid, "EE outside of restricted ws, sampling again.");
            }
            if (i > 1000) {
                throw runtime_error("drawJointValues failed to draw values");
            }
            i ++;
    }
    vector<double> joint_values;
    kinematic_state_->copyJointGroupPositions(joint_model_group_, joint_values);

    return joint_values;
}

vector<double> RobotEnv::getJointValues(const string& joint_group){
    const robot_state::JointModelGroup *joint_model_group = kinematic_state_->getJointModelGroup(joint_group);
    vector<string> joint_names = joint_model_group->getVariableNames();

    vector<double> values;
    for (const auto &name : joint_names) {
        values.push_back(kinematic_state_->getJointPositions(name)[0]);
    }
    return values;
}

vector<double> RobotEnv::getJointValuesWorld(const string& joint_group){
    if (isAnalyticalWorld()){
        throw runtime_error("Not defined for analytical world");
    }
    robot_state::RobotState state_copy(*kinematic_state_);
    planning_scene_monitor_->getStateMonitor()->setToCurrentState(state_copy);

    const robot_state::JointModelGroup *joint_model_group = state_copy.getJointModelGroup(joint_group);
    vector<string> joint_names = joint_model_group->getVariableNames();

    vector<double> values;
    for (const auto &name : joint_names) {
        values.push_back(state_copy.getJointPositions(name)[0]);
    }
    return values;
}

vector<string> RobotEnv::getJointNames(const string& joint_group){
    auto joint_model_group = kinematic_state_->getJointModelGroup(joint_group);
    return joint_model_group->getVariableNames();
}

vector<double> RobotEnv::getJointMinima(const string& joint_group){
    const vector<string> &jn = kinematic_state_->getJointModelGroup(joint_group)->getVariableNames();

    vector<double> values;
    for (const auto& name : jn){
        auto bounds = kinematic_state_->getJointModel(name)->getVariableBounds(name);
        values.push_back(bounds.min_position_);
    }
    return values;
}

vector<double> RobotEnv::getJointMaxima(const string& joint_group){
    const vector<string> &jn = kinematic_state_->getJointModelGroup(joint_group)->getVariableNames();

    vector<double> values;
    for (const auto& name : jn){
        auto bounds = kinematic_state_->getJointModel(name)->getVariableBounds(name);
        values.push_back(bounds.max_position_);
    }
    return values;
}

void RobotEnv::publishMarker(const vector<double> &marker_tf,
                                int marker_id,
                                const string &name_space,
                                const string &color,
                                double alpha,
                                const string &geometry,
                                const vector<double> &marker_scale,
                                const string &frame_id = (string &) "") {
    std_msgs::ColorRGBA c = myutils::getColorMsg(color, alpha);
    publishMarkerMsg(marker_tf, marker_id, name_space, c, geometry, marker_scale, frame_id);
}

void RobotEnv::publishMarkerRGB(const vector<double> &marker_tf,
                                   int marker_id,
                                   const string &name_space,
                                   const vector<double> &color,
                                   double alpha,
                                   const string &geometry,
                                   const vector<double> &marker_scale,
                                   const string &frame_id = (string &) "") {
    std_msgs::ColorRGBA c;
    c.r = color[0];
    c.g = color[1];
    c.b = color[2];
    c.a = alpha;
    publishMarkerMsg(marker_tf, marker_id, name_space, c, geometry, marker_scale, frame_id);
}

void RobotEnv::publishMarkerMsg(const vector<double> &marker_tf,
                                   int marker_id,
                                   const string &name_space,
                                   std_msgs::ColorRGBA &color,
                                   const string &geometry,
                                   const vector<double> &marker_scale,
                                   const string &frame_id = (string &) "") {
    std::string frame_id_marker = frame_id.empty() ? robot_config_.frame_id : frame_id;
    tf::Transform t = myutils::listToTf(marker_tf);
    tf::Vector3 scale = myutils::listToVector3(marker_scale);
    visualization_msgs::Marker marker = myutils::markerFromTransform(t, name_space, color, marker_id,
                                                                     frame_id_marker, geometry, scale);
    gripper_visualizer_.publish(marker);
}

void RobotEnv::publishRobotState(){
    moveit_msgs::DisplayRobotState drs;
    robot_state::robotStateToRobotStateMsg(*kinematic_state_, drs.state);
    // rviz won't accept these in map frame for some reason
    // drs.state.joint_state.header.frame_id = "map";
    // drs.state.multi_dof_joint_state.header.frame_id = "map";
    robstate_visualizer_.publish(drs);
}

void RobotEnv::publishTrajectory(){
  moveit_msgs::RobotTrajectory trajectory_msg;
  trajectory_->getRobotTrajectoryMsg(trajectory_msg);
  display_msg_.trajectory.clear();
  display_msg_.trajectory.push_back(trajectory_msg);
  traj_visualizer_.publish(display_msg_);
}

vector<double> RobotEnv::tipToGripperTf(const vector<double> &tip_tf) {
    tf::Transform tf = myutils::listToTf(tip_tf);
    return myutils::tfToList(myutils::tipToGripperGoal(tf, robot_config_.tip_to_gripper_offset, tf::Quaternion(0., 0., 0., 1.)));
}

vector<double> RobotEnv::gripperToTipTf(const vector<double> &tip_tf) {
    tf::Transform tf = myutils::listToTf(tip_tf);
    return myutils::tfToList(myutils::gripperToTipGoal(tf, robot_config_.tip_to_gripper_offset, tf::Quaternion(0., 0., 0., 1.)));
}

void RobotEnv::getBioIkOptions(bio_ik::BioIKKinematicsQueryOptions &ik_options, double dt) const{
    ik_options.return_approximate_solution = true;
    ik_options.replace = true;
    bool secondary = false;
    if (bioik_regularization_type_ == "reg"){
        auto *regularization_goal = new bio_ik::RegularizationGoal(bioik_regularization_weight_);
        ik_options.goals.emplace_back(regularization_goal);
    } else if (bioik_regularization_type_ == "mindispl") {
        auto *regularization_goal = new bio_ik::MinimalDisplacementGoal(bioik_regularization_weight_, secondary);
        ik_options.goals.emplace_back(regularization_goal);
    } else if (bioik_regularization_type_ == "mindisplcubed") {
        auto *regularization_goal = new MinimalDisplacementGoalCubed(bioik_regularization_weight_, secondary);
        ik_options.goals.emplace_back(regularization_goal);
    } else if (bioik_regularization_type_ == "abovevellimit") {
        auto *regularization_goal = new AboveVelLimitGoal(bioik_max_velocities_, dt, bioik_regularization_weight_, secondary);
        ik_options.goals.emplace_back(regularization_goal);
    // } else if (bioik_regularization_type_ == "mymindispl") {
    //     auto *regularization_goal = new MyMinimalDisplacementGoal(bioik_regularization_weight_, secondary);
    //     ik_options.goals.emplace_back(regularization_goal);
    // } else if (bioik_regularization_type_ == "vellimit") {
    //     auto *regularization_goal = new MyVelLimitDisplacementGoal(bioik_regularization_weight_, secondary);
    //     ik_options.goals.emplace_back(regularization_goal);
    } else {
        throw runtime_error("Unknown bioik_regularization_type_");
    }
    if (bioik_avoid_joint_limits_weight_ > 0.0) {
        auto *avoid_joint_limits_goal = new bio_ik::AvoidJointLimitsGoal(bioik_avoid_joint_limits_weight_);
        ik_options.goals.emplace_back(avoid_joint_limits_goal);
    }
    if (bioik_center_joints_weight_ > 0.0) {
        auto *center_joints_goal = new bio_ik::CenterJointsGoal(bioik_center_joints_weight_);
        ik_options.goals.emplace_back(center_joints_goal);
    }
}

double lineSegmentDistanceSquared(const tf2::Vector3 &position1, const tf2::Vector3 &position2, const tf2::Vector3 &point) {
    const tf2::Vector3 segment = position2 - position1;
    const double l2 = (segment).length();
    if (l2 < myutils::epsilon) {
        return (point - position1).length2();
    }
    const double t = myutils::clampDouble((point - position1).dot(segment), 0.0, 1.0);
    tf2::Vector3 projection = position1 + t * segment;
    return (point - projection).length2();
}
