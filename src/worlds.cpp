#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelConfiguration.h>
#include <gazebo_msgs/SetModelState.h>
#include <utility>
#include <modulation_rl/myutils.h>
#include <modulation_rl/worlds.h>


using namespace std;

BaseWorld::BaseWorld(string name, bool is_analytical, string global_link_transform, string base_footprint_name) :
    name_{move(name)},
    is_analytical_{is_analytical},
    global_link_transform_{std::move(global_link_transform)},
    base_footprint_name_{std::move(base_footprint_name)}
{
    awaitTfs();
}

void BaseWorld::awaitTfs() {
    if (name_ != "sim") {
        std::string error_msg;
        if (!listener_.waitForTransform("map", base_footprint_name_, ros::Time::now() + ros::Duration(0.5), ros::Duration(10.0), ros::Duration(0.01), &error_msg)) {
            throw runtime_error(error_msg.c_str());
        }
        if (!listener_.waitForTransform("map", global_link_transform_, ros::Time::now() + ros::Duration(0.5), ros::Duration(10.0), ros::Duration(0.01), &error_msg)) {
            throw runtime_error(error_msg.c_str());
        }
    }
}

tf::Transform BaseWorld::getBaseTransformWorld() {
    if (name_ != "sim") {
        tf::StampedTransform newBaseTransform;
        listener_.lookupTransform("map", base_footprint_name_, ros::Time(0), newBaseTransform);
        // Seems to sometimes return a non-zero z coordinate for e.g. PR2
        newBaseTransform.setOrigin(tf::Vector3(newBaseTransform.getOrigin().x(), newBaseTransform.getOrigin().y(), 0.0));
        return tf::Transform(newBaseTransform);
    } else {
        throw runtime_error("Not implemented for this world type: " + name_);
    }
}

tf::Transform BaseWorld::getGripperTransformWorld(const string &frame_id) {
    tf::StampedTransform newGripperTransform;
    listener_.lookupTransform(frame_id, global_link_transform_, ros::Time(0), newGripperTransform);
    return tf::Transform(newGripperTransform);
}

SimWorld::SimWorld(string global_link_transform, string base_footprint_name) : BaseWorld("sim", true, std::move(global_link_transform), base_footprint_name){};

void SimWorld::setModelState(const string &model_name,
                             const tf::Transform &world_transform,
                             ros::Publisher &cmd_base_vel_pub) {
}

GazeboWorld::GazeboWorld(string global_link_transform, string base_footprint_name) :
    BaseWorld("gazebo", false, std::move(global_link_transform), std::move(base_footprint_name)){
        // set_model_state_client_ = nh_->serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
        // set_model_configuration_client_ = nh_->serviceClient<gazebo_msgs::SetModelConfiguration>("/gazebo/set_model_configuration");
        // pause_gazebo_client_ = nh_->serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
        // unpause_gazebo_client_ = nh_->serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    };

 void GazeboWorld::setModelState(const string &model_name,
                                 const tf::Transform &world_transform,
                                 ros::Publisher &cmd_base_vel_pub) {
    // NOTE: controllers might try to return to previous pose -> stop and restart within inherited class if necessary
    // stop_controllers();

    // pause physics
    std_srvs::Empty emptySrv;
    ros::service::call("/gazebo/pause_physics", emptySrv);

    // set base in gazebo
    gazebo_msgs::ModelState modelstate;
    modelstate.model_name = model_name;
    modelstate.reference_frame = "map";
    modelstate.pose.position.x = world_transform.getOrigin().x();
    modelstate.pose.position.y = world_transform.getOrigin().y();
    modelstate.pose.position.z = world_transform.getOrigin().z();
    modelstate.pose.orientation.x = world_transform.getRotation().x();
    modelstate.pose.orientation.y = world_transform.getRotation().y();
    modelstate.pose.orientation.z = world_transform.getRotation().z();
    modelstate.pose.orientation.w = world_transform.getRotation().w();

    gazebo_msgs::SetModelState set_model_state;
    set_model_state.request.model_state = modelstate;
    if (!ros::service::call("/gazebo/set_model_state", set_model_state)) {
        ROS_ERROR("set_model_state_client_ failed");
    }

     // // set joint positions in gazebo
     // gazebo_msgs::SetModelConfigurationRequest model_configuration;
     // model_configuration.urdf_param_name = "robot_description";
     // model_configuration.model_name = model_name;
     // model_configuration.joint_names = joint_names;
     // model_configuration.joint_positions = joint_values;
     // gazebo_msgs::SetModelConfiguration set_model_configuration;
     // set_model_configuration.request = model_configuration;
     // if (!ros::service::call("/gazebo/set_model_configuration", set_model_configuration)) {
     //     ROS_INFO("set_model_configuration_client_ failed");
     // };

    // unpause physics
    ros::service::call("/gazebo/unpause_physics", emptySrv);
    ros::Duration(0.5).sleep();

    // start_controllers();
}

RealWorld::RealWorld(string global_link_transform,
                     string base_footprint_name,
                     const double base_vel,
                     const double base_rot_vel)
        : BaseWorld("world", false, std::move(global_link_transform), base_footprint_name),
          base_vel_{base_vel},
          base_rot_vel_{base_rot_vel} {}

bool RealWorld::isWithinWorld(const tf::Transform &base_transform) {
    double min_x = -0.0, max_x = 3.5, min_y = -0.0, max_y = 2.0, max_y_small = 1.0;

    tf::Vector3 g = base_transform.getOrigin();
    bool valid = (g.x() >= min_x) && (g.x() <= max_x) && (g.y() >= min_y) && (g.y() <= max_y);
    if (g.x() <= 1.0) {
        valid &= (g.y() <= max_y_small);
    }
    return valid;
}

void RealWorld::setModelState(const string &model_name,
                              const tf::Transform &world_transform,
                              ros::Publisher &cmd_base_vel_pub) {
    if (model_name != "pr2") {
        throw runtime_error("ONLY IMPLEMENTED FOR PR2 SO FAR");
    }
    ros::Rate rate(50);

    tf::Transform currentBaseTransform = getBaseTransformWorld();
    tf::Vector3 goal_vec = world_transform.getOrigin() - currentBaseTransform.getOrigin();

    while (goal_vec.length() > 0.025) {
        // location
        tf::Vector3 scaled_vel = myutils::normScaleVel(goal_vec, 0.0, base_vel_);
        tf::Transform desiredBaseTransform = currentBaseTransform;
        desiredBaseTransform.setOrigin(currentBaseTransform.getOrigin() + scaled_vel);

        // rotation
        double roll_, pitch_, yaw_, yaw2_;
        tf::Matrix3x3(world_transform.getRotation()).getRPY(roll_, pitch_, yaw_);
        tf::Matrix3x3(currentBaseTransform.getRotation()).getRPY(roll_, pitch_, yaw2_);
        double angle_diff = myutils::rpyAngleDiff(yaw_, yaw2_);
        double base_rotation = myutils::clampDouble(angle_diff, -base_rot_vel_, base_rot_vel_);

        tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), base_rotation);
        desiredBaseTransform.setRotation(q * currentBaseTransform.getRotation());

        // construct command
        tf::Transform relative_desired_pose = currentBaseTransform.inverse() * desiredBaseTransform;
        geometry_msgs::Twist base_cmd_rel;
        // double roll_, pitch_, yaw;
        relative_desired_pose.getBasis().getRPY(roll_, pitch_, yaw_);
        base_cmd_rel.linear.x = relative_desired_pose.getOrigin().getX();
        base_cmd_rel.linear.y = relative_desired_pose.getOrigin().getY();
        base_cmd_rel.angular.z = yaw_;

        // publish command
        cmd_base_vel_pub.publish(base_cmd_rel);
        rate.sleep();

        // update
        currentBaseTransform = getBaseTransformWorld();
        goal_vec = world_transform.getOrigin() - currentBaseTransform.getOrigin();
    }
}