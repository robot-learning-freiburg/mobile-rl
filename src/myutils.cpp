#include <math.h>
#include <modulation_rl/myutils.h>

using namespace std;

namespace myutils {
    void printVector3(tf::Vector3 v, const string& descr) {
        cout << descr << ": " << v.x() << ", " << v.y() << ", " << v.z() << endl;
    }

    void printQ(tf::Quaternion q, const string& descr) {
        cout << descr << ": " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << endl;
    }

    void printT(tf::Transform t, const string& descr) {
        tf::Vector3 v = t.getOrigin();
        tf::Quaternion q = t.getRotation();
        cout << std::fixed << std::setprecision(3) << std::setw(125) << descr << ". O: " << v.x() << ", " << v.y() << ", " << v.z()
                  << ", Q: " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << endl;
    }

    void printArrayDouble(vector<double> array, const string& descr) {
        cout << descr << ", size: " << array.size() << ", ";
        for (int i = 0; i < array.size(); i++) {
            cout << array[i] << ", ";
        }
        cout << endl;
    }

    void printArrayStr(vector<string> array, const string& descr) {
        cout << descr << array.size() << endl;
        for (int i = 0; i < array.size(); i++) {
            cout << array[i] << ", ";
        }
        cout << endl;
    }

    tf::Vector3 qToRpy(tf::Quaternion q) {
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        return tf::Vector3(roll, pitch, yaw);
    }

    double calcRotDist(const tf::Transform &a, const tf::Transform &b) {
        double inner_prod = a.getRotation().dot(b.getRotation());
        return 1.0 - pow(inner_prod, 2.0);
    }

    double vec3AbsMax(tf::Vector3 v) {
        tf::Vector3 v_abs = v.absolute();
        return max(max(v_abs.x(), v_abs.y()), v_abs.z());
    }

    visualization_msgs::Marker markerFromTransform(tf::Transform t,
                                                   string ns,
                                                   std_msgs::ColorRGBA color,
                                                   int marker_id,
                                                   string frame_id,
                                                   const string &geometry,
                                                   tf::Vector3 marker_scale) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = move(frame_id);
        marker.header.stamp = ros::Time();
        marker.ns = move(ns);

        if (geometry == "arrow") {
            marker.type = visualization_msgs::Marker::ARROW;
        } else if (geometry == "cube") {
            marker.type = visualization_msgs::Marker::CUBE;
        } else if (geometry == "cylinder") {
            marker.type = visualization_msgs::Marker::CYLINDER;
        } else if (geometry == "sphere") {
            marker.type = visualization_msgs::Marker::SPHERE;
        } else {
            throw runtime_error("Unknown marker geometry");
        }
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = t.getOrigin().x();
        marker.pose.position.y = t.getOrigin().y();
        marker.pose.position.z = t.getOrigin().z();
        marker.pose.orientation.x = t.getRotation().x();
        marker.pose.orientation.y = t.getRotation().y();
        marker.pose.orientation.z = t.getRotation().z();
        marker.pose.orientation.w = t.getRotation().w();
        marker.scale.x = marker_scale.x();
        marker.scale.y = marker_scale.y();
        marker.scale.z = marker_scale.z();

        // more and more red from 0 to 100
        marker.color = color;
        marker.id = marker_id;
        return marker;
    }

    std_msgs::ColorRGBA getColorMsg(const string &color_name, double alpha) {
        std_msgs::ColorRGBA c;
        if (color_name == "blue") {
            c.b = 1.0;
        } else if (color_name == "pink") {
            c.r = 1.0;
            c.g = 105.0 / 255.0;
            c.b = 147.0 / 255.0;
        } else if (color_name == "orange") {
            c.r = 1.0;
            c.g = 159.0 / 255.0;
            c.b = 0.0;
        } else if (color_name == "yellow") {
            c.r = 1.0;
            c.g = 1.0;
            c.b = 0.0;
        } else if (color_name == "cyan") {
            c.r = 0.0;
            c.g = 128.0 / 255.0;
            c.b = 1.0;
        } else if (color_name == "green") {
            c.r = 0.0;
            c.g = 1.0;
            c.b = 0.0;
        } else if (color_name == "red") {
            c.r = 1.0;
            c.g = 0.0;
            c.b = 0.0;
        } else {
            throw runtime_error("unknown color");
        }
        c.a = alpha;
        return c;
    }

    tf::Vector3 minMaxScaleVel(tf::Vector3 vel, double min_vel, double max_vel) {
        // find denominator to keep it in range [min_planner_velocity_, max_planner_velocity_]
        double max_abs_vector_value = myutils::vec3AbsMax(vel);
        // in case vel is a vector of all zeros avoid division by zero
        if (max_abs_vector_value == 0.0) {
            return tf::Vector3(min_vel, min_vel, min_vel);
        }
        double max_denom;
        if (min_vel < 0.001) {
            max_denom = 1.0;
        } else {
            max_denom = min(max_abs_vector_value / min_vel, 1.0);
        }
        double min_denom = max_abs_vector_value / max_vel;
        double denom = max(max_denom, min_denom);
        return vel / denom;
    }

    tf::Vector3 maxClipVel(tf::Vector3 vel, double max_vel) {
        tf::Vector3 clipped_vel;
        clipped_vel.setX(max(min(vel.x(), max_vel), -max_vel));
        clipped_vel.setY(max(min(vel.y(), max_vel), -max_vel));
        clipped_vel.setZ(max(min(vel.z(), max_vel), -max_vel));
        return clipped_vel;
    }

    tf::Vector3 normScaleVel(tf::Vector3 vel, double min_vel_norm, double max_vel_norm) {
        double norm = vel.length();
        if (norm == 0.0) {
            return vel;
        } else if (max_vel_norm == 0.0) {
            return tf::Vector3(0.0, 0.0, 0.0);
        } else {
            double max_denom;
            if (min_vel_norm < epsilon) {
                max_denom = 1.0;
            } else {
                max_denom = min(norm / min_vel_norm, 1.0);
            }
            double min_denom = norm / max_vel_norm;
            double denom = max(max_denom, min_denom);

            //        assert((vel / denom).length() >= min_vel_norm - 0.001);
            //        assert((vel / denom).length() <= max_vel_norm + 0.001);

            return vel / denom;
        }
    }

    vector<double> normScaleVelPython(vector<double> vel, double min_vel_norm, double max_vel_norm) {
        if (vel.size() != 3){
            throw runtime_error("Velocity should be a vector of length 3");
        }
        tf::Vector3 vel_scaled = myutils::normScaleVel(tf::Vector3(vel[0], vel[1], vel[2]), min_vel_norm, max_vel_norm);
        vector<double> vel_scaled_list{vel_scaled[0], vel_scaled[1], vel_scaled[2]};
        return vel_scaled_list;
    }

    double clampDouble(double value, double min_value, double max_value) { return max(min(value, max_value), min_value); }

    vector<double> tipToGripperGoalPython(const vector<double> &gripper_tip_goal_world,
                                          const vector<double> &tip_to_gripper_offset) {
        tf::Transform transform = tipToGripperGoal(
            listToTf(gripper_tip_goal_world), listToVector3(tip_to_gripper_offset), tf::Quaternion(0., 0., 0., 1.));
        return tfToList(transform);
    }

    tf::Transform tipToGripperGoal(const tf::Transform &gripper_tip_goal_world,
                                   const tf::Vector3 &tip_to_gripper_offset,
                                   const tf::Quaternion &gripper_to_base_rot_offset) {
        // gripper tip offset from wrist
        tf::Transform goal_no_trans(gripper_tip_goal_world);
        goal_no_trans.setOrigin(tf::Vector3(0, 0, 0));
        tf::Vector3 offset_pos = goal_no_trans * tip_to_gripper_offset;

        tf::Transform gripper_goal_wrist_world(gripper_tip_goal_world);
        gripper_goal_wrist_world.setOrigin(gripper_goal_wrist_world.getOrigin() - offset_pos);

        // different rotations between gripper joint and base/world
        gripper_goal_wrist_world.setRotation((gripper_goal_wrist_world.getRotation() * gripper_to_base_rot_offset).normalized());
        // myutils::print_vector3(offset_pos, "offset_pos");
        // myutils::print_q(gripper_to_base_rot_offset, "gripper_to_base_rot_offset");
        // myutils::print_t(gripper_goal_wrist_world, "gripper_goal_wrist_world");
        return gripper_goal_wrist_world;
    }

//    vector<double> gripperToTipGoalPython(const vector<double> &gripper_wrist_goal_world,
//                                         const vector<double> &tip_to_gripper_offset,
//                                         const vector<double> &gripper_to_base_rot_offset){
//        tf::Transform transform = gripperToTipGoal(listToTf(gripper_wrist_goal_world),
//                                                   listToVector3(tip_to_gripper_offset),
//                                                   listToQuaternion(gripper_to_base_rot_offset));
//        return tfToList(transform);
//    }

    tf::Transform gripperToTipGoal(const tf::Transform &gripper_wrist_goal_world,
                                   const tf::Vector3 &tip_to_gripper_offset,
                                   const tf::Quaternion &gripper_to_base_rot_offset) {
        tf::Transform gripper_goal_tip_world;
        gripper_goal_tip_world.setIdentity();

        // different rotations between gripper joint and base/world
        gripper_goal_tip_world.setRotation(gripper_wrist_goal_world.getRotation() * gripper_to_base_rot_offset.inverse());

        // gripper tip offset from wrist
        tf::Vector3 offset_pos = gripper_goal_tip_world * tip_to_gripper_offset;
        gripper_goal_tip_world.setOrigin(gripper_wrist_goal_world.getOrigin() + offset_pos);

        return gripper_goal_tip_world;
    }

    double rpyAngleDiff(double next, double prev) {
        double diff = next - prev;
        if (diff > M_PI) {
            diff = -2 * M_PI + diff;
        } else if (diff < -M_PI) {
            diff = 2 * M_PI + diff;
        }
        return diff;
    }

    bool startsWith(const string &str, const string &substr) { return (str.find(substr) == 0); }

    bool endsWith(const string &str, const string &substr) {
        size_t pos = str.rfind(substr);
        if (pos == string::npos)  // doesnt even contain it
            return false;

        size_t len = str.length();
        size_t elen = substr.length();
        // at end means: Pos found + length of end equal length of full string.
        if (pos + elen == len) {
            return true;
        }

        // not at end
        return false;
    }

    string trim(const string &s) {
        if (s.length() == 0)
            return s;
        size_t b = s.find_first_not_of(" \t\r\n");
        size_t e = s.find_last_not_of(" \t\r\n");
        if (b == string::npos)
            return "";
        return string(s, b, e - b + 1);
    }

    tf::Transform listToTf(const vector<double> &input) {
        tf::Quaternion rotation;
        if (input.size() == 6) {
            rotation.setRPY(input[3], input[4], input[5]);
        } else if (input.size() == 7) {
            rotation = tf::Quaternion(input[3], input[4], input[5], input[6]);
        } else {
            throw runtime_error("invalid length of specified gripper goal");
        }
        return tf::Transform(rotation, tf::Vector3(input[0], input[1], input[2]));
    }

    vector<double> tfToList(const tf::Transform &input, bool normalize_q) {
        vector<double> output;
        output.push_back(input.getOrigin().x());
        output.push_back(input.getOrigin().y());
        output.push_back(input.getOrigin().z());
        tf::Quaternion q(input.getRotation());
        if (normalize_q) {
            q.normalize();
        }
        output.push_back(q.x());
        output.push_back(q.y());
        output.push_back(q.z());
        output.push_back(q.w());
        return output;
    }

    vector<double> vector3ToList(const tf::Vector3 &input) {
        vector<double> output;
        output.push_back(input.x());
        output.push_back(input.y());
        output.push_back(input.z());
        return output;
    }

    tf::Vector3 listToVector3(const vector<double> &input) {
        if (input.size() == 3) {
            return tf::Vector3(input[0], input[1], input[2]);
        } else {
            throw runtime_error("invalid length of specified gripper goal");
        }
    }

    vector<double> quaternionToList(const tf::Quaternion &input, bool normalize_q) {
        vector<double> output;
        tf::Quaternion q(input);
        if (normalize_q) {
            q.normalize();
        }
        output.push_back(q.x());
        output.push_back(q.y());
        output.push_back(q.z());
        output.push_back(q.w());
        return output;
    }

    tf::Quaternion listToQuaternion(const vector<double> &input) {
        if (input.size() == 4) {
            return tf::Quaternion(input[0], input[1], input[2], input[3]);
        } else {
            throw runtime_error("invalid length of specified gripper goal");
        }
    }

    tf::Quaternion calcDq(tf::Quaternion current, tf::Quaternion next) {
        // planned change in rotation defined as dq * current == next
        return (next * current.inverse()).normalized();
    }

    // untested
    bool tfAlmostEqual(tf::Transform a, tf::Transform b) {
        bool equal = (a.getOrigin() - b.getOrigin()).length() < 0.05;
        // NOTE: not sure if this is invariant to all equivalent quaternions
        equal &= (1.0 - pow(a.getRotation().normalized().dot(b.getRotation().normalized()), 2.0)) < 0.05;
        if (!equal) {
            myutils::printT(a, "a");
            myutils::printT(b, "b");
        }
        return equal;
    }

    vector<double> pythonMultiplyTfs(const vector<double> &tf1_list, const vector<double> &tf2_list, bool invert_tf1) {
        tf::Transform tf1 = listToTf(tf1_list);
        tf::Transform tf2 = listToTf(tf2_list);
        if (invert_tf1) {
            tf1 = tf1.inverse();
        }
        return tfToList(tf1 * tf2);
    }

    double pythonAngleShortestPath(const vector<double> &q1, vector<double> &q2){
        return listToQuaternion(q1).angleShortestPath(listToQuaternion(q2));
    }

    tf::Transform calcDesiredBaseTfOmni(const tf::Transform &base_tf,
                                        const tf::Vector3 &base_translation_relative,
                                        const double base_rotation_relative,
                                        const double dt) {
        tf::Transform desired_base_tf;
        tf::Transform base_no_trans(base_tf.getRotation(), tf::Vector3(0., 0., 0.));

        // translate base
        tf::Vector3 base_translation_relative_dt = dt * base_translation_relative;
        tf::Vector3 base_translation_world_dt = base_no_trans * base_translation_relative_dt;
        desired_base_tf.setOrigin(base_tf.getOrigin() + base_translation_world_dt);
        // rotate base
        tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), dt * base_rotation_relative);
        desired_base_tf.setRotation(q * base_tf.getRotation());
        return desired_base_tf;
    }

    void calcDesiredBaseCommandOmni(const tf::Transform &current_base_tf,
                                    const tf::Transform &desired_base_tf,
                                    const double dt,
                                    tf::Vector3 &base_translation_per_second,
                                    double &base_rotation_per_second){
      if (dt < epsilon){
        throw runtime_error("dt cannot be zero");
      }
      tf::Transform base_no_trans(current_base_tf.getRotation(), tf::Vector3(0., 0., 0.));

      base_translation_per_second = base_no_trans.inverse() * (desired_base_tf.getOrigin() - current_base_tf.getOrigin()) / dt;
      base_rotation_per_second = rpyAngleDiff(qToRpy(desired_base_tf.getRotation()).z(), qToRpy(current_base_tf.getRotation()).z()) / dt;
    }

    tf::Transform calcDesiredBaseTfDiffDrive(const tf::Transform &base_tf,
                                             const tf::Vector3 &base_translation_relative,
                                             const double angle,
                                             const double dt) {
        double vel_forward_dt = dt * base_translation_relative.x();
        double angle_dt = dt * angle;

        tf::Transform desired_base_tf;

        tf::Quaternion q(tf::Vector3(0.0, 0.0, 1.0), angle_dt);
        desired_base_tf.setRotation(q * base_tf.getRotation());

        // world reference frame: move relative to base_angle defined by now rotated base
        double base_angle = desired_base_tf.getRotation().getAngle() * desired_base_tf.getRotation().getAxis().getZ();
        tf::Vector3 base_velocity_dif_drive(vel_forward_dt * cos(base_angle), vel_forward_dt * sin(base_angle), 0.0);
        desired_base_tf.setOrigin(base_tf.getOrigin() + base_velocity_dif_drive);
        return desired_base_tf;
    }

    void calcDesiredBaseCommandDiffDrive(const tf::Transform &current_base_tf,
                                         const tf::Transform &desired_base_tf,
                                         const double dt,
                                         tf::Vector3 &base_translation_per_second,
                                         double &base_rotation_per_second){
      if (dt < epsilon){
        throw runtime_error("dt cannot be zero");
      }
      throw runtime_error("Not implemented for diff-drive yet");

      tf::Transform base_no_trans(current_base_tf.getRotation(), tf::Vector3(0., 0., 0.));
      tf::Vector3 delta = base_no_trans.inverse() * (desired_base_tf.getOrigin() - current_base_tf.getOrigin());

      base_translation_per_second[0] = delta.length() / dt;
      if (delta.x() < 0.0){
        base_translation_per_second[0] *= -1.0;
      }

      base_rotation_per_second = std::atan2(delta.y(), delta.x());
      if (base_rotation_per_second > M_PI) {
        base_rotation_per_second -= 2 * M_PI;
      }
    }

    double trueModulo(double a, double b){
        return fmod(fmod(a, b) + b, b);
    }

    vector<vector<double>> interpolateZ(vector<double> &cum_dists, vector<double> &obstacle_zs, const double &current_z, const double & max_map_height) {
        double z_level = current_z;
        vector<double> ts {0.0};
        vector<double> zs {current_z};

        double t, z;
        double eps = 1e-3;
        for (int i = 1; i < cum_dists.size(); i++) {
            t = cum_dists[i];
            z = obstacle_zs[i];
            if ((z > z_level + eps) && (z < max_map_height)) {
                ts.push_back(t);
                zs.push_back(z);
                z_level = z;
            } else if (z < z_level - eps) {
                if (i > 1) {
                    ts.push_back(t);
                    // add with last z so we interpolate straight ahead to the end of the previous obstacle first
                    zs.push_back(z_level);
                }
                z_level = z;
            }
        }
        ts.push_back(cum_dists[cum_dists.size() - 1]);
        zs.push_back(obstacle_zs[obstacle_zs.size() - 1]);

        vector<vector<double>> out;
        out.push_back(ts);
        out.push_back(zs);
        return out;
    }

    vector<double> slerpPython(vector<double> &q_list, vector<double> &q2_list, const double &slerp_pct){
        tf::Quaternion planned_q = listToQuaternion(q_list).slerp(listToQuaternion(q2_list), clampDouble(slerp_pct, 0.0, 0.9999));
        return quaternionToList(planned_q.normalized());
    }
}  // namespace myutils
