//
// Created by honerkam on 1/24/22.
//

#ifndef MODULATION_RL_MY_INTERACTIVE_MARKER_H
#define MODULATION_RL_MY_INTERACTIVE_MARKER_H

#include <functional>
#include <memory>
#include <mutex>

#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/menu_handler.h>



/**
 * This class lets the user to command robot form interactive marker.
 */
class MyInteractiveMarker final {
public:
    /**
     * Constructor
     *
     * @param [in] nodeHandle: ROS node handle.
     * @param [in] topicPrefix: The TargetTrajectories will be published on "topicPrefix_mpc_target" topic. Moreover, the latest
     * observation is be expected on "topicPrefix_mpc_observation" topic.
     * @param [in] gaolPoseToTargetTrajectories: A function which transforms the commanded pose to TargetTrajectories.
     */
    MyInteractiveMarker(::ros::NodeHandle& nodeHandle, const std::string& topicPrefix);

    /**
     * Spins ROS to update the interactive markers.
     */
    void publishInteractiveMarker() { ::ros::spin(); }

private:
    visualization_msgs::InteractiveMarker createInteractiveMarker() const;
    void processFirstEntry(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback);
    void processSecondEntry(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback);

    interactive_markers::MenuHandler menuHandler_;
    interactive_markers::InteractiveMarkerServer server_;

    ros::Publisher publisher_;
    ros::Publisher publisher2_;
};


#endif //MODULATION_RL_MY_INTERACTIVE_MARKER_H
