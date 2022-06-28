//
// Created by honerkam on 1/24/22.
//

#include <ros/ros.h>
#include "modulation_rl/my_interactive_marker.h"


int main(int argc, char* argv[]) {
  const std::string robotName = "mobilerl";
  ros::init(argc, argv, robotName + "_target");
  ros::NodeHandle nodeHandle;

  MyInteractiveMarker targetPoseCommand(nodeHandle, robotName);
  targetPoseCommand.publishInteractiveMarker();

  // Successful exit
  return 0;
}
