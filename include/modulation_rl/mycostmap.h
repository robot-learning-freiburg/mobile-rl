#pragma once
//#ifndef MODULATION_RL_MYCOSTMAP_H
//#define MODULATION_RL_MYCOSTMAP_H

#include <cmath>
#include <map>

#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/inflation_layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/observation_buffer.h>
#include <costmap_2d/obstacle_layer.h>
#include <costmap_2d/static_layer.h>
#include <modulation_rl/my_static_layer.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/point_cloud2_iterator.h>

//#include <costmap_2d/testing_helper.h>
//#include <costmap_2d/costmap_math.h>

using namespace costmap_2d;
using geometry_msgs::Point;

// SEE https://github.com/ros-planning/navigation/blob/8c4933517b56a9ac8f193068df4d9ff333f21614/costmap_2d/test/inflation_tests.cpp
namespace my_costmap {
    costmap_2d::InflationLayer *addInflationLayer(costmap_2d::LayeredCostmap &layers, tf2_ros::Buffer &tf);
    costmap_2d::MyStaticLayer *addMyStaticLayer(costmap_2d::LayeredCostmap &layers, tf2_ros::Buffer &tf);

    std::vector<Point> setRadii(LayeredCostmap &layers, double length, double width, double inflation_radius);
    void printMap(costmap_2d::Costmap2D &costmap);
    char *getCostTranslationTable();
    // Source:
    // https://github.com/ros-planning/navigation/blob/8c4933517b56a9ac8f193068df4d9ff333f21614/costmap_2d/src/costmap_2d_publisher.cpp#L86
    void prepareGrid(Costmap2D *costmap, nav_msgs::OccupancyGrid &inflated_grid_msg, const std::string &frame_id);

    // gen.add("cost_scaling_factor", double_t, 0, "A scaling factor to apply to cost values during inflation.", 10, 0, 100)
    // gen.add("inflation_radius", double_t, 0, "The radius in meters to which the map inflates obstacle cost values.", 0.55, 0, 50)
    // gen.add("inflate_unknown", bool_t, 0, "Whether to inflate unknown cells.", False)
    nav_msgs::OccupancyGrid getInflatedMap(const nav_msgs::OccupancyGrid &new_map,
                                             double inflation_radius,
                                             double cost_scaling_factor);
    std::vector<int8_t> getInflatedMapPython(const std::vector<int8_t> &data,
                                                double resolution,
                                                double width,
                                                double height,
                                                const std::vector<double> &origin_vector,
                                                const std::string &frame_id,
                                                double inflation_radius,
                                                double cost_scaling_factor);
}  // namespace my_costmap

//#endif //MODULATION_RL_MYCOSTMAP_H
