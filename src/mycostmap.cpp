#include "modulation_rl/mycostmap.h"
#include <modulation_rl/myutils.h>


// SEE https://github.com/ros-planning/navigation/blob/8c4933517b56a9ac8f193068df4d9ff333f21614/costmap_2d/test/inflation_tests.cpp
namespace my_costmap{
    costmap_2d::InflationLayer* addInflationLayer(costmap_2d::LayeredCostmap& layers, tf2_ros::Buffer& tf){
        costmap_2d::InflationLayer* ilayer = new costmap_2d::InflationLayer();
        ilayer->initialize(&layers, "inflation", &tf);
        boost::shared_ptr<costmap_2d::Layer> ipointer(ilayer);
        layers.addPlugin(ipointer);
        return ilayer;
    }

//    costmap_2d::ObstacleLayer* addObstacleLayer(costmap_2d::LayeredCostmap& layers, tf2_ros::Buffer& tf){
//        costmap_2d::ObstacleLayer* olayer = new costmap_2d::ObstacleLayer();
//        olayer->initialize(&layers, "obstacles", &tf);
//        layers.addPlugin(boost::shared_ptr<costmap_2d::Layer>(olayer));
//        return olayer;
//    }

    costmap_2d::MyStaticLayer * addMyStaticLayer(costmap_2d::LayeredCostmap& layers, tf2_ros::Buffer& tf){
        costmap_2d::MyStaticLayer * slayer = new costmap_2d::MyStaticLayer();
        layers.addPlugin(boost::shared_ptr<costmap_2d::Layer>(slayer));
        slayer->initialize(&layers, "static", &tf);
        return slayer;
    }

    std::vector<Point> setRadii(LayeredCostmap& layers, double length, double width, double inflation_radius){
        std::vector<Point> polygon;
        Point p;
        p.x = width;
        p.y = length;
        polygon.push_back(p);
        p.x = width;
        p.y = -length;
        polygon.push_back(p);
        p.x = -width;
        p.y = -length;
        polygon.push_back(p);
        p.x = -width;
        p.y = length;
        polygon.push_back(p);
        layers.setFootprint(polygon);

        ros::NodeHandle nh;
        nh.setParam("/inflation_tests/inflation/inflation_radius", inflation_radius);

        return polygon;
    }

    void printMap(costmap_2d::Costmap2D& costmap){
        printf("map:\n");
        for (int i = 0; i < costmap.getSizeInCellsY(); i++){
            for (int j = 0; j < costmap.getSizeInCellsX(); j++){
                printf("%4d", int(costmap.getCost(j, i)));
            }
            printf("\n\n");
        }
    }

    char* getCostTranslationTable(){
        char* cost_translation_table = new char[256];

        // special values:
        cost_translation_table[0] = 0;  // NO obstacle
        cost_translation_table[253] = 99;  // INSCRIBED obstacle
        cost_translation_table[254] = 100;  // LETHAL obstacle
        cost_translation_table[255] = -1;  // UNKNOWN

        // regular cost values scale the range 1 to 252 (inclusive) to fit
        // into 1 to 98 (inclusive).
        for (int i = 1; i < 253; i++){
            cost_translation_table[ i ] = char(1 + (97 * (i - 1)) / 251);
        }
        return cost_translation_table;
    }

    // Source: https://github.com/ros-planning/navigation/blob/8c4933517b56a9ac8f193068df4d9ff333f21614/costmap_2d/src/costmap_2d_publisher.cpp#L86
    void prepareGrid(Costmap2D* costmap, nav_msgs::OccupancyGrid &inflated_grid_msg, const std::string &frame_id){


        boost::unique_lock<Costmap2D::mutex_t> lock(*(costmap->getMutex()));
        double resolution = costmap->getResolution();

        inflated_grid_msg.header.frame_id = frame_id;
        inflated_grid_msg.header.stamp = ros::Time::now();
        inflated_grid_msg.info.resolution = resolution;

        inflated_grid_msg.info.width = costmap->getSizeInCellsX();
        inflated_grid_msg.info.height = costmap->getSizeInCellsY();

        double wx, wy;
        costmap->mapToWorld(0, 0, wx, wy);
        inflated_grid_msg.info.origin.position.x = wx - resolution / 2;
        inflated_grid_msg.info.origin.position.y = wy - resolution / 2;
        inflated_grid_msg.info.origin.position.z = 0.0;
        inflated_grid_msg.info.origin.orientation.w = 1.0;
        // saved_origin_x_ = costmap->getOriginX();
        // saved_origin_y_ = costmap->getOriginY();

        inflated_grid_msg.data.resize(inflated_grid_msg.info.width * inflated_grid_msg.info.height);

        unsigned char* data = costmap->getCharMap();
        char* cost_translation_table = getCostTranslationTable();
        for (unsigned int i = 0; i < inflated_grid_msg.data.size(); i++){
            inflated_grid_msg.data[i] = cost_translation_table[ data[ i ]];
        }
    }

    // gen.add("cost_scaling_factor", double_t, 0, "A scaling factor to apply to cost values during inflation.", 10, 0, 100)
    // gen.add("inflation_radius", double_t, 0, "The radius in meters to which the map inflates obstacle cost values.", 0.55, 0, 50)
    // gen.add("inflate_unknown", bool_t, 0, "Whether to inflate unknown cells.", False)
    nav_msgs::OccupancyGrid getInflatedMap(const nav_msgs::OccupancyGrid& new_map, double inflation_radius, double cost_scaling_factor){
        tf2_ros::Buffer tf;
        LayeredCostmap layers(new_map.header.frame_id, false, false);

        MyStaticLayer *slayer = addMyStaticLayer(layers, tf);
        InflationLayer* ilayer = addInflationLayer(layers, tf);
        ilayer->setInflationParameters(inflation_radius, cost_scaling_factor);

        // TODO: don't know if needed to call setFootprint
        std::vector<Point> polygon = setRadii(layers, 0.0, 0.0, 10);
        layers.setFootprint(polygon);

        slayer->loadOccupancyGrid(new_map);
        layers.updateMap(new_map.info.origin.position.x ,new_map.info.origin.position.y,0);

        Costmap2D* costmap = layers.getCostmap();
        // printMap(*costmap);

        nav_msgs::OccupancyGrid inflated_grid_msg;
        prepareGrid(costmap, inflated_grid_msg, new_map.header.frame_id);
        return inflated_grid_msg;
    }

    std::vector<int8_t> getInflatedMapPython(const std::vector<int8_t> &data,
                                                double resolution,
                                                double width,
                                                double height,
                                                const std::vector<double> &origin_vector,
                                                const std::string &frame_id,
                                                double inflation_radius,
                                                double cost_scaling_factor){
        nav_msgs::OccupancyGrid new_map;
        new_map.header.frame_id = frame_id;
        new_map.info.resolution = resolution;
        new_map.info.width = width;
        new_map.info.height = height;
        geometry_msgs::Pose origin;
        origin.position.x = origin_vector[0];
        origin.position.y = origin_vector[1];
        origin.position.z = origin_vector[2];
        origin.orientation.x = origin_vector[3];
        origin.orientation.y = origin_vector[4];
        origin.orientation.z = origin_vector[5];
        origin.orientation.w = origin_vector[6];
        new_map.info.origin = origin;
        new_map.data = data;
        nav_msgs::OccupancyGrid inflated_map = getInflatedMap(new_map, inflation_radius, cost_scaling_factor);
        return inflated_map.data;
    }
}