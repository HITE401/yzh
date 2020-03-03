#pragma once

#include<ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include "estimator.h"


extern ros::Publisher  pub_image_track;
extern ros::Publisher pub_odometry;
extern ros::Publisher pub_path;


void registerPub(ros::NodeHandle &n);


void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);
void pubTF(const Estimator &estimator, const std_msgs::Header &header);
void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);