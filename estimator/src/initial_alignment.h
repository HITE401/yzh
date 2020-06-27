#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "imu_factor.h"
#include "utility.h"
#include <ros/ros.h>
#include <map>
#include "feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<Eigen::Matrix<double, 7, 1>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<Eigen::Matrix<double, 7, 1>>> points;
        double t;
        Matrix3d R;
        Vector3d T;
        IntegrationBase *pre_integration;
        bool is_key_frame;
};
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs);