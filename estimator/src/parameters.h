#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const double INIT_DEPTH = -1.0;    //特征点的初始深度
extern string VINS_OUT_PATH;
extern Eigen::Vector3d G;  
extern double MIN_PARALLAX;
extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern int ROW, COL;
extern string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern vector<std::string> CAM_NAMES;
extern int MAX_CNT; 
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;

extern int USE_IMU;
extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern string IMU_TOPIC;


void readParameters(std::string config_file);

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

