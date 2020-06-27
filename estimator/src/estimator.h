#pragma once

#include<opencv2/opencv.hpp>
#include<eigen3/Eigen/Dense>
#include<ceres/ceres.h>
#include<unordered_map>
#include<queue>
#include<mutex>
#include<thread>
#include<string>
#include<std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include"featuretracker.h"
#include"feature_manager.h"
#include"parameters.h"
#include "utility.h"
#include "projectionFactor.h"
#include "marginalization_factor.h"
#include "imu_factor.h"
#include "initial_alignment.h"

using namespace std;
using namespace Eigen;

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};



class Estimator
{
public:
 	Estimator();
	
	/**
	 @description: 设置一些必要的参数
	*/
	void setParameter();

	/**
	 @description: 对双目图像进行处理，得到特征点存到featureBuf中
	 @param: time[in]  左图像的时间戳
	@param: imgleft[in] 左图像opencv格式
	@param: imgright[in] 右图像opencv格式
	*/
	void inputImage(double time, cv::Mat &imgleft, cv::Mat &imgright);

	/**
	@description: 对IMU数据进行处理
	@param: time[in] imu数据的时间戳
	@return: acc[in] 加速度计的数据
	@return: gyr[in] 陀螺仪的数据
	*/
	void inputIMU(double time, Vector3d acc, Vector3d gyr);


	FeatureTracker featureTracker;
	FeatureManager f_manager;
	bool solver_flag;
	Vector3d tic[2];
    Matrix3d ric[2];    
	Vector3d        Ps[(WINDOW_SIZE + 1)];    //Rs=T^w_imu
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];


private:
	mutex mBuf;  //特征点Buf的互斥量
	mutex mProcess;  //这是什么的互斥量？？
	queue<pair<double, map<int, vector<Eigen::Matrix<double, 7, 1>>>>> featureBuf;


	
	bool marginalization_flag;    //=0则边缘化旧帧
	int frame_count;

	double Headers[(WINDOW_SIZE+1)];  //窗口时间戳,好像没用到？？

	double para_Pose[WINDOW_SIZE + 1][7];     //优化位姿变量
	double para_SpeedBias[WINDOW_SIZE + 1][9];
	double para_Feature[1000][1];              //优化特征点深度
	double para_Ex_Pose[2][7];					//固定的外参变量

	Vector3d g;  //世界坐标系下的重力加速度

	vector<double *> last_marginalization_parameter_blocks;    
	MarginalizationInfo *last_marginalization_info;


	void processMeasurements();
	void processImage(const map<int, vector<Eigen::Matrix<double, 7, 1>>> &image, const double time);

	//对窗口位姿进行优化
	void optimization();

	void updateLatestStates();

	void slideWindow();

	void clearState();     //主要是将特征点清除，状态变量初始化，特征管理类清除

	void outliersRejection(set<int> &removeIndex);

	double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj);

	void vector2double();
	void double2vector();

	void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
	 double latest_time;   //表示IMU最新的时间戳，用来求dt
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;
	std::mutex mPropagate;    //保护last相关变量
	Vector3d acc_0, gyr_0;   
	queue<pair<double, Eigen::Vector3d>> accBuf;  //存储IMU数据
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    double prevTime, curTime;

	vector<double> dt_buf[(WINDOW_SIZE + 1)];     //存储窗口内的所有IMU数据
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;
    bool initFirstPoseFlag;
    bool first_imu;   //给acc_0和gyr_0赋初值


	void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);
	bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector);
	void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);


};