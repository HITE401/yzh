#pragma once

#include<opencv2/opencv.hpp>
#include<eigen3/Eigen/Dense>
#include<map>
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

using namespace std;
using namespace Eigen;


class Estimator
{
public:
 	Estimator();

    ~Estimator();
	
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
	Vector3d Ps[(WINDOW_SIZE + 1)], tic[2];
    Matrix3d Rs[(WINDOW_SIZE + 1)], ric[2];     //Rs=T^w_imu
	

private:
	int inputImageCnt;
	mutex mBuf;  //特征点Buf的互斥量
	mutex mProcess;  //这是什么的互斥量？？
	queue<pair<double, map<int, vector<Eigen::Matrix<double, 7, 1>>>>> featureBuf;
	thread processThread;
	bool Flag_thread;   //子线程退出的标志位
	bool marginalization_flag;    //=0则边缘化旧帧
	int frame_count;

	double Headers[(WINDOW_SIZE+1)];  //窗口时间戳


	void processMeasurements();
	void processImage(const map<int, vector<Eigen::Matrix<double, 7, 1>>> &image);

	//对窗口位姿进行优化
	void optimization();

	void updateLatestStates();

	void slideWindow();

	void clearState();     //主要是将特征点清除，状态变量初始化，特征管理类清除

	void outliersRejection(set<int> &removeIndex);

	double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj);

};