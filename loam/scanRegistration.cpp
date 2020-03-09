/******************************读前须知*****************************************/
/*imu为x轴向前,y轴向左,z轴向上的右手坐标系，
  lidar被安装为x轴向前,y轴向左,z轴向上的右手坐标系，
  scanRegistration会把两者通过交换坐标轴，都统一到z轴向前,x轴向左,y轴向上的右手坐标系
  交换后：R = Ry(yaw)*Rx(pitch)*Rz(roll)
*******************************************************************************/

#include <cmath>
#include <vector>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <loam_velodyne/common.h>

using std::sin;
using std::cos;
using std::atan2;

const double scanPeriod = 0.1;   //扫描周期, velodyne频率10Hz，周期0.1s

const int systemDelay = 20;//弃用前20帧初始数据
int systemInitCount = 0;
bool systemInited = false;

const int N_SCANS = 16;   //激光雷达线数

float cloudCurvature[40000];    //点云曲率, 40000为一帧点云中点的最大数量
int cloudSortInd[40000];   //排序曲率点对应的索引
int cloudNeighborPicked[40000];   //点是否筛选过标志：0-未筛选过，1-筛选过
int cloudLabel[40000]; //(按照-1,0,1,2标签分的)

int imuPointerFront = 0;     //最早的imu时间戳大于当前点云时间戳
int imuPointerLast = -1;   //imu最新收到的点在数组中的位置
const int imuQueLength = 200;   //imu循环队列长度

//点云数据开始第一个点和当前点的位移/速度/欧拉角
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

//每次点云数据当前点相对于开始第一个点的畸变位移，速度
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

//IMU信息
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};   //世界———imu
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};
float imuAccX[imuQueLength] = {0};   //imu坐标系
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};
float imuVeloX[imuQueLength] = {0};   //世界坐标系
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};
float imuShiftX[imuQueLength] = {0};   //世界坐标
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

//计算局部坐标系下点云中的点相对第一个开始点的由于加减速运动产生的位移畸变(增量)
void ShiftToStartIMU(float pointTime)
{
  //计算相对于第一个点由于加减速产生的畸变位移(全局坐标系下畸变位移量delta_Tg)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

//从世界坐标系转到点云帧初始坐标系
  //绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;
  //绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
  //绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

//计算局部坐标系下点云中的点相对第一个开始点由于加减速产生的的速度畸变（增量）,有必要每遍都计算吗？是不是只用计算开始和末尾
void VeloToStartIMU()
{
  //计算相对于第一个点由于加减速产生的畸变速度(全局坐标系下畸变速度增量delta_Vg)
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

//从世界坐标系转到激光雷达坐标系
  //绕y轴旋转(-imuYawStart)，即Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;
  //绕x轴旋转(-imuPitchStart)，即Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;
  //绕z轴旋转(-imuRollStart)，即Rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

//去除点云加减速产生的位移畸变，p点坐标是不是已经在
void TransformToStartIMU(PointType *p)
{
//转到世界坐标系
  //绕z轴旋转(imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;
  //绕x轴旋转(imuPitchCur)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;
  //绕y轴旋转(imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  //再转到激光雷达第一个点的坐标系
  //绕y轴旋转(-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;
  //绕x轴旋转(-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;
  //绕z轴旋转(-imuRollStart)，然后叠加平移量
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

//坐标变换得到世界坐标系的加速度，再积分求得偏移量xyz
void AccumulateIMUShift()
{
  //当前IMU坐标系的加速度
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];   
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  //将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，转换得到世界坐标系下的加速度值(right hand rule)
  //绕z轴旋转(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //绕x轴旋转(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //绕y轴旋转(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2; 

  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  if (timeDiff < scanPeriod) {
	//（隐含从静止开始匀加速运动）
	imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff  + accX * timeDiff * timeDiff / 2;
	imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff  + accY * timeDiff * timeDiff / 2;
	imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff + accZ * timeDiff * timeDiff / 2;
	imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
	imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
	imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

//接收点云数据，velodyne雷达坐标系安装为x轴向前，y轴向左，z轴向上的右手坐标系
void  laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  if (!systemInited) {//丢弃前20个点云数据
	systemInitCount++;
	if (systemInitCount >= systemDelay) {
	  systemInited = true;
	}
	return;
  }

  //记录每个scan有曲率的点的开始和结束索引
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  double timeScanCur = laserCloudMsg->header.stamp.toSec();
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);    //移除空点
  int cloudSize = laserCloudIn.points.size();
  //lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;


//-pi和pi的边界处容易出问题，特殊处理
  if (endOri - startOri > 3 * M_PI) {
	endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
	endOri += 2 * M_PI;
  }

  //lidar扫描线是否旋转过半
  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++) {
	point.x = laserCloudIn.points[i].y;   //坐标轴交换
	point.y = laserCloudIn.points[i].z;
	point.z = laserCloudIn.points[i].x;

	//计算点的仰角从而确定激光线号，velodyne每两个scan之间间隔2度	
	float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
	int scanID;
	
	int roundedAngle = round(angle);
	if (roundedAngle > 0)
	  scanID = roundedAngle;
	else 
	  scanID = roundedAngle + (N_SCANS - 1);
	
	//过滤[-15度，+15度]范围之外的点
	if (scanID > (N_SCANS - 1) || scanID < 0 ){
	  count--;
	  continue;
	}

	//该点的旋转角
	float ori = -atan2(point.x, point.z);
	if (!halfPassed) {//根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿  ，确保-pi/2 < ori - startOri < 3*pi/2？？？
	  if (ori < startOri - M_PI / 2) 
		ori += 2 * M_PI;
	  else if (ori > startOri + M_PI * 3 / 2) 
		ori -= 2 * M_PI;
	  if (ori - startOri > M_PI) 
		halfPassed = true;
	} 
	else {
	  ori += 2 * M_PI;
	  if (ori < endOri - M_PI * 3 / 2)    //确保-3*pi/2 < ori - endOri < pi/2   ？？？
		ori += 2 * M_PI;
	  else if (ori > endOri + M_PI / 2) 
		ori -= 2 * M_PI;
	}
	float relTime = (ori - startOri) / (endOri - startOri);
	point.intensity = scanID + scanPeriod * relTime;   //点强度=线号+点相对时间（整数部分是线号，小数部分是该点的相对时间）

//如果收到IMU数据,使用IMU矫正点云畸变
	if (imuPointerLast >= 0) {
	  float pointTime = relTime * scanPeriod;
	  while (imuPointerFront != imuPointerLast) {
		if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
		  break;
		}
		imuPointerFront = (imuPointerFront + 1) % imuQueLength;
	  }

	  if (timeScanCur + pointTime > imuTime[imuPointerFront]) {//没找到,只能以当前最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用
		imuRollCur = imuRoll[imuPointerFront];
		imuPitchCur = imuPitch[imuPointerFront];
		imuYawCur = imuYaw[imuPointerFront];
		imuVeloXCur = imuVeloX[imuPointerFront];
		imuVeloYCur = imuVeloY[imuPointerFront];
		imuVeloZCur = imuVeloZ[imuPointerFront];
		imuShiftXCur = imuShiftX[imuPointerFront];
		imuShiftYCur = imuShiftY[imuPointerFront];
		imuShiftZCur = imuShiftZ[imuPointerFront];
	  } 
	  else {//线性插值，计算点云点的速度，位移和欧拉角
		int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
		float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack])  / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
		float ratioBack = 1-ratioFront;
		imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
		imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
		if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) 
		  imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
		else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) 
		  imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
		else 
		  imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;

		imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
		imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
		imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

		imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
		imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
		imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
	  }
	  if (i == 0){//如果是第一个点,记住点云起始位置的速度，位移，欧拉角
		imuRollStart = imuRollCur;
		imuPitchStart = imuPitchCur;
		imuYawStart = imuYawCur;

		imuVeloXStart = imuVeloXCur;
		imuVeloYStart = imuVeloYCur;
		imuVeloZStart = imuVeloZCur;

		imuShiftXStart = imuShiftXCur;
		imuShiftYStart = imuShiftYCur;
		imuShiftZStart = imuShiftZCur;
	  } 
	  else {//计算之后每个点相对于第一个点的由于加减速非匀速运动产生的位移速度畸变，并对点云中的每个点位置信息重新补偿矫正
		ShiftToStartIMU(pointTime);
		VeloToStartIMU();
		TransformToStartIMU(&point);
	  }
	}
	laserCloudScans[scanID].push_back(point);    //将每个补偿矫正的点放入对应线号的容器
  }
  cloudSize = count;
  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {//将所有的点按照线号从小到大放入一个容器
	*laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  for (int i = 5; i < cloudSize - 5; i++) {//使用每个点的前后五个点计算曲率，因此前五个与最后五个点跳过
	float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
				+ laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
				+ laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
				+ laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
				+ laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
				+ laserCloud->points[i + 5].x;
	float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
				+ laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
				+ laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
				+ laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
				+ laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
				+ laserCloud->points[i + 5].y;
	float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
				+ laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
				+ laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
				+ laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
				+ laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
				+ laserCloud->points[i + 5].z;
	cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;   //平均曲率计算
	cloudSortInd[i] = i;  
	cloudNeighborPicked[i] = 0;
	cloudLabel[i] = 0;   //初始化为less flat点
	//计算每个scan的开始和末尾的索引，注意每个scan的开始和末尾5个点都不算
	if (int(laserCloud->points[i].intensity) != scanCount) {     	
	  scanCount = int(laserCloud->points[i].intensity);
	  if (scanCount > 0 && scanCount < N_SCANS) {
		scanStartInd[scanCount] = i + 5;
		scanEndInd[scanCount - 1] = i - 5;
	  }
	}
  }
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  //筛除容易被挡住的点、平行线点以及离群点
  for (int i = 5; i < cloudSize - 6; i++) {
	float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
	float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
	float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
	float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
	if (diff > 0.1) {
	  float depth1 = sqrt(pow(laserCloud->points[i].x ,2) + pow(laserCloud->points[i].y ,2)+pow(laserCloud->points[i].z ,2));
	  float depth2 = sqrt(pow(laserCloud->points[i+1].x ,2) + pow(laserCloud->points[i+1].y ,2)+pow(laserCloud->points[i+1].z ,2));
	  if (depth1 > depth2) {
		diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
		diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
		diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;
		//夹角比较小，点处在近似与激光束平行的斜面上，筛除
		if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//排除容易被斜面挡住的点
			//该点及前面五个点（大致都在斜面上）全部置为筛选过
		  cloudNeighborPicked[i - 5] = 1;
		  cloudNeighborPicked[i - 4] = 1;
		  cloudNeighborPicked[i - 3] = 1;
		  cloudNeighborPicked[i - 2] = 1;
		  cloudNeighborPicked[i - 1] = 1;
		  cloudNeighborPicked[i] = 1;
		}
	  } 
	  else {
		diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
		diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
		diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;
		if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
		  cloudNeighborPicked[i + 1] = 1;
		  cloudNeighborPicked[i + 2] = 1;
		  cloudNeighborPicked[i + 3] = 1;
		  cloudNeighborPicked[i + 4] = 1;
		  cloudNeighborPicked[i + 5] = 1;
		  cloudNeighborPicked[i + 6] = 1;
		}
	  }
	}
	float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
	float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
	float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
	float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;
	float dis = pow(laserCloud->points[i].x,2)+pow(laserCloud->points[i].y,2)+pow(laserCloud->points[i].z,2);

	//与前后点的平方和都大于深度平方和的万分之二，这些点视为离群点，包括陡斜面上的点，强烈凸凹点和空旷区域中的某些点，置为筛选过，弃用
	if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) 
	  cloudNeighborPicked[i] = 1;
  }

  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  //将每条线上的点分入相应的类别：边沿点和平面点
  for (int i = 0; i < N_SCANS; i++) {
	pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
	//将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点
	for (int j = 0; j < 6; j++) {
		//六等份起点：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
	  int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
	  //六等份终点：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
	  int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

	  //按曲率从小到大冒泡排序
	  for (int k = sp + 1; k <= ep; k++) {
		for (int l = k; l >= sp + 1; l--) {
		  if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
			int temp = cloudSortInd[l - 1];
			cloudSortInd[l - 1] = cloudSortInd[l];
			cloudSortInd[l] = temp;
		  }
		}
	  }

	  //挑选每个分段的曲率很大和比较大的点
	  int largestPickedNum = 0;
	  for (int k = ep; k >= sp; k--) {
		int ind = cloudSortInd[k];  
		//如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
		if (cloudNeighborPicked[ind] == 0 &&cloudCurvature[ind] > 0.1) {
		  largestPickedNum++;
		  if (largestPickedNum <= 2) {//挑选曲率最大的前2个点放入sharp点集合
			cloudLabel[ind] = 2;
			cornerPointsSharp.push_back(laserCloud->points[ind]);
			cornerPointsLessSharp.push_back(laserCloud->points[ind]);
		  } 
		  else if (largestPickedNum <= 20) {//挑选曲率最大的前20个点放入less sharp点集合
			cloudLabel[ind] = 1;
			cornerPointsLessSharp.push_back(laserCloud->points[ind]);
		  } 
		  else 
			break;
		  cloudNeighborPicked[ind] = 1;//筛选标志置位
		  //将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，防止特征点聚集，使得特征点在每个方向上尽量分布均匀
		  for (int l = 1; l <= 5; l++) {
			float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
			float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
			float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
			if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) 
			  break;
			cloudNeighborPicked[ind + l] = 1;
		  }
		  for (int l = -1; l >= -5; l--) {
			float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
			float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
			float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
			if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
			  break;
			}
			cloudNeighborPicked[ind + l] = 1;
		  }
		}
	  }

	  //挑选每个分段的曲率很小比较小的点
	  int smallestPickedNum = 0;
	  for (int k = sp; k <= ep; k++) {
		int ind = cloudSortInd[k];
		//如果曲率的确比较小，并且未被筛选出
		if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1) {
		  cloudLabel[ind] = -1;
		  surfPointsFlat.push_back(laserCloud->points[ind]);
		  smallestPickedNum++;
		  if (smallestPickedNum >= 4) //只选最小的四个，剩下的Label==0,就都是曲率比较小的
			break;
		  cloudNeighborPicked[ind] = 1;
		  for (int l = 1; l <= 5; l++) {
			float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
			float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
			float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
			if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) 
			  break;
			cloudNeighborPicked[ind + l] = 1;
		  }
		  for (int l = -1; l >= -5; l--) {
			float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
			float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
			float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
			if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) 
			  break;
			cloudNeighborPicked[ind + l] = 1;
		  }
		}
	  }
	  //将剩余的点（包括之前被排除的点）全部归入平面点中less flat类别中
	  for (int k = sp; k <= ep; k++) {
		if (cloudLabel[k] <= 0) 
		  surfPointsLessFlatScan->push_back(laserCloud->points[k]);
	  }
	}

	//由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
	pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
	pcl::VoxelGrid<PointType> downSizeFilter;
	downSizeFilter.setInputCloud(surfPointsLessFlatScan);
	downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
	downSizeFilter.filter(surfPointsLessFlatScanDS);

	//less flat点汇总
	surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //publich消除非匀速运动畸变后的所有的点
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  //publich消除非匀速运动畸变后的平面点和边沿点
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  //publich IMU消息,由于循环到了最后，因此是Cur都是代表最后一个点，即最后一个点的欧拉角，畸变位移及一个点云周期增加的速度
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //起始点欧拉角
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  //最后一个点的欧拉角
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  //最后一个点相对于第一个点的畸变位移和速度
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

//接收imu欧拉角和线加速度的消息
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

//a_sum=a_it - (R^i0_it)^-1 * g_i0            (R^i0_it)=Rz*Ry*Rx
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;     //循环移位效果，形成环形数组

//代表的是当前时刻IMU位姿和加速度
  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;   
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 2, laserCloudHandler);
  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 2);
  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2);
  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2);
  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2);
  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2);
  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();
  return 0;
}