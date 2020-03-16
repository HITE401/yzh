#include "pub_topic.h"

ros::Publisher  pub_image_track;
ros::Publisher pub_path;
ros::Publisher pub_odometry;
ros::Publisher pub_point_cloud;



void registerPub(ros::NodeHandle &n)
{
    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
}

//发布里程计和path信息
void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;

        static nav_msgs::Path path;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        Eigen::Vector3d tmp_T = estimator.Ps[WINDOW_SIZE];
        // printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.toSec(), tmp_T.x(), tmp_T.y(), tmp_T.z(), tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());

        //添加了输出文件
        ofstream fout(VINS_OUT_PATH,ios::app);
        fout.setf(ios::fixed,ios::floatfield);
        fout.precision(0);
        fout<<header.stamp.toSec()* 1e09<<",";
        fout.precision(5);
        fout << tmp_T.x() << ","
            << tmp_T.y() << ","
            << tmp_T.z() << ","
            << tmp_Q.w() << ","
            << tmp_Q.x() << ","
            << tmp_Q.y() << ","
            << tmp_Q.z() << ","<< endl;
            fout.close();

    }
}

void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag)
    {
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        // body frame
        Vector3d correct_t;
        Quaterniond correct_q;
        correct_t = estimator.Ps[WINDOW_SIZE];
        correct_q = estimator.Rs[WINDOW_SIZE];

        transform.setOrigin(tf::Vector3(correct_t(0), correct_t(1),  correct_t(2)));
        q.setW(correct_q.w());
        q.setX(correct_q.x());
        q.setY(correct_q.y());
        q.setZ(correct_q.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

        // camera frame
        transform.setOrigin(tf::Vector3(estimator.tic[0].x(), estimator.tic[0].y(), estimator.tic[0].z()));
        q.setW(Quaterniond(estimator.ric[0]).w());
        q.setX(Quaterniond(estimator.ric[0]).x());
        q.setY(Quaterniond(estimator.ric[0]).y());
        q.setZ(Quaterniond(estimator.ric[0]).z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));
    }
}

//发布当前窗口特征点组成的点云
void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);
}