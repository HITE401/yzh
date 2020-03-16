#include"estimator.h"
#include "pub_topic.h"  //nb！


bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}


Estimator::Estimator() 
{
    clearState();
}

void Estimator::processMeasurements()
{
	while(1)
	{
		if(!featureBuf.empty())
		{
			pair<double, map<int, vector<Eigen::Matrix<double, 7, 1>>>> feature;
			mBuf.lock();
            feature = featureBuf.front();
			featureBuf.pop();
			mBuf.unlock();
            mProcess.lock();
			processImage(feature.second);
			
			std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubPointCloud(*this, header);
           pubTF(*this, header);
            mProcess.unlock();
		}
        break;
	}
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!featureBuf.empty())
        featureBuf.pop();
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
    }
    for (int i = 0; i < 2; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }
    frame_count = 0;
    solver_flag = 0;
    f_manager.clearState();
    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < 2; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    mProcess.unlock();
}

void Estimator::inputImage(double time, cv::Mat &imgleft, cv::Mat &imgright)
{
	map<int, vector<Eigen::Matrix<double, 7, 1>>> featureFrame;
	featureFrame=featureTracker.trackImage(time,imgleft,imgright);
     if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.imTrack;
        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);
        sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
        pub_image_track.publish(imgTrackMsg);
    }

    mBuf.lock();
    featureBuf.push(make_pair(time,featureFrame));
    mBuf.unlock();
    processMeasurements();
}

void Estimator::inputIMU(double time, Vector3d acc, Vector3d gyr)
{
	return;
}

void Estimator::processImage(const map<int, vector< Eigen::Matrix<double, 7, 1>>> &image)
{
	if(f_manager.addFeatureCheckParallax(frame_count,image))   //返回1表示视差大，要去掉最旧帧
	{	
        marginalization_flag=0;
        //ROS_INFO("delte old");
    }
	else
	{
        marginalization_flag=1;
        //ROS_INFO("delte last new");
    }

	if(solver_flag==0)
	{
		f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
		optimization();
		if(frame_count==WINDOW_SIZE)
		{
			solver_flag=1;
			slideWindow();
            ROS_INFO("Initialization finish!");
		}
		else
		{
		    frame_count++;
            //给个初值，防止pnp失败导致位姿误差过大
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
		}
	}
	else
	{
		f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
		optimization();
		set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);    //去除深度估计误差较大的点
		
        //featureTracker.removeOutliers(removeIndex);
        //predictPtsInNextFrame();
        
        slideWindow();
        f_manager.removeFailures();
	}
}


void Estimator::slideWindow()
{
    if (marginalization_flag == 0)
    {
		Matrix3d back_R0 = Rs[0];
        Vector3d back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

			Matrix3d R0, R1;
			Vector3d P0, P1;
			R0 = back_R0 * ric[0];
			R1 = Rs[0] * ric[0];
			P0 = back_P0 + back_R0 * tic[0];
			P1 = Ps[0] + Rs[0] * tic[0];
			f_manager.removeBackShiftDepth(R0, P0, R1, P1);

        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
           	f_manager.removeFront(frame_count);
        }
    }
}


void Estimator::outliersRejection(set<int> &removeIndex)
{
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
            }
            if(it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
				double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
													Rs[imu_j], Ps[imu_j], ric[1], tic[1],
													depth, pts_i, pts_j_right);
				err += tmp_error;
				errCnt++;
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();
    }
    for (int i = 0; i < 2; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }
    VectorXd dep = f_manager.getDepthVector();    //获得特征点的逆深度
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
}


void Estimator::double2vector()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
}



void Estimator::optimization()
{
    vector2double();  //变成c++可识别的数据类型，方便ceres优化库优化变量
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function=new ceres::HuberLoss(1.0);     // δ=1.0
    //对位姿变量做流形上的处理(过参数化)
    for(int i=0; i<=frame_count; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], 7, local_parameterization);
    }
    //后期加上imu，下句代码要删掉    
    problem.SetParameterBlockConstant(para_Pose[0]);  //为什么不用imu就要把最前帧设为常数？？？

    for(size_t i=0; i< 2; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i],7, local_parameterization);
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);    //外参设为常数
    }

    //添加边缘化先验项，注意先验项没有loss函数
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for(auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num =it_per_id.feature_per_frame.size();
        if(it_per_id.used_num< 4)
            continue;
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
         Vector3d pts_i = it_per_id.feature_per_frame[0].point;


        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            if( it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)  
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index]);
                }
            }
            f_m_cnt++;
        }
    }
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);   //计算有多少条边

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    if (marginalization_flag == 0)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    double2vector();

    if(frame_count < WINDOW_SIZE)     //双目初始化的时候也会做优化
        return;

    //开始计算边缘化的先验信息，用于给下次优化时使用
    if (marginalization_flag == 0)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();   //为啥还要再变一次，难道需要优化？？

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (size_t i = 0; i < last_marginalization_parameter_blocks.size(); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (it_per_id.used_num < 4)
                continue;
            feature_index++;
            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if(imu_i != imu_j)
                {
                    Vector3d pts_j = it_per_frame.point;
                    ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                    vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                    vector<int>{0, 3});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
                if(it_per_frame.is_stereo)
                {
                    Vector3d pts_j_right = it_per_frame.pointRight;
                    if(imu_i != imu_j)
                    {
                        ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index]},
                                                                                        vector<int>{0, 4});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                        vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index]},
                                                                                        vector<int>{2});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }
        marginalization_info->preMarginalize();
        marginalization_info->marginalize();

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
        for (int i = 0; i < 2; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info && count(begin(last_marginalization_parameter_blocks),end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (size_t i = 0; i < last_marginalization_parameter_blocks.size(); i++)
                {
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            marginalization_info->preMarginalize();
            marginalization_info->marginalize();
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                else
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
            }
            for (int i = 0; i < 2; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
}