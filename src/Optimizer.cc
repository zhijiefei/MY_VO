#include "Optimizer.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

#include  "timer.h"
namespace my_vo 
{
  /**
 * @brief Pose Only Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param   pFrame Frame
 * @return  inliers数量
 */
//这里的位姿一元边优化为什么这么耗时
  int Optimizer::PoseOptimization(Frame *pFrame)
  {
    //1、创建优化器
    g2o::SparseOptimizer optimizer;
    //2、创建线求解器
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver=new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    //3、创建块求解器
    g2o::BlockSolver_6_3 * solver_ptr=new g2o::BlockSolver_6_3(linearSolver);
    
    g2o::OptimizationAlgorithmLevenberg *solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //4、给优化器赋予求解算法
    optimizer.setAlgorithm(solver);

    //有效边的数量
    int nInitialCorrespondences=0;    
    
    //创建顶点,
    //当前顶点也就是当前帧的位姿，一元边
    g2o::VertexSE3Expmap * vSE3=new g2o::VertexSE3Expmap();
    //设置初始值，四元素
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    //设置ID
    vSE3->setId(0);
    //设置是否参与优化
    vSE3->setFixed(false);
    //向优化器中添加顶点
    optimizer.addVertex(vSE3);
    
    //创建边，并向优化器添加边
    
    const int N=pFrame->N;
    
    //创建存放边的容器
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    //创建存放边索引的容器，索引对应了该帧地图点的索引
    vector<size_t>  vnIndexEdgeStereo;
    
    //给两个容器声明空间
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);
    
      
    const float deltaStereo = sqrt(7.815);
    for(int i=0;i<N;i++)
    {
      MapPoint * pMP=pFrame->mvpMapPoints[i];
      
      if(pMP)
      {
	nInitialCorrespondences++;
        pFrame->mvbOutlier[i]= false;
	Eigen::Matrix<double,3,1> obs;
	const cv::KeyPoint &kpUn=pFrame->mvKeysUn[i];
	const float &kp_ur=pFrame->mvuRight[i];
	obs<<kpUn.pt.x,kpUn.pt.y,kp_ur;//注意中间是用逗号不是<<
	
      g2o::EdgeStereoSE3ProjectXYZOnlyPose * e=new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
      //向边添加唯一的顶点
      e->setVertex(0,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
      e->setMeasurement(obs);
      //获取当前特征点的金字塔层对应的尺度因子平方的倒数
      const float invSigma2=pFrame->mvInvLevelSigma2[kpUn.octave];
      //设置信息矩阵

     Eigen::Matrix3d in= Eigen::Matrix3d::Identity()*invSigma2;
     e->setInformation(in);
      
      //创建核函数，防止误匹配对优化产生太大影响
      g2o::RobustKernelHuber* kt=new g2o::RobustKernelHuber();
      e->setRobustKernel(kt);
      kt->setDelta(deltaStereo);
      
      e->fx=pFrame->fx;
      e->fy=pFrame->fy;
      e->cx=pFrame->cx;
      e->cy=pFrame->cy;
      
      e->bf = pFrame->mbf;
      //给边添加初始值
      cv::Mat Xw=pMP->getWorldPos();
//      if(Xw.rows!=3)
//	continue;
      e->Xw[0]=Xw.at<float>(0);
      e->Xw[1]=Xw.at<float>(1);
      e->Xw[2]=Xw.at<float>(2);
      
      optimizer.addEdge(e);
      
      vpEdgesStereo.push_back(e);
      vnIndexEdgeStereo.push_back(i);//边的索引就是地图点的索引
      
      }
    }
    
      //边太少就取消这次优化
    if(nInitialCorrespondences<3)
        return 0;
    
        // 步骤4：开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数
    
    int nBad=0;
    
    for(size_t i=0;i<4;i++)
    {
    TicToc Optimizetime;
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    //对优化器进行初始化，只优化level是0的边
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[i]);
    cout<<"the optimize 1s used time:"<<Optimizetime.toc()<<endl;
    nBad=0;
    for(size_t j=0;j<vpEdgesStereo.size();j++)
    {
      g2o::EdgeStereoSE3ProjectXYZOnlyPose* e=vpEdgesStereo[j];
      const size_t idx=vnIndexEdgeStereo[j];
      
      if(pFrame->mvbOutlier[idx])
      {
	e->computeError();
      }
      
      const float chi2=e->chi2();
      
      if(chi2>chi2Stereo[i])
      {
	pFrame->mvbOutlier[idx]=true;
	e->setLevel(1);
	nBad++;
      }
      else
      {
	e->setLevel(0);
	pFrame->mvbOutlier[idx]=false;
	
      }
      
      if(i==2)
         e->setRobustKernel(0);//在第四次迭代的时候，0作为鲁棒核函数？？？那第四次迭代还有什么用
      
    }
    cout<<"the optimizer edges number:"<<optimizer.edges().size()<<endl;
     if(optimizer.edges().size()<10)
            break;
    }
    
    //把优化后的顶点取出来
    g2o::VertexSE3Expmap* vSE3_recov=static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    //得到优化后顶点的四元素位姿
    g2o::SE3Quat SE3quat_recov=vSE3_recov->estimate();
    //四元素转化成变换矩阵
    cv::Mat Pose= Converter::toCvMat(SE3quat_recov);//由四元素转化成矩阵
    
    pFrame->SetPose(Pose);
    
    return nInitialCorrespondences-nBad;
  }
  
  
 
  
  
  /*
  int Optimizer::PoseOptimization(Frame *pFrame)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位

    // 步骤1：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    //使用dense cholesky分解法。，求解Hのx=-g线性方程，由于H可能很大，直接求逆可能很困难，因此需要使用一些求解方法
    //LinearSolverCholmod ：使用sparse cholesky分解法。继承自LinearSolverCCS
    //LinearSolverCSparse：使用CSparse法。继承自LinearSolverCCS
    //LinearSolverPCG ：使用preconditioned conjugate gradient 法，继承自LinearSolver
    //LinearSolverDense ：使用dense cholesky分解法。继承自LinearSolver
    //LinearSolverEigen： 依赖项只有eigen，使用eigen中sparse Cholesky 求解，
    //因此编译好后可以方便的在其他地方使用，性能和CSparse差不多。继承自LinearSolver
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    //BlockSolver_6_3 ：表示pose 是6维，观测点是3维。用于3D SLAM中的BA
    //BlockSolver_7_3：在BlockSolver_6_3 的基础上多了一个scale
    //BlockSolver_3_2：表示pose 是3维，观测点是2维
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    // 步骤2：添加顶点：待优化当前帧的Tcw
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);//唯一的顶点
    vSE3->setFixed(false);//不固定该顶点，就是参与优化的意思
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    // for Monocular
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    //为什么没有for RGBD
    
    // for Stereo
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    // 步骤3：添加一元边：相机投影模型
    {
    

    //顶点只有一个，边有很多条
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            // 单目情况, 也有可能在双目下, 当前帧的左兴趣点找不到匹配的右兴趣点
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

		//因为整个图就只有一个顶点
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		//Measurement类型，也就是边定义的误差值类型，Vector2d
		//对初始测量值（图像的出来的值）进行赋值
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->getWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation 双目
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;// 这里和单目不同
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;// 这里和单目不同，深度摄像机kp_ur=kpU.pt.x-mbf/d，前提是d>0的点

                //EdgeStereoSE3ProjectXYZOnlyPose这个类型的边只有一个位姿顶点，且误差是3维的
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();// 这里和单目不同

		//因为顶点位姿只有一个，都是第0个顶点，那么每一条边连接的都是相同的位姿顶点，那就是第0个
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		//Measurement类型，也就是边定义的误差值类型，Vector3d
		//对初始测量值（图像的出来的值）进行赋值
                e->setMeasurement(obs);
		//该特征点被提取的金字塔层kpUn.octave
		//mvInvLevelSigma2该层的逆向尺度因子的平方
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
		//Eigen::Matrix3d::Identity()全1矩阵
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
		//typedef Matrix<double, D, D> InformationType
		//赋值的是信息矩阵，且其类型是上一行
                e->setInformation(Info);

		//核函数，替换以原先误差的二范数的代价函数
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->getWorldPos();
		//为每一条边添加初始值（地图点的世界坐标）
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);//边的索引就是地图点的索引
            }
        }

    }
    }


    //边太少就取消这次优化
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 步骤4：开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);// 对level为0的边进行优化，默认参数也是0
        optimizer.optimize(its[it]);//参数表填的是每次迭代的次数

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

	    //边的索引
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError(); // NOTE g2o只会计算active edge的误差
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);                 // 设置为outlier
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);                 // 设置为inlier
            }

            if(it==2)
                e->setRobustKernel(0); // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

	    //地图点的索引就是边的索引
	    //如果当前边连接的地图点是外点，直接进行误差计算（有什么用）
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            //chi2()返回的是_error.dot(information()*_error)
            const float chi2 = e->chi2();

	    //为什么通过这个判断来进行内外点的设置？
            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);//把当前边的level设为1，那么前面声明了，只优化level=0的边，那么这里的目的是把属于外点的边不参与下次优化
                nBad++;//坏边数量加1
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);//在第四次迭代的时候，0作为鲁棒核函数？？？那第四次迭代还有什么用
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();//得到优化之后的位姿
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);//由四元素转化成矩阵
    pFrame->SetPose(pose);//赋值新的位姿给当前帧

    return nInitialCorrespondences-nBad;//返回经过四次优化，剩下好的边数量
}
  
 */ 
  
}