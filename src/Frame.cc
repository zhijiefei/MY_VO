#include  "Frame.h"

namespace my_vo
{
long unsigned int Frame::nNextId=0;   //因为nNextId是static变量，所以必须要在类的定义中进行赋值，在这个类被第一次创建的时候，或者说程序开始前就被赋值，之后不会再执行这条指令
double Frame::DTotaltimems=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;

    Frame::Frame(){}


    //Mat类型全用克隆，重新创建一个区域
    //词袋变量不进行复制，在进行词袋匹配是会进行重新计算（）()
    Frame::Frame(const Frame &frame)
    :
    mTimeStamp(frame.mTimeStamp),
    mpORBvocabulary(frame.mpORBvocabulary),
    mpORBextractorLeft(frame.mpORBextractorLeft),
    mK(frame.mK.clone()),
    mbf(frame.mbf),
    mb(frame.mb),
    mDistCoef(frame.mDistCoef.clone()),
    mGray(frame.mGray.clone()),
    N(frame.N),
    mvKeys(frame.mvKeys),
    mvKeysUn(frame.mvKeysUn),
    mvDepth(frame.mvDepth),
    mvuRight(frame.mvuRight),
    mDescriptors(frame.mDescriptors.clone()),
    mvpMapPoints(frame.mvpMapPoints),
    mnScaleLevels(frame.mnScaleLevels),
    mfScaleFactor(frame.mfScaleFactor)
    ,mfLogScaleFactor(frame.mfLogScaleFactor),
    mvScaleFactors(frame.mvScaleFactors),
    mvInvScaleFactors(frame.mvInvScaleFactors),
    mvLevelSigma2(frame.mvLevelSigma2)
    ,mvInvLevelSigma2(frame.mvInvLevelSigma2),
    mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
   mnId(frame.mnId)
   {
     
     if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
     
  }

    // Frame::Frame(const Frame &frame)
    // :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
    //  mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
    //  mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
    //  mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
    //  mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
    //  mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
    //  mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
    //  mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
    //  mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
    //  mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
    //  mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)

//clone()函数可以重新创建一个存储空间，让实参和类的成员变量相互独立
//extractor特征提取器应该在创建Frame对象时已经被创建
    Frame::Frame(const cv::Mat &imGray,const cv::Mat &imDepth,const double &timeStamp,ORBVocabulary *voc,ORBextractor* extractor, const float &bf,cv::Mat &K, cv::Mat &distCoef)
    :mTimeStamp(timeStamp),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mK(K.clone()),mbf(bf),mDistCoef(distCoef.clone()),mGray(imGray)
    {
        mnId=nNextId++;//从0开始计数，因为i++是赋值结束后才执行
        //直接从特征提取器中获取信息
        mnScaleLevels=mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        ExtractORB(imGray);
   
        N=mvKeys.size();
        if(mvKeys.empty())
        return;

        
    if(mbInitialComputations)
    {
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }
    mb = mbf/fx;
    
        UndistortKeyPoints();

        ComputeStereoFromRGBD(imDepth);
	//要提前给容器分配空间，要不然出现段错误
	mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	
	mvbOutlier = vector<bool>(N,false);
    }

    void Frame::ExtractORB(const cv::Mat &im)
    {
        double start=static_cast<double>(cvGetTickCount());
        //mpORBextractorLeft声明的时候是一个指针，加*号则是表示的是该指针下的地址
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
	//记录当前所有帧的特征提取所花的时间
	double time=((double)cvGetTickCount() - start) / cvGetTickFrequency();
         DTotaltimems += time;
        cout<<"The Frame ID "<<mnId<<"Extract ORB feature used time:"<<time/1000<<"ms"<<endl;
    }

    void Frame::UndistortKeyPoints()
    {
        if(mDistCoef.at<float>(0)==0.0)
        {
            mvKeysUn=mvKeys;
            return ;
        }

        cv::Mat mat(N,2,CV_32F);
        //把提取到的特征点转移到Mat容器中（N×2）
        for(int i=0;i<N;i++)
        {
            mat.at<float>(i,0)=mvKeys[i].pt.x;
            mat.at<float>(i,1)=mvKeys[i].pt.y;

        }
        //reshape有两个参数：

        //其中，参数：cn为新的通道数，如果cn = 0，表示通道数不会改变。

        //参数rows为新的行数，如果rows = 0，表示行数不会改变。

        //注意：新的行*列必须与原来的行*列相等。就是说，如果原来是5行3列，
        //新的行和列可以是1行15列，3行5列，5行3列，15行1列。仅此几种，否则会报错。

        // 调整mat的通道为2，矩阵的行列形状不变
        mat=mat.reshape(2);

        //调用opencv的矫正函数，校正后的也会存放在mat中
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);

        mat=mat.reshape(1);

        mvKeysUn.resize(N);

        for(int i=0;i<N;i++)
        {
            cv::KeyPoint kp=mvKeys[i];
            kp.pt.x=mat.at<float>(i,0);
            kp.pt.y=mat.at<float>(i,1);
            mvKeysUn[i]=kp;
        }
        

    }


    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
    {
        mvuRight=vector<float>(N,-1);
        mvDepth=vector<float>(N,-1);

        for(int i=0;i<N;i++)
        {
            //const cv::KeyPoint &kp=mvKeys[i];
            const cv::KeyPoint &kpU=mvKeysUn[i];

            // const float &v=kp.pt.y;
            // const float &u=kp.pt.x;

            const float &v1=kpU.pt.y;
            const float &u1=kpU.pt.x;
               //深度图信息为什么要从未纠正后的特征点进行提取
               //其中像素平面u是横坐标v是纵坐标，第v行第u列
            //const float d=imDepth.at<float>(v,u);
            const float d1=imDepth.at<float>(v1,u1);
	    //不能下面这样取，只能得到某一行的首地址
            // const unsigned char  *d2=imDepth.ptr<unsigned char>(v1)[u1];

            if(d1>0)
            {
                mvDepth[i]=d1;
                //从下面式子可以得出在此基线和焦距的条件下，得到的右图像像素坐标值
                //这仅仅是为了在后面最小二乘优化的过程中增加了一条约束
                //误差的类型就变成了三维的向量
                mvuRight[i]=kpU.pt.x-mbf/d1;

                // if(d1>0)
                // {
                //     cout<<"d:"<<d<<' '<<"d1:"<<d1<<endl;
                // }
            }

        }
    }


    void Frame::ComputeBoW()
    {
      
        if(mBowVec.empty())
        {
            //把矩阵形式的描述子转存为数组模式的存储方式，就相当于一长串的连续存储
            vector<cv::Mat> vCurrentDesc=Converter::toDescriptorVector(mDescriptors);
            //对mBowVec、mFeatVec两个变量的计算和填充,帮当前帧转化成词袋向量，用一个向量表示该帧
            mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        }
    }
    
    //该函数是反投影由像素坐标得到世界坐标,先得相机坐标->世界坐标
    cv::Mat Frame::UnprojectStereo(const int &i)
    {
      const float z=mvDepth[i];
      if(z>0)
      {
	const float u=mvKeysUn[i].pt.x;
	const float v=mvKeysUn[i].pt.y;
	const float X=(u-cx)*z/fx;
	const float Y=(v-cy)*z/fy;
	cv::Mat x3D=(cv::Mat_<float>(3,1)<<X,Y,z);
	return mRwc*x3D+mOw;
      }
      else
	return cv::Mat();
    }

    
    void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

/**
 * @brief Computes rotation, translation and camera center matrices from the camera pose.
 *
 * 根据Tcw计算mRcw、mtcw和mRwc、mOw
 */
void Frame::UpdatePoseMatrices()
{
    // [x_camera 1] = [R|t]*[x_world 1]，坐标为齐次形式
    // x_camera = R*x_world + t
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    // mtcw, 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系
    // mOw, 即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系
    mOw = -mRcw.t()*mtcw;
}
}

