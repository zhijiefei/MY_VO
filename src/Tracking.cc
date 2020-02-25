#include "Tracking.h"

namespace my_vo 
{
  using namespace std;
  Tracking::Tracking( ORBVocabulary* pVoc,const string &strSettingPath, const int sensor):mState(NO_IMAGES_YET),mSensor(sensor),mpORBVocabulary(pVoc)
  
  {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    mVelocity=cv::Mat::eye(4,4,CV_32F);
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];//这里的K3校正系数有什么特殊之处？
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // 双目摄像头baseline * fx 50
    mbf = fSettings["Camera.bf"];//基线

    float fps = fSettings["Camera.fps"];//帧频率
    if(fps==0)
        fps=30;
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    // 1:RGB 0:BGR判断是什么类型的图片
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    // mpORBextractor=new my_vo::ORBextractor(1000,1.2,8,20,8);
    
       if(sensor==1)
    {
        // 深度相机disparity转化为depth时的因子，难道是深度转换因子？？
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }
    
  }
  
  
void Tracking::Track()
{
      // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }
   if(mState==NOT_INITIALIZED)
     StereoInitialization(); 
   else
   {
     bool bOK;
     
     
     if(mState==OK)
     {
       TicToc TrackWithMotionModeltime;
       bOK=TrackWithMotionModel();
       cout<<"the TrackWithMotionModeltime time:"<<TrackWithMotionModeltime.toc()<<endl;
     }
     
     if(bOK)
       UpdateMotionModel();
     
     mLastFrame = mCurrentFrame;
     
     cout<<"The Frame "<<mCurrentFrame.mnId<<" Tcw:"<<mCurrentFrame.mTcw<<endl;
   }
   
   
}

void Tracking::StereoInitialization()
{
   if(mCurrentFrame.N>500)
   {
    mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
    cout<<"the frist frame T:"<<mCurrentFrame.mTcw<<endl;
    for(int i=0;i<mCurrentFrame.N;i++)
    {
      float z=mCurrentFrame.mvDepth[i];
      if(z>0)
      {
	cv::Mat x3D=mCurrentFrame.UnprojectStereo(i);
	//std::cout<<x3D<<endl;
	MapPoint* NewPoint=new MapPoint(x3D,&mCurrentFrame,i);
	mCurrentFrame.mvpMapPoints[i]=NewPoint;
      }
    }
    mCurrentFrame.ComputeBoW();
    mState=OK;
    mLastFrame=mCurrentFrame;
    //std::cout<<"The Frame"<<mCurrentFrame.mnId<<"MapPoint："<<mCurrentFrame.mvpMapPoints.size()<<std::endl;
   }
}
//处理正常摄像头采集进来的图片，进行类型等转换
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
  cv::Mat mImGray=imRGB;
  cv::Mat imDepth=imD;
  
  CmRGB=imRGB;
  ORBMatcher matcher(0.7,true);
  if(mImGray.channels()==3)
  {
    if(mbRGB)
      cv::cvtColor(mImGray,mImGray,CV_RGB2GRAY);
    else
      cv::cvtColor(mImGray,mImGray,CV_BGR2BGRA);
  }
  
  if(mImGray.channels()==4)
  {
    if(mbRGB)
      cv::cvtColor(mImGray,mImGray,CV_RGBA2BGRA);
    else
      cv::cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
  }
  
      // 步骤2：将深度相机的disparity转为Depth
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
    mCurrentFrame=Frame(mImGray,imDepth,timestamp,mpORBVocabulary,mpORBextractorLeft,mbf,mK,mDistCoef);
    
    Track();
    
    LmRGB=CmRGB;
   
    return mCurrentFrame.mTcw.clone();
    
}

//利用运动模型进行前后两帧的跟踪
bool Tracking::TrackWithMotionModel()
{
  
  ORBMatcher matcher(0.9,true);
  TicToc UpdateLastFrametime;
  UpdateLastFrame();
  cout<<"the UpdateLastFrametime time:"<<UpdateLastFrametime.toc()<<endl;
  //这里是通过运动模型，（认为这两帧之间的相对运动和之前两帧间相对运动相同)估计当前帧的位姿
 // mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
  
  fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
  
  //int th=7;
  
  std::vector<int> vpKeyPointMatches;
  TicToc matchtime;
  int nmatches = matcher.SearchByBoW(mLastFrame,mCurrentFrame,vpKeyPointMatches);
  cout<<"the match time:"<<matchtime.toc()<<endl;
  TicToc drawertime;
  Drawer drawer;
     cv::Mat conbin=drawer.Convine(CmRGB,LmRGB);
    drawer.Drawingline(conbin,mLastFrame,mCurrentFrame,vpKeyPointMatches);
  cout<<"the drawer time:"<<drawertime.toc()<<endl;
  int MPs=0;
  vector<cv::Point3f> pts3d;
  vector<cv::Point2f> pts2d;
  cv::Mat rvec, tvec, inliers,Rcw;
  
  for(size_t i=0;i<vpKeyPointMatches.size();i++)
  {
    if(vpKeyPointMatches[i]>=0)
    {
    MapPoint * nMP=mLastFrame.mvpMapPoints[vpKeyPointMatches[i]];
    if(nMP)
    {
      const float z=mLastFrame.mvDepth[vpKeyPointMatches[i]];
      if(z>0)
      {
      cv::Mat x3D= mLastFrame.UnprojectStereo(vpKeyPointMatches[i]);
      pts3d.push_back(cv::Point3f(x3D.at<float>(0),x3D.at<float>(1),x3D.at<float>(2)));
      pts2d.push_back(cv::Point2f(mCurrentFrame.mvKeysUn[i].pt.x,mCurrentFrame.mvKeysUn[i].pt.y));
      MapPoint* pNewMP = new MapPoint(x3D,&mLastFrame,i);
      mCurrentFrame.mvpMapPoints[i]=pNewMP;
      MPs++;
      }
    }
    }
  }
  TicToc Optimizetime;
  cv::solvePnPRansac( pts3d, pts2d, mK, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
  cout<<"the optimize time:"<<Optimizetime.toc()<<endl;
  cv::Rodrigues(rvec,Rcw);
  cv::Mat Tcw= cv::Mat::eye(4,4,CV_32F);;
  Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
  tvec.copyTo(Tcw.rowRange(0,3).col(3));
  mCurrentFrame.SetPose(Tcw);
  cout<<"Pnp Ransac Tcw:"<<Tcw<<endl;
  std::cout<<"tracking MP number："<<MPs<<std::endl;
  
  if(nmatches<20)
  {
    cerr<<"matches points :"<<nmatches<<endl;
    return false;
  }
        
  // Optimize frame pose with all matches
    // 步骤3：优化位姿
  /*  Optimizer::PoseOptimization(&mCurrentFrame);
    
        // Discard outliers
    // 步骤4：优化位姿后剔除outlier的mvpMapPoints
    int nmatchesMap = 0;
    for(int i=0;i<mCurrentFrame.N;i++)
    {
      if(!mCurrentFrame.mvpMapPoints[i])
      {
	//外点容器未定义
	if(mCurrentFrame.mvbOutlier[i])
	{
	  mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
	  mCurrentFrame.mvbOutlier[i]=false;
	  nmatches--;
	}
	else
	  nmatchesMap++;
      }

    }
    cout<<"The inline points:"<<nmatchesMap<<endl;
   */
 cout<<"the CurrentFrame MP number:"<<mCurrentFrame.mvpMapPoints.size()<<endl;
    return mCurrentFrame.mvpMapPoints.size()>=10;
    
}

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points
    // 步骤1：对于双目或rgbd摄像头，单目不进行地图点添加，根据深度值为上一关键帧生成新的MapPoints
    // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
    // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
    // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
void Tracking::UpdateLastFrame()
{
  vector<pair<float,int>> vDepthIdx;
  vDepthIdx.resize(mLastFrame.N);
  for(int i=0;i<mLastFrame.N;i++)
  {
    float z=mLastFrame.mvDepth[i];
    if(z>0)
    {
      vDepthIdx.push_back(make_pair(z,i));
    }
    
  }
  if(vDepthIdx.empty())
  return;
  
  //按深度大小从小到大排列,把深度值前100个点进行地图点填充
  sort(vDepthIdx.begin(),vDepthIdx.end());
  
  //用来统计总共有多少个地图点
  int nPoints = 0;
  
  for(size_t i=0;i<vDepthIdx.size();i++)
  {
    int  n=vDepthIdx[i].second;
    bool bCreateNew = false;
    MapPoint * pMP=mLastFrame.mvpMapPoints[n];
    
    if(!pMP)
      bCreateNew=true;
    
    if(bCreateNew)
    {
      cv::Mat x3D=mLastFrame.UnprojectStereo(n);
      //因为MapPoint参数是Frame指针，加&就是地址，也就是指针了
      MapPoint * pNewMP=new MapPoint(x3D,&mLastFrame,n);
      mLastFrame.mvpMapPoints[n]=pNewMP;
      nPoints++;
    }
    
    //这里不想源码一样增加一个远近点判断
    if(nPoints>200)
      break;
  }
  
  cout<<"add the points number:"<<nPoints<<endl;
}

//在每一帧跟踪结束后去更新运动模型mVelocity，上一帧到当前帧的相对位姿
//
void Tracking::UpdateMotionModel()
{
  if(!mLastFrame.mTcw.empty())
  {
    cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
    cout<<"the Last frame Tcw:"<<endl<<mLastFrame.mTcw<<endl;
    cout<<"the Last frame Rwc:"<<endl<<mLastFrame.GetRotationInverse()<<endl;
    cout<<"the Last frame twc:"<<endl<<mLastFrame.GetCameraCenter()<<endl;
    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
    mVelocity=mCurrentFrame.mTcw*LastTwc;   //Tcw*Twl=Tcl
    cout<<"the mVelocity:"<<endl<<mVelocity<<endl;
  }
  else
    mVelocity=cv::Mat();
}


void Tracking::matchestest()
{
  ORBMatcher matcher(0.7,true);
  std::vector<int> vpKeyPointMatches;
  TicToc matchtime;
  int nmatches = matcher.SearchByBoW(mLastFrame,mCurrentFrame,vpKeyPointMatches);
  cout<<"the match time:"<<matchtime.toc()<<endl;
  TicToc drawertime;
  Drawer drawer;
     cv::Mat conbin=drawer.Convine(CmRGB,LmRGB);
    drawer.Drawingline(conbin,mLastFrame,mCurrentFrame,vpKeyPointMatches);
    cout<<"the drawertime time:"<<drawertime.toc()<<endl;
}



}