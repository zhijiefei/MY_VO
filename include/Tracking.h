#ifndef TRACKING_
#define TRACKING_

#include "Frame.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include  "MapPoint.h"
#include  "timer.h"
#include  "ORBMatcher.h"
#include  "Optimizer.h"
#include  "Drawer.h"

namespace my_vo
{
  class Tracking
  {
  public:
    Tracking( ORBVocabulary* pVoc,const string &strSettingPath, const int sensor);
    
    
// 输入左目RGB或RGBA图像和深度图
// 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    
    bool TrackWithMotionModel();
    
    void UpdateLastFrame();
    
    void UpdateMotionModel();
    void matchestest();
  public:
    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;//上一帧
    
        // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
  protected:
    void Track();
    // Map initialization for stereo and RGB-D
    void StereoInitialization();
    // Input sensor:MONOCULAR, STEREO, RGBD
    int mSensor;
  private:
    
    
    cv::Mat CmRGB;
    cv::Mat LmRGB;
        //BoW
    ORBVocabulary* mpORBVocabulary;
        //Calibration matrix
    cv::Mat mK;          //内参矩阵
    cv::Mat mDistCoef;   //畸变矫正系数
    float mbf;           //得到双摄像头baseline要用到的一个参数、 判断一个3D点远/近的阈值
    bool mbRGB;
    
        // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    //深度转换因子(尺度)，用来计算深度用的一个因子
    float mDepthMapFactor;
    
   my_vo::ORBextractor * mpORBextractorLeft;
   //my_vo::ORBextractor * mpORBextractor;
   cv::Mat mVelocity;//速度模型
   
   
   
   
  };
}

#endif