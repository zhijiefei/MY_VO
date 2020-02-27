#include "MapPoint.h"

namespace my_vo 
{
  long unsigned int MapPoint::nNextId=0;
  //A第三个参数特征点的索引
  MapPoint::MapPoint(const cv::Mat &Pos, Frame *pFrame, const int &idxF)
  {
    Pos.copyTo(mWorldPos);
    
    mnId=nNextId++;
    
  }
  cv::Mat MapPoint::getWorldPos()
  {
      return mWorldPos.clone();
  }
  cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}
//返回观测到这个地图点的相机个数
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}
}
