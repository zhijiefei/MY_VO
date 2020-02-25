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
}