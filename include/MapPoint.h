#ifndef  MAPPOINT_
#define  MAPPOINT_

#include "Frame.h"

#include<opencv2/core/core.hpp>

namespace my_vo
{
  class Frame;
  
  class MapPoint
  {
  public:
    MapPoint(const cv::Mat &Pos, Frame  *pFrame, const int &idxF);
    
  cv::Mat getWorldPos();

  private:
    
    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    
    
    cv::Mat mWorldPos; ///< MapPoint在世界坐标系下的坐标
    
  };
}

#endif