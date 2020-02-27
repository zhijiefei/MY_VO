#ifndef  MAPPOINT_
#define  MAPPOINT_

#include "Frame.h"

#include<opencv2/core/core.hpp>
#include<mutex>
namespace my_vo
{
  class Frame;
  
  class MapPoint
  {
  public:
    MapPoint(const cv::Mat &Pos, Frame  *pFrame, const int &idxF);
    
  cv::Mat getWorldPos();

  cv::Mat GetDescriptor();
  
  int Observations();
  private:
    
    long unsigned int mnId; ///< Global ID for MapPoint
    static long unsigned int nNextId;
    
    
    //代表的是观测到这个地图点的相机个数
    int nObs;
    cv::Mat mWorldPos; ///< MapPoint在世界坐标系下的坐标
    
    // Best descriptor to fast matching
    // 每个3D点也有一个descriptor
    // 如果MapPoint与很多帧图像特征点对应（由keyframe来构造时），那么距离其它描述子的平均距离最小的描述子是最佳描述子
    // MapPoint只与一帧的图像特征点对应（由frame来构造时），那么这个特征点的描述子就是该3D点的描述子
    cv::Mat mDescriptor; ///< 通过 ComputeDistinctiveDescriptors() 得到的最优描述子
    
    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
  };
}

#endif
