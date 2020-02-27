#ifndef FEYFRAME_H
#define FEYFRAME_H

#include "Frame.h"
#include "MapPoint.h"
#include "KeyFrameDatabase.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include <mutex>

namespace my_vo
{
class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;
class KeyFrame
{
public:
  KeyFrame(Frame &F,Map *pMap ,KeyFrameDatabase *pKFDB);  
  
public:
        // nNextID名字改为nLastID更合适，表示上一个KeyFrame的ID号
    static long unsigned int nNextId;
    // 在nNextID的基础上加1就得到了mnID，为当前KeyFrame的ID号
    long unsigned int mnId;
    // 每个KeyFrame基本属性是它是一个Frame，KeyFrame初始化的时候需要Frame，
    // mnFrameId记录了该KeyFrame是由哪个Frame初始化的
    const long unsigned int mnFrameId;

    const double mTimeStamp;

    // Grid (to speed up feature matching)
    // 和Frame类中的定义相同
    // 前面两个是关键帧仅有的，这样直接可以承接Frame中   
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    //这两个宏定义，这样就不需要在关键帧类中再重新加上这个宏定
    //直接用于对当前帧进行48*64的网格划分，在GetFeaturesInArea函数中用到
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;
 
// Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    // 和Frame类中的定义相同
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec; ///< Vector of words to represent images
    DBoW2::FeatureVector mFeatVec; ///< Vector of nodes with indexes of local features

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;// 尺度因子，scale^n，scale=1.2，n为层数
    const std::vector<float> mvLevelSigma2;// 尺度因子的平方
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;
    
private:
    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;
        // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > > mGrid;
        // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;
    Map* mpMap;
};
}
#endif
