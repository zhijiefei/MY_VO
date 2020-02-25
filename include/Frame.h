#ifndef FRAME_H
#define FRAME_H
#include <ORBextractor.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include"ORBVocabulary.h"
#include"Converter.h"
#include "MapPoint.h"
using namespace std;
namespace my_vo
{
  
  class MapPoint;
    class Frame
    {
        public:
        Frame();
       
       //帧的复制构造函数
       Frame(const Frame &frame);

        Frame(const cv::Mat &imGray,const cv::Mat &imDepth,const double &timeStamp,ORBVocabulary *voc,
        ORBextractor* extractor, const float &bf,cv::Mat &K,cv::Mat &distCoef);
        //括号中的成员变量的赋值顺序要和它的声明顺序一致，否则会出现警告： warning: ‘my_vo::Frame::mTimeStamp’ will be initialized after [-Wreorder]

        //在构造函数中会被调用
        void ExtractORB(const cv::Mat &im);
        
        void ComputeBoW();

        //这里有一个问题：深度相机的基线是代表的是什么，为什么会有基线？
        //该函数要做的就是根据已知的基线和焦距fx把视差求出来，再进一步得出根据该基线右图像（可能是深度图）中所有特征点的横坐标ur
        void ComputeStereoFromRGBD(const cv::Mat &imDepth);

	
	// Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
       cv::Mat UnprojectStereo(const int &i);
	
	
	void SetPose(cv::Mat Tcw);
	void UpdatePoseMatrices();
	
	inline cv::Mat GetCameraCenter()
	{
	  return mOw.clone();
	}
	inline cv::Mat GetRotationInverse()
	{
	  return mRwc.clone();
	}
	
	
        public:
        
        
        double mTimeStamp;

        ORBVocabulary* mpORBvocabulary;

        ORBextractor* mpORBextractorLeft;

        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;

        //对于双目相机来说：这是基线×fx的值
        //对于RGBD相机来说：这是IR投影设备与单目相机的基线×fx的值
        float mbf;
        //对于立体相机来说是两个相机的基线
        //RGBD来说，摄像头和IR设备的基线
        //其中两个种情况都可以通过mbf/fx得到
        float mb;

        cv::Mat mDistCoef;   //校正系数矩阵

        cv::Mat mGray;
        
        int N;               //特征点数量

        //存放特征点的容器
        std::vector<cv::KeyPoint> mvKeys;
        std::vector<cv::KeyPoint> mvKeysUn;
        //特征点对应的深度信息
        std::vector<float> mvDepth;
        std::vector<float> mvuRight;
        //特征点的描述子
        cv::Mat mDescriptors;
	
    // 每个特征点对应的MapPoint
     std::vector<MapPoint*> mvpMapPoints;
     
         // 外点
    std::vector<bool> mvbOutlier;
            // Scale pyramid info.
            //在特征提取器被创建时，它的构造函数中已经进行了这些参数的计算，都已经计算出来，在构造函数中进行填充就可以
        int mnScaleLevels;//图像提金字塔的层数
        float mfScaleFactor;//图像提金字塔的尺度因子
        float mfLogScaleFactor;//
        vector<float> mvScaleFactors;
        vector<float> mvInvScaleFactors;
        vector<float> mvLevelSigma2;
        vector<float> mvInvLevelSigma2;
        
	//在每帧初始被创建的时候这是默认是true，该变量是为了在初始创建帧的时候进行对当前帧的固有性质进行保护（图像边界点，中心点的纠正
	//，图像边缘尺寸，相机内参）在帧被创建后不能再进行对相关参数进行修改，因此进行保护
        static bool mbInitialComputations;
        //词袋变量
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec; 


        // Camera pose.
        cv::Mat mTcw; //< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵
        
        //用来记录当前帧和下一帧的ID
        static long unsigned int nNextId;
        long unsigned int mnId;

        //每一帧ORB特征提取花费的时间
        static double DTotaltimems;
        private:
        //矫正特征点,会在构造函数中调用
        void UndistortKeyPoints();
    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world
    };
}

#endif