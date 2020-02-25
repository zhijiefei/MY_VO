#ifndef DRAWER_H
#define DREWER_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/stitching.hpp>
#include<opencv2/features2d/features2d.hpp>
#include "Frame.h"

using namespace std;

using namespace cv;

class Drawer
{
    public:
    Drawer(){}
    
    //把CurrentRGB|LastRGB两幅同分辨率图像进行拼接，最终行数是不变的，列数×2，
    //这样方便让两幅图在同一窗口下显示并进行绘图操作
    cv::Mat Convine(const cv::Mat &CurrentRGB,const cv::Mat &LastRGB);

    //画两个特征点匹配的线
    //第一个参数是通过Convine函数两幅图像拼接之后的图像，最后一个参数是基于当前帧特征点顺序下存储每个特征点和上一帧匹配的特征点索引的容器
    void Drawingline(cv::Mat &imRGB,my_vo::Frame &LastFrame,my_vo::Frame &CurrentFrame,std::vector<int> &vpKeyPointMatches);
    
    //产生随机颜色参数
    Scalar randomColor( RNG& rng );

};




#endif