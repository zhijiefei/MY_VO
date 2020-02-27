#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Frame.h"
#include"ORBVocabulary.h"
namespace my_vo
{
class ORBMatcher
{
    public:
    //使用默认参数，第一个参数用来作为最近邻和第二近邻进行配对独特性判断
    ////加上方向一致性检测时间消耗相差微小，但是误匹配明显减少
    ORBMatcher(float nnratio=0.6,bool CheckOrientation=true);
    ////F1是上一帧，F2是当前帧
    int SearchByBoW(Frame &F1,Frame &F2,std::vector< int> &vpKeyPointMatches);
    
    int DescriptorDistance(const cv::Mat &a,const cv::Mat &b);
    
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);
    
    public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;
    //是否进行方向检测
    bool mbCheckOrientation;

    private:

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    //这是用来作为最近邻和第二近邻进行配对独特性判断
    float mfNNratio;
  

};

}


#endif
