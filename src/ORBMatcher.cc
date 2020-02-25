#include "ORBMatcher.h"
#include "timer.h"
namespace my_vo
{

     const int ORBMatcher::TH_LOW=50;
     const int ORBMatcher::TH_HIGH=100;
    ORBMatcher::ORBMatcher(float nnratio,bool CheckOrientation):mbCheckOrientation(CheckOrientation),mfNNratio(nnratio){}
    

    //F1是上一帧，F2是当前帧
    int ORBMatcher::SearchByBoW(Frame &F1,Frame &F2,std::vector< int> &vpKeyPointMatches)
    {
      //其中词袋计算是耗时比较久的，大概6ms每帧，所以在常用的运动模型跟踪里面最好是避免要词袋模型匹配
      //用重投影匹配
      TicToc bowtime2;
   F1.ComputeBoW();
   F2.ComputeBoW();
	cout<<"the ComputeBoW2 time:"<<bowtime2.toc()<<endl;
    vector<int> rotHist[30];
    //把360度分成30份，用于建立方向直方图
    for(int i=0;i<30;i++)
        rotHist[i].reserve(500);
    const float factor = 30/360.0f;
         //这个容器是当前帧F2特征点顺序进行排列的
        vpKeyPointMatches=vector<int>(F2.N,-1);

        int nmatches=0;

        //分别创建两个关键帧的词袋特征向量索引
        //用于下面正向索引的遍历匹配
        DBoW2::FeatureVector::const_iterator F1it = F1.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator F2it = F2.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator F1end = F1.mFeatVec.end();
        DBoW2::FeatureVector::const_iterator F2end = F2.mFeatVec.end();
TicToc findmachetime;
        while(F1it!=F1end&&F2it!=F2end)
        {
            if(F1it->first==F2it->first)
            {
                //这得到的是该node节点下特征点索引
                const vector<unsigned int> vIndicesF1=F1it->second;
                const vector<unsigned int> vIndicesF2=F2it->second;

                for(size_t iF1=0;iF1<vIndicesF1.size();iF1++)
                {
                    const unsigned int realIdxF1=vIndicesF1[iF1];
                    //得到对应的描述子
                    const cv::Mat &dF1=F1.mDescriptors.row(realIdxF1);
                    
                    int bestDist1=256; // 最好的距离（最小距离）
                    int bestIdxF =-1 ;
                    int bestDist2=256; // 倒数第二好距离（倒数第二小距离）

                    for(size_t iF2=0;iF2<vIndicesF2.size();iF2++)
                    {
                        const unsigned int realIdxF2=vIndicesF2[iF2];
                         //如果当前匹配点已经匹配过则跳过
                        if(vpKeyPointMatches[realIdxF2]>0)
                        continue;
                        const cv::Mat &dF2=F2.mDescriptors.row(realIdxF2);

                        const int dist=DescriptorDistance(dF1,dF2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF = realIdxF2;//最优点索引
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    
                   
                  }
                   if(bestDist1<TH_LOW)
                    {
                        //如果符合区分度要求
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpKeyPointMatches[bestIdxF]=realIdxF1;
                            nmatches++;
			    //cout<<nmatches<<endl;
			    if(mbCheckOrientation)
			    {
			    float rot=F1.mvKeysUn[realIdxF1].angle-F2.mvKeysUn[bestIdxF].angle;
			    //经过下面输出显示，mvKeysUn，mvKeys两个容器中的特征点仅有像素坐标不一样外，其它
			    //信息是一样的，包括方向角、描述子计算区域
			    //cout<<F1.mnId<<"mvKeysUn"<<realIdxF1<<' '<<F1.mvKeysUn[realIdxF1].angle
			   // <<"    "<<"mvKeys"<<realIdxF1<<' '<<F1.mvKeys[realIdxF1].angle<<endl;
			    if(rot<0)
			      rot+=360.0;
			    int bin=round(rot*factor);
			    
			    if(bin==30)//rot=360度
			    bin=0;
			    //不满足则终止程序，如果不满足就会超出了这个rotHist的空间范围，会出现内存泄露，可能结果很严重所以终止
			    assert(bin>=0&&bin<30);
			    rotHist[bin].push_back(bestIdxF);
			    }
			    
                        }
                    }
                }
                
                
            F1it++;
            F2it++;
            }
            else if(F1it->first<F2it->first)
            {
                //找出vFeatVec1中第一个比f2it->first大或者等于的迭代器                              
                F1it=F1.mFeatVec.lower_bound(F2it->first);
            }
            else if(F1it->first>F2it->first)
            {
                F2it=F2.mFeatVec.lower_bound(F1it->first);
            }

        }
        
         cout<<"the findmachetime time:"<<findmachetime.toc()<<endl;
	 
	 TicToc CheckOrientationtime;
        if(mbCheckOrientation)
	{
	  cout<<"the CheckOrientationtime time:"<<CheckOrientationtime.toc()<<endl;
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
	
	ComputeThreeMaxima(rotHist,30,ind1,ind2,ind3);
	
	for(int i=0;i<30;i++)
	{
	  if(i==ind1||i==ind2||i==ind3)
	  continue;
	    for(int j=0,jend=rotHist[i].size();j<jend;j++)
	    {
	      vpKeyPointMatches[rotHist[i][j]]=-1;
	      nmatches--;
	    }
	  
	}
	
	}
        return nmatches;
    }

    int ORBMatcher::DescriptorDistance(const cv::Mat &a,const cv::Mat &b)
    {
            //分别创建了a，b  Mat类的首地址指针
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;
        //这里选取的特征点描述子位数是8*32=256位
        //这里就没32位进行一次比较，总共比较8次
        //下面每一次循环其实就是计算一对32位的二进制串的汉明距离方法，是作者参考相应文献得到的
        for(int i=0; i<8; i++, pa++, pb++)
        {
        //异或，相同为0相异为1
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }
    

    
    //把一个直方图的最长的三个方图序号找出来
    void ORBMatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
       //max123的范围是0到30
    int max1=0;//最大的行
    int max2=0;//第二大
    int max3=0;//第三大

    for(int i=0; i<L; i++)
    {
      //行的尺寸s
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
     }
    }

   int ORBMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th)
  {
    int nmatches=0;
    vector<int> rotHist[30];
    for(int i=0;i<30;i++)
      rotHist[i].reserve(500);
    const float factor=30/360.0f;
    const cv::Mat Rcw=CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw=CurrentFrame.mTcw.rowRange(0,3).col(3);
    
    const cv::Mat twc=-Rcw.t()*tcw;//正交矩阵的转置等于其逆
    
    const cv::Mat Rlw=LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw=LastFrame.mTcw.rowRange(0,3).col(3);
    
    const cv::Mat tlc=Rlw*twc+tlw;// Rlw*twc(w) = twc(l), twc(l) + tlw(l) = tlc(l)
    
    const bool bForward =tlc.at<float>(2)>CurrentFrame.mb;   //如果位置Z坐标（像极坐标系Z轴指向前）大于基线则便是前进
    const bool bBackward=-tlc.at<float>(2)>CurrentFrame.mb;
    
    
    for(int i=0;i<LastFrame.N;i++)
    {
      MapPoint * pMP=LastFrame.mvpMapPoints[i];
      if(pMP)
      {
	if(!LastFrame.mvbOutlier[i])
	{
	  cv::Mat x3Dw=pMP->getWorldPos();
	  cv::Mat x3Dc=Rcw*x3Dw+tcw;
	  
	  const float xc=x3Dc.at<float>(0);
	  const float yc=x3Dc.at<float>(1);
	  const float zc=x3Dc.at<float>(2);
	  
	  const float invzc=1/zc;
	  
	  //在摄像机后方的点剔除掉
	  if(invzc<0)
	    continue;
	  
	  float u=CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
	  float v=CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
	  
	  
	  if(u<CurrentFrame.mnMinX||u>CurrentFrame.mnMaxX)
	    continue;
	  if(v<CurrentFrame.mnMinY||v>CurrentFrame.mnMacX)
	    continue;
	  
	  int nLastOctave=LastFrame.mvKeys[i].octave;
	  
	  float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];
	  
	  
	}
      }
    }
    
    return 0;
  }
}