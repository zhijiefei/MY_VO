#include <iostream>
#include <algorithm>
#include <fstream>
//#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "ORBextractor.h"
#include "Frame.h"
#include "ORBMatcher.h"
#include "Drawer.h"
#include  "timer.h"
using namespace std;
using namespace cv;
using namespace my_vo;
cv::Mat mK;
cv::Mat mDistCoef;
//class ORBextractor;

//用来判断词典文件类型，不同类型文件用不同的词典加载函数
//find函数是查找调用它的字符串(str)中是否存在第一个参数suffix的字符串或者字符，
//并返回存在字符串str的第几位，找不到返回npos=-1
bool has_suffix(const std::string &str, const std::string &suffix) {     
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);//看看是否找到
}

void LoadImages(const string &strAssociationFilename,vector<string> &vstrImageFilenamesRGB,
vector<string> &vstrImageFilenamesD,vector<double> &vTimestamps);
void  KindleRGB(const cv::Mat &im);//处理单张RGB图像，提取特征点，使用ORBSLAM中的特征点提取器
cv::Mat KindleORB(const cv::Mat &im);//处理单张RGB图像，提取特征点，直接用opencv的orb特征点提取代码
cv::Mat RGB2GRAY(const cv::Mat &im);//RGB转灰度图

//对帧类的特征提取进行测试
void FrameTest(const cv::Mat &imRGB,const cv::Mat &imDepth,const double &timeStamp,ORBVocabulary* voc,
my_vo::ORBextractor* extractor,const float &bf,cv::Mat &K, cv::Mat &distCoef);

//对前后帧匹配并绘图显示
void FrameMatche( Frame &CurFrame, Frame &LastFrame,const cv::Mat &CurrentRGB,const cv::Mat &LastRGB);


int main(int argc,char **argv)
{

    
	if(argc!=4)
	{
		cerr<<endl<<"Usage: ./rgbd_run  path_to_sequence path_to_association path_to_vocabulary"<<endl;
		return 1;
	}
    float fx = 517.306408;
    float fy = 516.469215;
    float cx = 318.643040;
    float cy = 255.313989;

    float k1 = 0.262383;
    float k2 = -0.953104;
    float p1 = -0.005358;
    float p2 = 0.002628;
    float k3 = 1.163314;
cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    //K.copyTo(mK);
mK=K;
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = k1;
    DistCoef.at<float>(1) = k2;
    DistCoef.at<float>(2) = p1;
    DistCoef.at<float>(3) = p2;
    const float k3_ = k3;
    if(k3_!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3_;
    }
   //DistCoef.copyTo(mDistCoef);
mDistCoef=DistCoef;

    float bf=40.0;
    
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    cv::Mat mDescriptors;
    string strAssociationFilename=string(argv[2]);
    LoadImages(strAssociationFilename,vstrImageFilenamesRGB,vstrImageFilenamesD,vTimestamps);
 



int nImages=vstrImageFilenamesRGB.size();

if(vstrImageFilenamesRGB.empty())
{
     cerr<<endl<<"No images found in provided path." << endl;
	 return 1;
}
else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
{
	cerr<<endl<<"Different number of images for rgb and depth." << endl;
	return 1;
}
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    
    //要想词典能够正常工作，必须要加载训练好的词袋
     ORBVocabulary* mpVocabulary = new ORBVocabulary();   //创建一个词袋模型指针
     
     
      cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
         bool bVocLoad = false; // chose loading method based on file extension
     
      string strVocFile=string(argv[3]);
      //判断字典是否是.txt类型
    if (has_suffix(strVocFile, ".txt"))
      //加载词典.txt类型
	  bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
      //判断字典是否是.bin类型
	else if(has_suffix(strVocFile, ".bin"))
	  bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
	else
	  bVocLoad = false;
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Failed to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;  
    
    
    
    
	cv::Mat imRGB,imDepth;
    cv::Mat CurRGB,LastRGB;
    Frame CurFrame,LastFrame;
	for(int ni=0;ni<nImages;ni++)
   {
     imRGB=imread(string(argv[1])+"/"+vstrImageFilenamesRGB[ni],-1);//以原始图像读取
     imDepth=imread(string(argv[1])+"/"+vstrImageFilenamesD[ni],-1);
	 double tframe=vTimestamps[ni];

	 if(imRGB.empty())
	 {
		 cerr<<endl<<"Failed to load image at: "<<string(argv[1])<<"/"<<vstrImageFilenamesRGB[ni]<<endl;
		 return 1;
	 }
     my_vo::ORBextractor * mpORBextractorLeft=new my_vo::ORBextractor(1000,1.2,8,20,8);
     
     
     
     CurRGB=imRGB.clone();

     cv::Mat mGray=RGB2GRAY(CurRGB);
     CurFrame=Frame(mGray,imDepth,tframe,mpVocabulary,mpORBextractorLeft,bf,mK,mDistCoef);
     

if(CurFrame.mnId==0)
{
   if(CurFrame.N>500)
   {
    CurFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
    cout<<"the frist frame T:"<<CurFrame.mTcw<<endl;
    for(int i=0;i<CurFrame.N;i++)
    {
      float z=CurFrame.mvDepth[i];
      if(z>0)
      {
	cv::Mat x3D=CurFrame.UnprojectStereo(i);
	//std::cout<<x3D<<endl;
	MapPoint* NewPoint=new MapPoint(x3D,&CurFrame,i);
	CurFrame.mvpMapPoints[i]=NewPoint;
      }
    }
   
    LastFrame=CurFrame;
    //std::cout<<"The Frame"<<mCurrentFrame.mnId<<"MapPoint："<<mCurrentFrame.mvpMapPoints.size()<<std::endl;
   }
}
     
    // CurFrame.ComputeBoW();
     if(ni==0)
     {
         LastFrame=CurFrame;//需要调用复制构造函数
         LastRGB=CurRGB.clone();
     }
     
     cv::Mat outimg1;
     
     drawKeypoints(CurRGB,CurFrame.mvKeysUn,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    
     imshow("FrameORB3",outimg1);
     
     FrameMatche(CurFrame,LastFrame,CurRGB,LastRGB);

     //FrameTest(imRGB,imDepth,tframe,mpVocabulary,mpORBextractorLeft,bf,mK,mDistCoef);

     //KindleRGB(imRGB);
     
     LastRGB=CurRGB.clone();
     
     LastFrame=CurFrame;
    cv::waitKey(0);
   }
  
    cout<<"ORB extract Median time :"<<Frame::DTotaltimems/(Frame::nNextId*1000)<<"ms"<<endl;
   



}






//测试Frame帧，同样实现mvKeys，mvKeysUn的对比
//mvKeysUn的点与图像边缘有一定的距离，特征点基本不会出现在图像边缘
void FrameTest(const cv::Mat &imRGB,const cv::Mat &imDepth,const double &timeStamp,ORBVocabulary* voc,
ORBextractor* extractor,const float &bf,cv::Mat &K, cv::Mat &distCoef)
{
    cv::Mat mGray;
    cv::Mat mRGB=imRGB;
    cv::Mat imD=imDepth;
    float mbf=bf;
    const double tframe=timeStamp;
    mGray=RGB2GRAY(mRGB);
    

    Frame mCurrentFrame(mGray,imD,tframe,voc,extractor,mbf,K,distCoef) ;
    
    cv::Mat outimg1,outimg2;
    drawKeypoints(mRGB,mCurrentFrame.mvKeys,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    drawKeypoints(mRGB,mCurrentFrame.mvKeysUn,outimg2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("FrameORB1",outimg1);
    imshow("UNFrameORB2",outimg2);
    //cout<<mCurrentFrame.mnId<<' ';
    cv::waitKey(0);
}

//RGB转灰度图
cv::Mat RGB2GRAY(const cv::Mat &im)
{
    cv::Mat mImGray=im;
    bool mbRGB=1;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);//转化成灰度图  头文件#include <opencv2/opencv.hpp>
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    } 

    return mImGray;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;   //stringstream可以轻易的进行类型转换
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

//直接使用Opencv的orb提取算法，由于没有八叉树对特征点均匀分布的优化，导致图像上的特征点会比较集中
cv::Mat KindleORB(const cv::Mat &im)
{
    cv::Mat mImGray=im;
    cv::Mat mImclor=im;
    std::vector<cv::KeyPoint> mvKeys;
    //std::vector<cv::KeyPoint> mvKeysUn;

    cv::Mat mDescriptors;

    Ptr<ORB> orb=ORB::create(1000,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
    orb->detect(mImclor,mvKeys);
    orb->compute(mImclor,mvKeys,mDescriptors);
    cv::Mat outimg;
    drawKeypoints(mImclor,mvKeys,outimg,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    // imshow("ORB1",outimg);
    // cv::waitKey(1);
    return outimg;
    
}


     //转化成灰度，然后进行特征提取，之后显示出来
     //做了opencv自带的特征提取算法和ORBslam改进的特征提取算法进行对比，可以看到增加八叉树进行特征点的均匀分布改进之后
     //特征点不会像opencv原带的算法提取的这么密集

void  KindleRGB(const cv::Mat &im)
{
    cv::Mat mImGray=im;
    cv::Mat mImclor=im;
    //cv::Mat mK , mDistCoef;
    //K.copyTo(mK);
    //DistCoef.copyTo(mDistCoef);
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    cv::Mat mDescriptors;
    bool mbRGB=1;
    my_vo::ORBextractor * mpORBextractorLeft=new my_vo::ORBextractor(1000,1.2,8,20,8);
       if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);//转化成灰度图  头文件#include <opencv2/opencv.hpp>
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    } 

    
    (*mpORBextractorLeft)(mImGray,cv::Mat(),mvKeys,mDescriptors);  //提取特征点

    int N = mvKeys.size();
    cv::Mat mat(N,2,CV_32F);//N×2矩阵
    for(int i=0;i<N;i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    mat=mat.reshape(2);

    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK); // 用cv的函数进行失真校正
    mat=mat.reshape(1);

    mvKeysUn.resize(N);//扩容存放矫正后特征点的容器

    for(int i=0;i<N;i++)
    {
        //mat存放的是之前提取的特征点新的，矫正过后的像素坐标值
        //但是特征点的索引序号没有发生改变，因此还是i
        //所以先从mvKey得到索引，在从mat得到新坐标，再放入mvKeysUn
        cv::KeyPoint kp=mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }



    cv::Mat outimg1,outimg2;
    drawKeypoints(mImclor,mvKeysUn,outimg2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
     drawKeypoints(mImclor,mvKeys,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);

     


      imshow("UNORB3",outimg2);

      imshow("ORB4",outimg1);
     cv::waitKey(0);
    // return outimg1;

}


void FrameMatche( Frame &CurFrame, Frame &LastFrame,const cv::Mat &CurrentRGB,const cv::Mat &LastRGB)
{
  //加上方向一致性检测时间消耗相差微小，但是误匹配明显减少
     ORBMatcher matcher(0.7,true);
     
     Drawer drawer;
     std::vector< int> vpKeyPointMatches;
     double start=static_cast<double>(cvGetTickCount());
     int nmatches=matcher.SearchByBoW(LastFrame,CurFrame,vpKeyPointMatches);
     cout<<CurFrame.mnId<<" matche to "<<LastFrame.mnId<<"  nmatches:"<<nmatches<<endl;
     cout<<"matche cost time:"<<(((double)cvGetTickCount() - start) / cvGetTickFrequency())/1000<<"ms"<<endl;
     cout<<"vpKeyPointMatches.size()="<<vpKeyPointMatches.size()<<endl;
     
     // cv::waitKey(0);
	//     cout<<"the pose :"<<pose<<endl;
	 //    cout<<"the pose (0,0):"<<pose.at<float>(0,0)<<endl;
     cv::Mat conbin=drawer.Convine(CurrentRGB,LastRGB);
    drawer.Drawingline(conbin,LastFrame,CurFrame,vpKeyPointMatches);


  vector<cv::Point3f> pts3d;
  vector<cv::Point2f> pts2d;
  cv::Mat rvec, tvec, inliers,Rcw;
  int MPs=0;
  for(size_t i=0;i<vpKeyPointMatches.size();i++)
  {
    if(vpKeyPointMatches[i]>=0)
    {
    MapPoint * nMP=LastFrame.mvpMapPoints[vpKeyPointMatches[i]];
    if(nMP)
    {
      const float z=LastFrame.mvDepth[vpKeyPointMatches[i]];
      if(z>0)
      {
      cv::Mat x3D= LastFrame.UnprojectStereo(vpKeyPointMatches[i]);
      pts3d.push_back(cv::Point3f(x3D.at<float>(0),x3D.at<float>(1),x3D.at<float>(2)));
      pts2d.push_back(cv::Point2f(CurFrame.mvKeysUn[i].pt.x,CurFrame.mvKeysUn[i].pt.y));
      MapPoint* pNewMP = new MapPoint(x3D,&LastFrame,i);
      CurFrame.mvpMapPoints[i]=pNewMP;
 MPs++;
      }
    }
    }
  }
  cout<<"MPs numbers:"<<MPs<<endl;
  TicToc Optimizetime;
  cv::solvePnPRansac( pts3d, pts2d, mK, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
  cout<<"the optimize time:"<<Optimizetime.toc()<<endl;
  cv::Rodrigues(rvec,Rcw);
  cv::Mat Tcw= cv::Mat::eye(4,4,CV_32F);;
  Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
  tvec.copyTo(Tcw.rowRange(0,3).col(3));
  CurFrame.SetPose(Tcw);
  cout<<"Pnp Ransac Tcw:"<<Tcw<<endl;

}

