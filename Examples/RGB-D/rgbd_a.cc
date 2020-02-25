
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
//#include "Drawer.h"
#include "Tracking.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/viz.hpp> 


using namespace std;
using namespace cv;
using namespace my_vo;




void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

//用来判断词典文件类型，不同类型文件用不同的词典加载函数
//find函数是查找调用它的字符串(str)中是否存在第一个参数suffix的字符串或者字符，
//并返回存在字符串str的第几位，找不到返回npos=-1
bool has_suffix(const std::string &str, const std::string &suffix) {     
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);//看看是否找到
}

int main(int argc,char **argv)
{
      if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }
 /*   
    ////////////////////////////////////
    	
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


    float bf=40.0;
  */  
    //要想词典能够正常工作，必须要加载训练好的词袋
     ORBVocabulary* mpVocabulary = new ORBVocabulary();   //创建一个词袋模型指针
     
     
      cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
         bool bVocLoad = false; // chose loading method based on file extension
     
      string strVocFile=string(argv[1]);
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
    
    ////////////////////////////////////
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
     string strAssociationFilename=string(argv[4]);
    LoadImages(strAssociationFilename,vstrImageFilenamesRGB,vstrImageFilenamesD,vTimestamps);
    
    int nImages=vstrImageFilenamesRGB.size();
    
    if(vstrImageFilenamesRGB.empty())
    {
      cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    
    if(vstrImageFilenamesRGB.empty()!=vstrImageFilenamesD.empty())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    
        //图像化，使用cv中的库
    ////////////////////////////////////////////////////
    cv::viz::Viz3d vis("Visual Odometry");                                //创建一个可视化窗口
    cv::viz::WCoordinateSystem world_coor(1.0) , camera_coor(0.5);        //创建坐标系,并给定尺寸
    cv::Point3d cam_pos(0,-1.0,-1.0),cam_focal_point(0,0,0),cam_y_dir(0,1,0);    //自定义相机在世界坐标系的三个坐标，第一个时相机整体在世界坐标系的坐标
                                                                                 //第二个是相机镜头对准的物体在世界坐标的位置，第三个是确定相机顶部朝向（正常方向）
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos,cam_focal_point,cam_y_dir);    //可以得到相机在世界坐标中的位姿
    
    vis.setViewerPose(cam_pose);                                                           //设置相机位姿在世界坐标系中  
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH,2.0);                              //第一个参数是线宽（）枚举类型
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH,1.0);
    vis.showWidget("world",world_coor);                     
    vis.showWidget("Camera",camera_coor);   
    
    /*            cv::Affine3d M(
            cv::Affine3d::Mat3( 
                0.99401367, -0.038789548, -0.10213833,
		0.022107767, 0.98692518, -0.15965563,
		0.10699586, 0.15644182, 0.98187464
            ), 
            cv::Affine3d::Vec3(
                0.12694818,0.066097863,-0.34077099
            )
        );

	
    vis.setWidgetPose("Camera",M);
     vis.spinOnce(1,false); */
    ///////////////////////////////////////////////////
    
    Tracking Tracker(mpVocabulary,string(argv[2]),1);
    cv::Mat imRGB,imD;
    for(int i=0;i<nImages;i++)
    {
      imRGB=cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[i],CV_LOAD_IMAGE_UNCHANGED);
      imD=cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[i],CV_LOAD_IMAGE_UNCHANGED);
      double tframe = vTimestamps[i];
      
      if(imRGB.empty())
      {
	cerr<< "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[i] << endl;
            return 1;
      }
      cout<<endl<<endl;
      TicToc trackframe;
      cv::Mat pose=Tracker.GrabImageRGBD(imRGB,imD,tframe);
     
      cout<<"frameTracke time:"<< trackframe.toc();
      pose=pose.inv();
             cv::Affine3d M(
            cv::Affine3d::Mat3( 
                 pose.at<float>(0,0),pose.at<float>(0,1),pose.at<float>(0,2),
			       pose.at<float>(1,0),pose.at<float>(1,1),pose.at<float>(1,2),
			       pose.at<float>(2,0),pose.at<float>(2,1),pose.at<float>(2,2)
            ), 
            cv::Affine3d::Vec3(
                pose.at<float>(0,3),pose.at<float>(1,3),pose.at<float>(2,3)
            )
        );
       
     // cv::waitKey(0);
	//     cout<<"the pose :"<<pose<<endl;
	 //    cout<<"the pose (0,0):"<<pose.at<float>(0,0)<<endl;
       vis.setWidgetPose("Camera",M);
       vis.spinOnce(1,false);   
    }
    
    
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
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

