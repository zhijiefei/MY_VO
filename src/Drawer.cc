#include "Drawer.h"


//把两张图片进行拼接[imRGBa,imRGBb]
cv::Mat Drawer::Convine(const cv::Mat &CurrentRGB,const cv::Mat &LastRGB)
{
    cv::Mat a=CurrentRGB,b=LastRGB;
    cv::Mat combine;

    // hconcat函数：用于两个Mat矩阵或者图像的水平拼接。 
    // vconcat函数：用于两个Mat矩阵或者图像的垂直拼接。
    hconcat(a,b,combine);
    
    return  combine;

}

//当前帧和上一帧的匹配点画线
void Drawer::Drawingline(cv::Mat &imRGB,my_vo::Frame &LastFrame,my_vo::Frame &CurrentFrame,std::vector<int> &vpKeyPointMatches)
{
    RNG rng(0xFFFFFFFF);//随机数对象
    for(unsigned int i=0;i<vpKeyPointMatches.size();i++)
    {
        if(vpKeyPointMatches[i]<0)
        continue;

        Point pt1,pt2;
        pt1.x=CurrentFrame.mvKeysUn[i].pt.x;
        pt1.y=CurrentFrame.mvKeysUn[i].pt.y;
	//两幅图像左右拼接后纵坐标像素值（y）是不改变，改变只有横坐标值（x）
	//cols是图像的列数，同时也是横坐标方向
	//rows是图像的行数，同时也是纵坐标方向
        pt2.x=CurrentFrame.mGray.cols+LastFrame.mvKeysUn[vpKeyPointMatches[i]].pt.x;
        pt2.y=LastFrame.mvKeysUn[vpKeyPointMatches[i]].pt.y;
        //从第四个参数开始：颜色，线宽，线型
        line(imRGB,pt1,pt2,randomColor(rng),1,8);
    }
    
    imshow("KPsMatches",imRGB);
    waitKey(1);
}


Scalar Drawer::randomColor( RNG& rng )
  {
  int icolor = (unsigned) rng;
  return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
  }