#ifndef DETECTORBASE_H
#define DETECTORBASE_H

#include "opencv2/opencv.hpp"//添加Opencv相关头文�
#include "ncnn/net.h"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

class ncnnModelBase
{
public:
    ncnnModelBase();
    explicit ncnnModelBase(string modelName);
    ~ncnnModelBase();

public:
    virtual bool    predict(cv::Mat & frame) = 0;
    bool    load(string baseDir, string modelName);
    void    unload();

    bool    hasLoadNet(){return bLoad; }

protected:
    ncnn::Net net;
    //ncnn::Extractor* extractor;
    bool    bLoad;

    void    printMat(const ncnn::Mat& m);
    Mat     resize_img(cv::Mat src, int long_size);
};


#endif // DETECTORBASE_H
