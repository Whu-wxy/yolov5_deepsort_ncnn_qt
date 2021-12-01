#ifndef DETECTORBASE_H
#define DETECTORBASE_H

#include <QObject>
#include <QTime>
#include <QDebug>
#include <QDir>
#include <QtGlobal>

#include "opencv2/opencv.hpp"//添加Opencv相关头文�
#include "ncnn/net.h"

using namespace std;
using namespace cv;

class ncnnModelBase : public QObject
{
    Q_OBJECT
public:
    ncnnModelBase(QObject *parent = 0);
    explicit ncnnModelBase(QString modelName, QObject *parent = 0);
    ~ncnnModelBase();

public:
    virtual bool    predict(cv::Mat & frame) = 0;
    bool    load(QString baseDir, QString modelName);
    bool    load(QString modelPath);
    void    unload();

    bool    hasLoadNet(){return bLoad; }

signals:

public slots:

protected:
    ncnn::Net net;
    //ncnn::Extractor* extractor;
    bool    bLoad;

    bool    moveFiles(QString modelName);
    void    printMat(const ncnn::Mat& m);
    Mat     resize_img(cv::Mat src, int long_size);
};


#endif // DETECTORBASE_H
