#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "model.h"
#include "dataType.h"
#include "ncnn/net.h"
#include "ncnn/benchmark.h"
#include "ncnnmodelbase.h"

typedef unsigned char uint8;

using namespace std;
using namespace cv;

class DeepSort: public ncnnModelBase
{
public:
    DeepSort(QObject *parent = 0);
    ~DeepSort();

    bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);
    virtual bool    predict(cv::Mat & frame){ }

private:
	int feature_dim;
};
