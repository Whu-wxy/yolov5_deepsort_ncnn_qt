#ifndef YOLOV5_CUSTOMLAYER_H
#define YOLOV5_CUSTOMLAYER_H

#include "ncnn/net.h"
#include "ncnn/benchmark.h"
//#include "YoloV5.h"

#include "opencv2/opencv.hpp"//添加Opencv相关头文件
#include "ncnnmodelbase.h"

#include <QTime>

using namespace std;
using namespace cv;


namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}

typedef struct {
    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;


struct YoloObject {
//    cv::Rect_<float> rect;
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};


class YoloV5CustomLayer : public ncnnModelBase {
public:
    YoloV5CustomLayer(QObject *parent = 0);
    virtual ~YoloV5CustomLayer();

    virtual bool    predict(cv::Mat & frame);
    std::vector<YoloObject> detect(cv::Mat image, float threshold, float nms_threshold);
    std::vector<QString> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush"};
private:
    int input_size = 640;

//m model
//    std::vector<YoloLayerData> layers{
//            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},
//            {"492",    16, {{30,  61}, {62,  45},  {59,  119}}},
//            {"517",    32, {{116, 90}, {156, 198}, {373, 326}}},
//    };


    // s, n model
    std::vector<YoloLayerData> layers{
            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},
            {"375",    16, {{30,  61}, {62,  45},  {59,  119}}},
            {"400",    32, {{116, 90}, {156, 198}, {373, 326}}},
    };


//    std::vector<YoloLayerData> layers{
//            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},
//            {"376",    16, {{30,  61}, {62,  45},  {59,  119}}},
//            {"401",    32, {{116, 90}, {156, 198}, {373, 326}}},
//    };

public:


};


#endif //YOLOV5_CUSTOMLAYER_H
