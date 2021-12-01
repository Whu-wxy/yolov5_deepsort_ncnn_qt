#include <QString>
#include <QDebug>
#include <QFile>
#include <iostream>

#include "YoloV5CustomLayer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>
#include "./DeepAppearanceDescriptor/deepsort.h"

#include "KalmanFilter/tracker.h"
#include "ncnn/net.h"

using namespace std;
using namespace cv;
using namespace ncnn;


//Deep SORT parameter
const int nn_budget=100;
const float max_cosine_distance=0.2;
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const  std::vector<YoloObject>& out,   DETECTIONS& d);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d);

int main(int argc, char** argv)
{
    //deep SORT
    tracker mytracker(max_cosine_distance, nn_budget);

    // Open a video file or an image file or a camera stream.
    std::string str, outputFile;
    cv::VideoCapture cap;
    cv::VideoWriter video;
    cv::Mat frame, blob;

    YoloV5CustomLayer net;
    DeepSort deepsort;

    outputFile = "D:\\QtWork\\DeepSORT-master\\src\\deep_sort2.avi";

    // Open the video file      anquanmao.mp4   human.flv    ship.mp4
    bool readRes = cap.open("D:\\QtWork\\DeepSORT-master\\src\\anquanmao.mp4");  //ship.mp4
    cout<<"read: "<<readRes<<endl;
    // Get the video writer initialized to save the output video
    bool writeRes = video.open(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), 12.0,
               cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
    cout<<"write: "<<writeRes<<endl;

    // Create a window
    static const  std::string kWinName = "Multiple Object Tracking";
    namedWindow(kWinName, cv::WINDOW_NORMAL);


    int count = 0;
    // Process frames.
    while (readRes)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty())
        {
            std::cout << "Done processing !!!" <<  std::endl;
            std::cout << "Output file is stored as " << outputFile <<  std::endl;
            cv::waitKey(3000);
            break;
        }

        std::vector<YoloObject> outs = net.detect(frame, 0.25, 0.45);
        DETECTIONS detections;
        postprocess(frame, outs,detections);

        std::cout<<"Detections size:"<<detections.size()<<std::endl;
        if(deepsort.getRectsFeature(frame, detections))
        {
            QTime time;
            time.start();
            mytracker.predict();
            mytracker.update(detections);
            qDebug()<<"QTime: "<<time.elapsed()<<"ms";

            std::vector<RESULT_DATA> result;
            for(Track& track : mytracker.tracks) {
                if(!track.is_confirmed() || track.time_since_update > 1) continue;
                result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            }
            for(unsigned int k = 0; k < detections.size(); k++)
            {
                DETECTBOX tmpbox = detections[k].tlwh;
                cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
//                cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
                // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B

                for(unsigned int k = 0; k < result.size(); k++)
                {
                    DETECTBOX tmp = result[k].second;
                    cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                    rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

                    std::string label = cv::format("%d", result[k].first);
                    cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
                }
            }
        }

        // Write the frame with the detection boxes
        cv::Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        count++;
        cout<<"frame count: "<<count<<endl;

//        imshow(kWinName, frame);
        waitKey(300);  //延时20ms
    }

    cap.release();
    video.release();

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const  std::vector<YoloObject>& outs,DETECTIONS& d)
{
    for(const YoloObject &info: outs)
    {
        //目标检测 代码的可视化
//        drawPred(info.label, info.prob, info.x, info.y,info.w+info.x, info.y+info.h, frame);

        get_detections(DETECTBOX(info.x, info.y,info.w,  info.h),
                       info.prob,d);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    std::string label  = to_string(classId);

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}
