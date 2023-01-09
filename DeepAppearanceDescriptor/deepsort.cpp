#include "DeepSort.h"
#include <iostream>

DeepSort::DeepSort() : ncnnModelBase("deepsort_sim-opt-fp16")
{
    feature_dim = 512;
}

DeepSort::~DeepSort() {

}


bool DeepSort::getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
	std::vector<cv::Mat> mats;
	for(DETECTION_ROW& dbox : d) {
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols-rc.x));
		rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));

		cv::Mat mattmp = img(rc).clone();
		cv::resize(mattmp, mattmp, cv::Size(64, 128));
		mats.push_back(mattmp);
	}
    int count = mats.size();

    float norm[3] = {0.229, 0.224, 0.225};
    float mean[3] = {0.485, 0.456, 0.406};
    for (int i=0; i<count; i++)
    {
        ncnn::Mat in_net = ncnn::Mat::from_pixels(mats[i].data, ncnn::Mat::PIXEL_BGR2RGB, 64, 128);
        in_net.substract_mean_normalize(mean, norm);

        ncnn::Mat out_net;
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(4);
    //    if (toUseGPU) {  // 消除提示
    //        ex.set_vulkan_compute(toUseGPU);
    //    }
        ex.input("input", in_net);
        ex.extract("output", out_net);

        cv::Mat tmp(out_net.h, out_net.w, CV_32FC1, (void*)(const float*)out_net.channel(0));
        const float* tp = tmp.ptr<float>(0);
        for(int j = 0; j < feature_dim; j++)
        {
            d[i].feature[j] = tp[j];
        }

    }
	return true;
}
