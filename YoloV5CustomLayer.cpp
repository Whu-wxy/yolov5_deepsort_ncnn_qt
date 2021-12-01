#include "YoloV5CustomLayer.h"
#include "ncnn/layer.h"

void draw_objects(const cv::Mat& bgr, const std::vector<YoloObject>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const YoloObject& obj = objects[i];

        cv::rectangle(image, Rect(obj.x, obj.y, obj.w, obj.h), cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x;
        int y = obj.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}


inline float intersection_area(const YoloObject &a, const YoloObject &b) {
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y) {
        // no intersection
        return 0.f;
    }
    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

void qsort_descent_inplace(std::vector<YoloObject> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<YoloObject> &faceobjects) {
    if (faceobjects.empty()) {
        return;
    }
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(const std::vector<YoloObject> &faceobjects, std::vector<int> &picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++) {
        const YoloObject &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const YoloObject &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_proposals(const ncnn::Mat &anchors, int stride,
                        const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob,
                        float prob_threshold, std::vector<YoloObject> &objects) {
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                const float *featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
                    float score = featptr[5 + k];
                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold) {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    YoloObject obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


//yolov5n_ship-opt-fp16
YoloV5CustomLayer::YoloV5CustomLayer(QObject *parent)
    : ncnnModelBase("yolov5n_human-opt-fp16", parent)
{
//        // opt 需要在加载前设置
//        Net->opt.use_vulkan_compute = toUseGPU;  // gpu
       net.opt.use_fp16_arithmetic = true;  // fp16运算加速

}

YoloV5CustomLayer::~YoloV5CustomLayer()
{

}

bool YoloV5CustomLayer::predict(cv::Mat & frame)
{
    double ncnnstart = ncnn::get_current_time();

    QTime time;
    time.start();
    std::vector<YoloObject> boxes = detect(frame, 0.25, 0.45);
    qDebug()<<"QTime: "<<time.elapsed()<<"ms";
//    draw_objects(frame, boxes);

    for(YoloObject &boxInfo: boxes)
    {
//        qDebug()<<"boxInfo: "<<boxInfo.x1<<", "<<boxInfo.y1<<", "<<boxInfo.x2<<", "<<boxInfo.y2;
        rectangle(frame, Rect(boxInfo.x, boxInfo.y, boxInfo.w, boxInfo.h), Scalar(0, 255, 0), 2);
//        putText(frame, labels[boxInfo.label], Point(boxInfo.x1, boxInfo.y1), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 5);
    }
    double ncnnfinish = ncnn::get_current_time();
    double model_time = (double)(ncnnfinish - ncnnstart) / 1000;
    qDebug()<<"model_time: "<<model_time;
    putText(frame, to_string(model_time), Point(frame.cols/2, frame.rows/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
    return true;
}

//std::vector<YoloObject> YoloV5CustomLayer::detect(cv::Mat image, float threshold, float nms_threshold) {

//    // letterbox pad to multiple of 32
//    int w = image.cols;
//    int h = image.rows;
//    int width = w;
//    int height = h;
//    float scale = 1.f;
//    // 长边缩放到input_size
//    if (w > h) {
//        scale = (float) input_size / w;
//        w = input_size;
//        h = h * scale;
//    } else {
//        scale = (float) input_size / h;
//        h = input_size;
//        w = w * scale;
//    }

//    resize(image, image, Size(640,640));

//    cout<<"w: "<<image.cols<<endl;
//    cout<<"h: "<<image.rows<<endl;
////    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, 640,640);

//    ncnn::Mat in_net = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, 640,640);

//    // pad to target_size rectangle
//    // yolov5/utils/datasets.py letterbox
//    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
//    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
//    cout<<"wpad: "<<wpad<<endl;
//    cout<<"hpad: "<<hpad<<endl;
//    ncnn::Mat in_pad = in_net;
////    ncnn::copy_make_border(in_net, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

//    cout<<"in_pad w: "<<in_pad.w<<endl;
//    cout<<"in_pad h: "<<in_pad.h<<endl;

//    float mean[3] = {0, 0, 0};
//    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
//    in_pad.substract_mean_normalize(mean, norm);
//    auto ex = net.create_extractor();
//    ex.set_light_mode(true);
//    ex.set_num_threads(4);
//    ex.input("images", in_pad);

//    std::vector<YoloObject> proposals;
//    // anchor setting from yolov5/models/yolov5s.yaml

//    for (const auto &layer: layers) {
//        ncnn::Mat blob;
//        ex.extract(layer.name.c_str(), blob);
//        ncnn::Mat anchors(6);
//        anchors[0] = layer.anchors[0].width;
//        anchors[1] = layer.anchors[0].height;
//        anchors[2] = layer.anchors[1].width;
//        anchors[3] = layer.anchors[1].height;
//        anchors[4] = layer.anchors[2].width;
//        anchors[5] = layer.anchors[2].height;
//        std::vector<YoloObject> objectsx;
//        generate_proposals(anchors, layer.stride, in_pad, blob, threshold, objectsx);

//        proposals.insert(proposals.end(), objectsx.begin(), objectsx.end());
//    }

//    qDebug()<<"proposals: "<<proposals.size();

//    // sort all proposals by score from highest to lowest
//    qsort_descent_inplace(proposals);

//    // apply nms with nms_threshold
//    std::vector<int> picked;
//    nms_sorted_bboxes(proposals, picked, nms_threshold);

//    int count = picked.size();
//    qDebug()<<"nms picked: "<<count;

//    std::vector<YoloObject> objects;
//    objects.resize(count);
//    for (int i = 0; i < count; i++) {
//        objects[i] = proposals[picked[i]];

//        // adjust offset to original unpadded
//        float x0 = (objects[i].x - (wpad / 2)) / scale;
//        float y0 = (objects[i].y - (hpad / 2)) / scale;
//        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
//        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

//        // clip
//        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
//        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
//        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
//        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

//        objects[i].x = x0;
//        objects[i].y = y0;
//        objects[i].w = x1 - x0;
//        objects[i].h = y1 - y0;
//    }

//    return objects;

//}


std::vector<YoloObject> YoloV5CustomLayer::detect(cv::Mat image, float threshold, float nms_threshold) {

    // letterbox pad to multiple of 32
    int w = image.cols;
    int h = image.rows;
    int width = w;
    int height = h;
    float scaleW = 1.f;
    float scaleH = 1.f;

    scaleW = (float) input_size / w;
    scaleH = (float) input_size / h;


    resize(image, image, Size(input_size,input_size));

    cout<<"w: "<<image.cols<<endl;
    cout<<"h: "<<image.rows<<endl;
//    ncnn::Mat in_net = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, 640,640);

    ncnn::Mat in_net = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, input_size,input_size);

    float mean[3] = {0, 0, 0};
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_net.substract_mean_normalize(mean, norm);
    auto ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("images", in_net);

    std::vector<YoloObject> proposals;
    // anchor setting from yolov5/models/yolov5s.yaml

    for (const auto &layer: layers) {
        ncnn::Mat blob;
        ex.extract(layer.name.c_str(), blob);
        ncnn::Mat anchors(6);
        anchors[0] = layer.anchors[0].width;
        anchors[1] = layer.anchors[0].height;
        anchors[2] = layer.anchors[1].width;
        anchors[3] = layer.anchors[1].height;
        anchors[4] = layer.anchors[2].width;
        anchors[5] = layer.anchors[2].height;
        std::vector<YoloObject> objectsx;
        generate_proposals(anchors, layer.stride, in_net, blob, threshold, objectsx);

        proposals.insert(proposals.end(), objectsx.begin(), objectsx.end());
    }

    qDebug()<<"proposals: "<<proposals.size();

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    qDebug()<<"nms picked: "<<count;

    std::vector<YoloObject> objects;
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = objects[i].x / scaleW;
        float y0 = objects[i].y / scaleH;
        float x1 = (objects[i].x + objects[i].w) / scaleW;
        float y1 = (objects[i].y + objects[i].h) / scaleH;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1 - x0;
        objects[i].h = y1 - y0;
    }

    return objects;

}

