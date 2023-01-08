#include "ncnnModelBase.h"

ncnnModelBase::ncnnModelBase()
{
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = ncnn::get_gpu_count() > 0;
#endif
}

ncnnModelBase::ncnnModelBase(string modelName)
{
    bLoad = false;
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = ncnn::get_gpu_count() > 0;
#endif

    load(dataDir, modelName);
}

ncnnModelBase::~ncnnModelBase()
{
    unload();
}


bool ncnnModelBase::load(string baseDir, string modelName)
{
    unload();

    cout<<"[NCNN] model load begin"<<endl;

    string binFile = baseDir + "//" + modelName + ".bin";
    string paramFile = baseDir + "//" + modelName + ".param";

    try {
        int res = net.load_param(paramFile.c_str());
        cout<<"[NCNN] param consumed: "<<res;

        int consumed = net.load_model(binFile.c_str());
        cout<<"[NCNN] bin consumed: "<<consumed<<endl;

        cout<<"[NCNN] model loaded: "<<binFile<<endl;
        bLoad = true;
    }
    catch (...) {
        cout<<"[NCNN] model loaded failed. "<<endl;
    }

    return bLoad;
}

void ncnnModelBase::unload()
{
    if(!bLoad) return;

    bLoad = false;
    net.clear();
}


void ncnnModelBase::printMat(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(5);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

Mat ncnnModelBase::resize_img(cv::Mat src, int long_size)
{
    int w = src.cols;
    int h = src.rows;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)long_size / w;
        w = long_size;
        h = h * scale;
    }
    else
    {
        scale = (float)long_size / h;
        h = long_size;
        w = w * scale;
    }
    if (h % 32 != 0)
    {
        h = (h / 32 + 1) * 32;
    }
    if (w % 32 != 0)
    {
        w = (w / 32 + 1) * 32;
    }
    // std::cout<<"缂╂斁灏哄 (" << w << ", "<<h<<")"<<std::endl;
    cv::Mat result;
    cv::resize(src, result, cv::Size(w, h));
    return result;
}
