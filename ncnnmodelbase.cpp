#include "ncnnModelBase.h"

ncnnModelBase::ncnnModelBase(QObject *parent) : QObject(parent)
{
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = ncnn::get_gpu_count() > 0;
#endif
}

ncnnModelBase::ncnnModelBase(QString modelName, QObject *parent) : QObject(parent)
{
    bLoad = false;
#if NCNN_VULKAN
    net.opt.use_vulkan_compute = ncnn::get_gpu_count() > 0;
#endif

    bool res = moveFiles(modelName);
    if(res)
    {
        QString dataDir;
#ifdef Q_OS_ANDROID
        AndroidSetup setup;
        dataDir = setup.getAppDataDir();
#else
        dataDir = "D:\\QtWork\\DeepSORT-master\\src";
#endif

        load(dataDir, modelName);
    }
}

ncnnModelBase::~ncnnModelBase()
{
    unload();
}


bool ncnnModelBase::load(QString baseDir, QString modelName)
{
    unload();

    qDebug()<<"[NCNN] model load begin";

    QString binFile = baseDir + "//" + modelName + ".bin";
    QString paramFile = baseDir + "//" + modelName + ".param";
    if(QFile::exists(binFile) && QFile::exists(binFile))
    {
        int res = net.load_param(paramFile.toLatin1().data());
        qDebug()<<"[NCNN] param consumed: "<<res;

        int consumed = net.load_model(binFile.toLatin1().data());
        qDebug()<<"[NCNN] bin consumed: "<<consumed;

        qDebug()<<"[NCNN] model loaded: "<<binFile;
        bLoad = true;
    }
    else
    {
        qDebug()<<"[NCNN] model not loaded.";
        bLoad = false;
    }
    return bLoad;
}

bool ncnnModelBase::load(QString modelPath)
{
    QFileInfo fileInfo(modelPath);
    return load(fileInfo.absolutePath(), fileInfo.baseName());
}

void ncnnModelBase::unload()
{
    if(!bLoad) return;

    bLoad = false;
    net.clear();
}


bool ncnnModelBase::moveFiles(QString modelName)
{
#ifdef Q_OS_ANDROID
    AndroidSetup setup;
    QString dataDir = setup.getAppDataDir();
    qDebug()<<"data Dir:"<<dataDir;

    if(QFile::exists(dataDir + "/" + modelName + ".bin"))
        return true;

    QString binFile = QString("assets:/dst/%1.bin").arg(modelName);
    QString paramFile = QString("assets:/dst/%1.param").arg(modelName);

    if(QFile::exists(binFile))
        qDebug()<<binFile<<" exist";
    else
        qDebug()<<binFile<<" not exist";

    if(QFile::exists(paramFile))
        qDebug()<<paramFile<<" exist";
    else
        qDebug()<<paramFile<<" not exist";

    bool bMove = true;
    QString dstName = dataDir + "/" + modelName + ".bin";
    if(!QFile::copy(binFile, dstName))
    {
        qDebug()<<"copy bin fail";
        bMove = false;
    }

    dstName = dataDir + "/" + modelName + ".param";
    if(!QFile::copy(paramFile, dstName))
    {
        qDebug()<<"copy param fail";
        bMove = false;
    }

    dstName = dataDir + "/test.jpg";
    if(!QFile::copy("assets:/dst/test.jpg", dstName))
    {
        qDebug()<<"copy image fail";
        bMove = false;
    }

    qDebug()<<"move status:"<<bMove;
    return bMove;
#endif // Q_OS_ANDROID
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
