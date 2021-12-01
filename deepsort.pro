TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
QT       += core gui

win32{
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -fopenmp -lgomp

#INCLUDEPATH += D:\OpenCVMinGW3.4.1\include
#LIBS += D:\OpenCVMinGW3.4.1\bin\libopencv_*.dll    # 没有ffmpeg
INCLUDEPATH +=  D:\OpenCV-MinGW-Build-OpenCV-3.4.6\include
LIBS += D:\OpenCV-MinGW-Build-OpenCV-3.4.6\x86\mingw\bin\libopencv_*.dll

LIBS += -L$$PWD/../../ncnn-lib/winlib/lib/ -lncnn

INCLUDEPATH += $$PWD/../../ncnn-lib/winlib/include
DEPENDPATH += $$PWD/../../ncnn-lib/winlib/include

PRE_TARGETDEPS += $$PWD/../../ncnn-lib/winlib/lib/libncnn.a

INCLUDEPATH += D:/eigen-3.4.0
}

SOURCES += \
    DeepAppearanceDescriptor/deepsort.cpp \
    DeepAppearanceDescriptor/model.cpp \
    KalmanFilter/kalmanfilter.cpp \
    KalmanFilter/linear_assignment.cpp \
    KalmanFilter/nn_matching.cpp \
    KalmanFilter/track.cpp \
    KalmanFilter/tracker.cpp \
    MunkresAssignment/munkres/munkres.cpp \
    MunkresAssignment/hungarianoper.cpp \
    YoloV5CustomLayer.cpp \
    main.cpp \
    ncnnmodelbase.cpp



HEADERS += \
    DeepAppearanceDescriptor/dataType.h \
    DeepAppearanceDescriptor/deepsort.h \
    DeepAppearanceDescriptor/model.h \
    KalmanFilter/kalmanfilter.h \
    KalmanFilter/linear_assignment.h \
    KalmanFilter/nn_matching.h \
    KalmanFilter/track.h \
    KalmanFilter/tracker.h \
    MunkresAssignment/munkres/matrix.h \
    MunkresAssignment/munkres/munkres.h \
    MunkresAssignment/hungarianoper.h \
    YoloV5CustomLayer.h \
    ncnnmodelbase.h
