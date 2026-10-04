
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#include <cstdint>
#include <string>
#include <array>

#include <opencv2/opencv.hpp>

class DartCongfig
{
public:

    DartCongfig(/* args */);
    ~DartCongfig();

    struct LeftCameraParams {
        std::string deviceIp;   // 相机设备 IP（相机模式必填）
        std::string netIp;      // 本机网口 IP（相机模式必填）
        float exposure;         // 曝光时间（微秒，相机模式必填）
        float gain;             // 增益（相机模式必填）
        int width, height;      // 图像分辨率（像素）
        cv::Mat cameraMatrix;       // 3x3 CV_64F 内参矩阵
        cv::Mat distCoeffs;         // Nx1 CV_64F 畸变系数
        // 相机输入模式（CameraInputMode）extra_info 延迟（秒）：
        // 相机模式必填（config: extra_info_delay）。
        double extraInfoDelay;
    };

    struct RightCameraParams {
        std::string deviceIp;   // 相机设备 IP（相机模式必填）
        std::string netIp;      // 本机网口 IP（相机模式必填）
        float exposure;         // 曝光时间（微秒，相机模式必填）
        float gain;             // 增益（相机模式必填）
        int width, height;      // 图像分辨率（像素）
        cv::Mat cameraMatrix;       // 3x3 CV_64F 内参矩阵
        cv::Mat distCoeffs;         // Nx1 CV_64F 畸变系数
        // 相机输入模式（CameraInputMode）extra_info 延迟（秒）：
        // 相机模式必填（config: extra_info_delay）。
        double extraInfoDelay;
    };
    
};

DartCongfig::DartCongfig(/* args */){
    
}

DartCongfig::~DartCongfig(){

}
