#pragma once

#include "Computation/StereoscopicMeasurement.h"
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

// 右相机原始坐标：光心为原点，X向右、Y向下、Z沿光轴向前，单位毫米。
// 镖口坐标：轨道终点为原点，X向右、Y沿发射方向的地面投影向前、Z向上。
// 使用固定外参，要求运行时两坐标系的相对位置和朝向与标定时一致。
class LocationTransform {
public:
    struct LocationTransformResult {
        cv::Point3d RealLocation{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()}; // 镖口坐标，毫米
        TargetClasses TargetName = TargetClasses::UNKNOWN;
        bool stable = false; // 原样传递测距阶段对当前目标的稳定性判断。
    };

    /// @brief 初始化坐标转换器，保存外参矩阵的独立副本。
    /// @param rightCameraToDartR 右相机坐标到镖口坐标的旋转，3x3 CV_64F。
    /// @param rightCameraToDartTmm 右相机坐标到镖口坐标的平移，3x1 CV_64F，毫米。
    /// @pre 调用方传入已解析并校验的外参，R为合法旋转，T有限。
    /// @note p_dart_mm = R * p_right_camera_mm + T_mm。
    ///       R包含轴定义转换及安装旋转；T_mm是相机光心在镖口坐标系中的位置。
    ///       只补偿平移且相机右/前/下与镖口右/前/下对齐时，
    ///       R应为[1,0,0; 0,0,1; 0,-1,0]。
    LocationTransform(const cv::Mat& rightCameraToDartR,
                      const cv::Mat& rightCameraToDartTmm)
        : rightCameraToDartR_(rightCameraToDartR.clone()),
          rightCameraToDartTmm_(rightCameraToDartTmm.clone()) {}
    
    /// @brief 将双目测距结果转换到镖口坐标系，输入输出坐标均为毫米。
    /// @pre 各项RealLocation有限，且位于原始右相机坐标系。
    /// @return 按输入顺序返回转换后的坐标和原目标类别；空输入返回空列表。
    std::vector<LocationTransformResult> RunTransform(
        const std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult>& results) const;

private:
    cv::Mat rightCameraToDartR_;
    cv::Mat rightCameraToDartTmm_;
};
