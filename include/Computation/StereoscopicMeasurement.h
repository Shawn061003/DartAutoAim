#pragma once

#include "Perception/GuideLightDetect.h"
#include "common/DartConfig.h"
#include <cstddef>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

// 像素坐标：对应输入图像的全图左上角为原点，u向右、v向下。
// 三维坐标：右相机光心为原点，X向右、Y向下、Z沿光轴向前，单位毫米。
// 输出使用原始右相机坐标系；三角化若使用校正坐标系，须先转换再输出。
enum class TargetClasses { BASE, OUTPOST, UNKNOWN };

class StereoscopicMeasurement {
public:
    struct StereoscopicMeasurementResult {
        // 同一目标在左右输入图像上的原始全图像素中心，未去畸变。
        cv::Point2f leftPixel{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()};
        cv::Point2f rightPixel{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()};
        cv::Point3d RealLocation{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()}; // 右相机光心坐标，毫米
        TargetClasses TargetName = TargetClasses::UNKNOWN;
        // 去畸变像素坐标中的重投影误差，与几何筛选使用的误差定义一致。
        double leftReprojectionErrorPx = std::numeric_limits<double>::quiet_NaN();
        double rightReprojectionErrorPx = std::numeric_limits<double>::quiet_NaN();
        // 两侧重投影误差的最大值，单位像素；由SelectFromCandidate从选中候选填入。
        double reprojectionErrorPx = std::numeric_limits<double>::quiet_NaN();
    };

    /// @brief 初始化测距器，保存标定矩阵的独立副本，不读取配置文件。
    /// @param leftCamera 左相机参数，仅使用成像尺寸、内参和畸变系数。
    /// @param rightCamera 右相机参数，仅使用成像尺寸、内参和畸变系数。
    /// @param leftToRightR 左相机坐标到右相机坐标的旋转，3x3 CV_64F。
    /// @param leftToRightTmm 左相机坐标到右相机坐标的平移，3x1 CV_64F，毫米。
    /// @pre 调用方传入有效标定：内参3x3、畸变5x1，均为CV_64F且数值有限；
    ///      成像尺寸为正，R为合法旋转，T有限且基线非零。
    /// @note X_right_mm = leftToRightR * X_left_mm + leftToRightTmm。
    ///       本阶段构造函数只保存参数，上述参数校验留待后续实现。
    StereoscopicMeasurement(const CameraParams& leftCamera,
                            const CameraParams& rightCamera,
                            const cv::Mat& leftToRightR,
                            const cv::Mat& leftToRightTmm)
        : leftImageSize_(leftCamera.width, leftCamera.height),
          rightImageSize_(rightCamera.width, rightCamera.height),
          leftCameraMatrix_(leftCamera.cameraMatrix.clone()),
          rightCameraMatrix_(rightCamera.cameraMatrix.clone()),
          leftDistCoeffs_(leftCamera.distCoeffs.clone()),
          rightDistCoeffs_(rightCamera.distCoeffs.clone()),
          leftToRightR_(leftToRightR.clone()),
          leftToRightTmm_(leftToRightTmm.clone()) {}

    /// @brief 双目测距主入口：检查检测结果、生成几何候选、联合配对并识别目标。
    /// @param result 同一对已同步左右帧的检测结果，不修改调用方数据。
    /// @pre 左右各最多两个检测项，侧别正确；像素坐标与传入内参的成像空间一致。
    ///      输入若缩放或裁剪，调用方须同步调整内参和成像尺寸。
    /// @return 仅包含通过几何与先验检查的有效结果，数量0～2；BASE/OUTPOST各最多一个。
    /// @note 调用方通过TargetName区分类别，不依赖下标；目标缺失时不添加占位项。
    ///       UNKNOWN仅用于默认初始化，当前入口不输出未识别目标。
    std::vector<StereoscopicMeasurementResult> RunMeasurement(
        const GuideLightDetect::StereoDetectResult& result) const;

private:
    struct Candidate {
        // 索引指向原始StereoDetectResult的左右集合；过滤失败项时仍保留原索引。
        std::size_t leftIndex = 0;
        std::size_t rightIndex = 0;
        StereoscopicMeasurementResult measurement; // 候选阶段TargetName为UNKNOWN
        // 去畸变像素坐标中，两侧点到对应极线距离的最大值，单位像素。
        double epipolarErrorPx = std::numeric_limits<double>::quiet_NaN();
        // 去畸变像素坐标中，两侧重投影欧氏误差的最大值，单位像素。
        double reprojectionErrorPx = std::numeric_limits<double>::quiet_NaN();
    };

    /// @brief 从原始检测集合中跳过FAILED或中心非有限项，枚举最多四组左右配对。
    /// @param result 已满足入口前置条件的检测结果；左右相同下标不代表同一物理目标。
    /// @return 通过极线约束、非退化三角化、双相机正深度与重投影检查的候选。
    /// @note 解算使用去畸变后的点和相匹配的投影矩阵；候选记录原始像素及原始索引。
    ///       几何误差阈值后续在.cpp中固定定义，单位像素；误差等于阈值时保留。
    std::vector<Candidate> GetCandidate(
        const GuideLightDetect::StereoDetectResult& result) const;

    /// @brief 按右相机XZ平面投影距离sqrt(X*X+Z*Z)选择目标，优先BASE，最多返回两个。
    /// @param candidates 已通过几何检查的候选，不修改候选本身。
    /// @return 复制选中候选的测量信息并标记BASE/OUTPOST，无合法分配时返回空集合。
    /// @note BASE的投影距离范围为[23500,25500]毫米，选最接近24400毫米的候选；
    ///       再从不复用BASE左检测或右检测的候选中选择OUTPOST，
    ///       OUTPOST的投影距离范围为[14000,16000]毫米，选最接近15200毫米的候选。
    ///       同误差保留输入中先出现者；各类别最多一个，缺失时不添加占位项。
    std::vector<StereoscopicMeasurementResult> SelectFromCandidate(
        const std::vector<Candidate>& candidates) const;

    cv::Size leftImageSize_;
    cv::Size rightImageSize_;
    cv::Mat leftCameraMatrix_;
    cv::Mat rightCameraMatrix_;
    cv::Mat leftDistCoeffs_;
    cv::Mat rightDistCoeffs_;
    cv::Mat leftToRightR_;
    cv::Mat leftToRightTmm_;
};
