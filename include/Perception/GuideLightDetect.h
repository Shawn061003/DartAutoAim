#pragma once

// 坐标约定：本文件中的roi、CenterPoint和contours均位于对应输入图像的全图坐标系。
// 全图左上角为原点，x向右、y向下；ROI内部提取的局部坐标须加roi左上角后输出。
// 输入图像与YOLO结果须来自同一帧、同一图像坐标空间。

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>
#include "Acquisition/CameraInput.h"
#include "Perception/YOLOinfer.h"

class GuideLightDetect
{
public:
    // 主程序传入 YAML 中的 HSV 阈值；默认参数供离线工具直接构造使用。
    explicit GuideLightDetect(const GuideLightParams& params = {});

    enum class DetectStatus {
        SUCCESS,
        FAILED
    };

    struct GuideLightDetectResult {
        DetectStatus status = DetectStatus::FAILED; // 仅SUCCESS时中心与轮廓可用于解算
        cv::Rect roi;                               // 保存roi区域信息
        cv::Point2f CenterPoint;                    // 中心点检测结果
        std::vector<cv::Point2f> contours;          // 目标轮廓
        std::uint64_t frame_id = 0;                 // 来源帧号
        std::int64_t timestamp_ms = 0;              // 来源帧采集时间戳（毫秒）
        CameraSide camera_side = CameraSide::Unknown; // 包括FAILED项，均保留输入帧的相机侧别
    };

    using DetectResult = std::vector<GuideLightDetectResult>;
    using LeftDetectResult = DetectResult;
    using RightDetectResult = DetectResult;

    struct StereoDetectResult {
        LeftDetectResult left;
        RightDetectResult right;
    };

    /// @brief 传统视觉检测主入口
    /// @param result 已保证ROI有效；来源帧号、时间戳、侧别须与对应CameraFrame一致。
    /// @pre 两帧camera_side须分别为Left/Right，且YOLO结果侧别一致；不一致时抛异常。
    /// @return 左右各自按输入ROI顺序返回结果，每个ROI保留一项，失败时标记FAILED。
    /// @note 失败项中心为(NaN, NaN)，保留已提取的轮廓供排查；帧信息取对应CameraFrame。
    ///       左右结果独立，相同下标不表示同一物理目标；无ROI时对应集合为空。
    StereoDetectResult RunDetection(const CameraFrame& leftFrame,
                                    const CameraFrame& rightFrame,
                                    const YOLOInference::StereoYOLOResult& result);

private:
    // 调参测试访问同一份GetContours实现，不开放生产流程中的中间步骤。
    friend class GuideLightDetectTestAccess;
    // 由构造参数初始化；OpenCV 8位HSV：H 0~179，S/V 0~255。
    cv::Scalar right_hsv_lower_;
    cv::Scalar right_hsv_upper_;
    cv::Scalar left_hsv_lower_;
    cv::Scalar left_hsv_upper_;
    /// @brief 在单个ROI内筛选高亮绿色，并选取圆度最高的有效外轮廓。
    /// @param image CV_8UC3 BGR全图；roi为YOLO提供的有效区域。
    /// @param side 输入来自左相机还是右相机，用于选择对应HSV阈值。
    /// @return 全图坐标下的有序外轮廓，保留全部边界像素；无有效轮廓时返回空集合。
    std::vector<cv::Point2f> GetContours(const cv::Mat& image, const cv::Rect& roi,
                                         CameraSide side);

    /// @brief 对有序外轮廓拟合椭圆，获取浮点中心，不对结果取整。
    /// @param contours GetContours输出的全图坐标轮廓，至少需要5个点。
    /// @return 全图坐标中心；点数不足、轮廓退化或拟合失败时返回(NaN, NaN)。
    /// @note 调用方须检查中心是否有限，再将检测状态设为SUCCESS。
    cv::Point2f GetCenterPoint(const std::vector<cv::Point2f>& contours);
};
