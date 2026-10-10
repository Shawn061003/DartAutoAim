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
    // 主程序传入 YAML 中的绿色差分上下界；默认参数供离线工具直接构造使用。
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
    // 测试访问生产流程中的差分和轮廓实现，不公开中间步骤。
    friend class GuideLightDetectTestAccess;
    // 由构造参数初始化；两侧各自使用(lower,upper]差分区间。
    double right_diff_lower_;
    double right_diff_upper_;
    double left_diff_lower_;
    double left_diff_upper_;

    /// @brief 逐像素计算绿色差分 D = G - (R+B)/2，不修改输入图像。
    /// @param image 非空CV_8UC3 BGR全图。
    /// @param side 来源相机；预留侧别参数，当前两侧使用相同公式，返回Mat不携带侧别。
    /// @return 与输入同尺寸的CV_32FC1差分全图，保留[-255,255]内的负值和小数，不归一化。
    cv::Mat GetGreenDifference(const cv::Mat& image, CameraSide side);

    /// @brief 在单个ROI内保留差分下界<D<=上界的像素，并选取圆度最高的有效外轮廓。
    /// @note 候选轮廓面积须严格大于直径15像素圆的面积，约176.7像素平方，再比较圆度。
    /// @param image 已由GetGreenDifference生成的CV_32FC1单通道差分全图。
    /// @param roi YOLO提供的有效全图区域。
    /// @param side 来源相机，用于选择该侧的差分上下界。
    /// @return 全图坐标下的有序外轮廓，保留全部边界像素；无有效轮廓时返回空集合。
    std::vector<cv::Point2f> GetContours(const cv::Mat& image, const cv::Rect& roi,
                                         CameraSide side);

    /// @brief 对有序外轮廓拟合椭圆，获取浮点中心，不对结果取整。
    /// @param contours GetContours输出的全图坐标轮廓，至少需要5个点。
    /// @return 全图坐标中心；点数不足、轮廓退化或拟合失败时返回(NaN, NaN)。
    /// @note 调用方须检查中心是否有限，再将检测状态设为SUCCESS。
    cv::Point2f GetCenterPoint(const std::vector<cv::Point2f>& contours);
};
