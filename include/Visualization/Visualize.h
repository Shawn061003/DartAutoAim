#pragma once

#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

// 可视化约定：所有绘制实现在 Visualize.cpp；自定义数据类型均在本文件定义，
// 以 Visualize 结尾。仅使用标准库/OpenCV 基础类型，不依赖采集、推理或检测类型。
// 调用方负责转换数据；ROI、轮廓、中心均为对应原图坐标。Draw/Show 不修改输入。
struct CameraFrameVisualize {
    cv::Mat image; // CV_8UC3 BGR；同步绘制期间由调用方保持图像有效且不被修改。
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_ms = 0;
};

class YOLOInferenceVisualize {
public:
    struct TargetVisualize {
        cv::Rect roi;
        double confidence = 0.0;
    };
    struct FrameVisualize {
        CameraFrameVisualize frame;
        std::vector<TargetVisualize> targets;
    };

    // 返回左右并排的 BGR 画布，便于无窗口测试或由调用方保存。
    static cv::Mat Draw(const FrameVisualize& left, const FrameVisualize& right);
    // delay_ms > 0 时短暂刷新；0 时等待按键或关闭窗口。GUI 入口只在主线程调用。
    static void Show(const FrameVisualize& left, const FrameVisualize& right,
                     int delay_ms = 1);
};

class GuideLightDetectVisualize {
public:
    struct TargetVisualize {
        cv::Rect roi;
        bool success = false;
        cv::Point2f center{std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::quiet_NaN()};
        std::vector<cv::Point2f> contours;
    };
    struct FrameVisualize {
        CameraFrameVisualize frame;
        std::vector<TargetVisualize> targets;
    };

    // 绿色轮廓/十字表示成功，橙色表示失败；失败项仅显示诊断轮廓，不绘制中心。
    // 左右目标各自编号，编号相同不代表已完成双目关联。
    static cv::Mat Draw(const FrameVisualize& left, const FrameVisualize& right);
    static void Show(const FrameVisualize& left, const FrameVisualize& right,
                     int delay_ms = 1);
};
