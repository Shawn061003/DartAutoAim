#pragma once

#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <vector>

// 可视化约定：所有绘制实现在 Visualize.cpp；自定义数据类型均在本文件定义，
// 以 Visualize 结尾。仅使用标准库/OpenCV 基础类型，不依赖采集、推理或检测类型。
// 调用方负责转换数据；ROI、轮廓、中心均为对应原图坐标。Draw/Show 不修改输入。
// 中文由OpenCV FreeType绘制；可用DART_VISUALIZATION_FONT指定中文字体文件。
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
    // delay_ms > 0 时短暂刷新；0 时等待按键，GTK3 下关闭全部窗口也可结束等待。GUI 入口只在主线程调用。
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

    // 红色轮廓/十字表示成功，橙色表示失败；失败项仅显示诊断轮廓，不绘制中心。
    // 左右目标各自编号，编号相同不代表已完成双目关联。
    static cv::Mat Draw(const FrameVisualize& left, const FrameVisualize& right);
    static void Show(const FrameVisualize& left, const FrameVisualize& right,
                     int delay_ms = 1);
};

// detection：上层 YOLO、下层引导灯；每层均为左/右原图和 ROI 放大预览。
class DetectionVisualize {
public:
    static cv::Mat Draw(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                        const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                        const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                        const GuideLightDetectVisualize::FrameVisualize& guideRight);
    // 返回 waitKey 的键值，无按键/关闭全部窗口时为 -1；窗口生命周期由调用方管理。
    static int Show(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                     const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                     const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                     const GuideLightDetectVisualize::FrameVisualize& guideRight,
                     int delay_ms = 1);
};

class StereoscopicMeasurementVisualize {
public:
    enum class TargetClassVisualize { BASE, OUTPOST, UNKNOWN };
    struct TargetVisualize {
        // LocationTransform输出的镖口坐标：X向右、Y向前、Z向上，调用方将毫米转换为米。
        cv::Point3d position_m;
        TargetClassVisualize target_class = TargetClassVisualize::UNKNOWN;
    };
    struct FrameVisualize {
        std::vector<TargetVisualize> targets;
    };
    struct MetricsVisualize {
        double horizontal_distance_m = 0.0; // 到镖口原点的XY投影距离。
        double height_m = 0.0; // Z，高于镖口为正。
    };
    // 镖口坐标允许负Y和负Z；仅拒绝非有限位置。缺失目标不沿用上一帧。
    static MetricsVisualize Metrics(const cv::Point3d& position_m);
    // 左侧XY俯视、右侧YZ侧视；0.5m方格、等比例，镖口为唯一参考原点。
    static cv::Mat Draw(const FrameVisualize& data);
    // delay_ms=-1 只提交画面并返回 -1；其余情况返回 waitKey 键值，按键后保留窗口。
    static int Show(const FrameVisualize& data, int delay_ms = 1);
};
