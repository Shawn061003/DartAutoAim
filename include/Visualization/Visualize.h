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
    static void Show(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                     const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                     const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                     const GuideLightDetectVisualize::FrameVisualize& guideRight,
                     int delay_ms = 1);
};

class StereoscopicMeasurementVisualize {
public:
    enum class TargetClassVisualize { BASE, OUTPOST, UNKNOWN };
    struct TargetVisualize {
        // 右相机坐标系：X向右、Y向下、Z向前。调用方将测距的毫米转换为米。
        cv::Point3d position_m;
        TargetClassVisualize target_class = TargetClassVisualize::UNKNOWN;
        double left_reprojection_error_px = std::numeric_limits<double>::quiet_NaN();
        double right_reprojection_error_px = std::numeric_limits<double>::quiet_NaN();
    };
    struct FrameVisualize {
        // X_right = R * X_left + T，因此右光心为(0,0,0)，左光心为T；此处单位米。
        cv::Point3d left_camera_m;
        std::vector<TargetVisualize> targets;
        std::uint64_t left_frame_id = 0;
        std::uint64_t right_frame_id = 0;
    };
    struct MetricsVisualize {
        double l0_m = 0.0; // XZ投影中，到双目光心连线中点的距离。
        double l1_m = 0.0; // XZ投影中，到右光心的距离。
        double delta_h_m = 0.0; // -Y：高于右光心为正；非重力坐标系中的绝对高度。
    };
    // 对无穷/NaN/非正深度拒绝计算；绘图跳过无效目标，缺失目标不沿用上一帧。
    static MetricsVisualize Metrics(const cv::Point3d& position_m,
                                    const cv::Point3d& left_camera_m);
    // 左侧俯视，右侧侧视；0.5m方格，两视图均等比例；红色相机、绿色目标。
    static cv::Mat Draw(const FrameVisualize& data);
    static void Show(const FrameVisualize& data, int delay_ms = 1);
};
