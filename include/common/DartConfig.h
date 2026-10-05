#ifndef DART_CONFIG_H
#define DART_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>

// 运行模式仅由 YAML 的 input.mode 决定。
enum class InputMode { Camera, Video, Image };

struct InputParams {
    InputMode mode = InputMode::Image;
    std::string leftPath;  // image/video 的媒体路径，解析后为绝对路径
    std::string rightPath;
};

struct FrameQueueParams {
    std::size_t capacity = 4;               // 每侧缓存帧数
    std::int64_t maxTimestampDiffMs = 5;    // 配对允许时间差，含等于边界
};

// 检测器持有独立参数副本；默认值供离线工具和直接构造的测试使用。
struct YOLOParams {
    std::string modelPath;
    std::string device = "CPU";
    int inputSize = 512;        // 模型输入正方形边长，须与模型支持的形状一致
    int tileOverlapPx = 30;     // 四切相邻区域的总重叠宽度，须小于原图宽高
    int leftPaddingPx = 25;     // 左图 ROI 每边外扩量
    int rightPaddingPx = 10;    // 右图 ROI 每边外扩量
    float confidenceThreshold = 0.25F;
    float nmsIouThreshold = 0.45F;
    int warmupCount = 3;        // 每套 batch 请求的预热次数，至少 1 次

    // 同时用于 YAML 和直接构造检测器，保证两种入口具有相同数值约束。
    void validate() const {
        if (modelPath.empty()) throw std::invalid_argument("yolo.model_path: expected a non-empty path");
        if (device.empty()) throw std::invalid_argument("yolo.device: expected a non-empty device");
        if (inputSize <= 0) throw std::invalid_argument("yolo.input_size: expected a positive integer");
        if (tileOverlapPx < 0) throw std::invalid_argument("yolo.tile_overlap_px: expected a non-negative integer");
        if (leftPaddingPx < 0 || rightPaddingPx < 0)
            throw std::invalid_argument("yolo padding: expected non-negative integers");
        if (!std::isfinite(confidenceThreshold) || confidenceThreshold < 0 || confidenceThreshold > 1)
            throw std::invalid_argument("yolo.confidence_threshold: expected a value in [0,1]");
        if (!std::isfinite(nmsIouThreshold) || nmsIouThreshold < 0 || nmsIouThreshold > 1)
            throw std::invalid_argument("yolo.nms_iou_threshold: expected a value in [0,1]");
        if (warmupCount < 1) throw std::invalid_argument("yolo.warmup_count: expected a positive integer");
    }
};

struct GuideLightParams {
    cv::Scalar rightHsvLower{60, 102, 81};
    cv::Scalar rightHsvUpper{86, 212, 255};
    cv::Scalar leftHsvLower{61, 210, 140};
    cv::Scalar leftHsvUpper{74, 230, 240};

    // 两侧分别校验；OpenCV 8位HSV：H为0..179，S/V为0..255，包含上下界。
    void validate() const {
        const auto check = [](const cv::Scalar& lower, const cv::Scalar& upper, const char* side) {
            for (int i = 0; i < 3; ++i) {
                const double maximum = i == 0 ? 179 : 255;
                if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) ||
                    lower[i] < 0 || upper[i] > maximum || lower[i] > upper[i] ||
                    std::floor(lower[i]) != lower[i] || std::floor(upper[i]) != upper[i])
                    throw std::invalid_argument(std::string("guide_light ") + side +
                        " HSV: expected ordered integer bounds in H[0,179], S/V[0,255]");
            }
        };
        check(leftHsvLower, leftHsvUpper, "left");
        check(rightHsvLower, rightHsvUpper, "right");
    }
};

// 左右相机共用字段定义，各自保存一份参数；数值字段的初始值为待配置值。
struct CameraParams {
    std::string deviceIp;       // 相机设备 IP，启用相机前填写
    std::string netIp;          // 本机网口 IP，启用相机前填写
    float exposure = 0.0F;      // 曝光时间（微秒）；0 表示待配置
    float gain = 0.0F;          // 增益；取值范围和单位按相机 SDK 约定，上机前确认
    int width = 0;              // 标定图像宽度（像素），对应 YAML 的 image_width
    int height = 0;             // 标定图像高度（像素），对应 YAML 的 image_height
    // K = [fx, s, cx; 0, fy, cy; 0, 0, 1]，3x3 CV_64F。
    // fx/fy 为像素焦距，cx/cy 为主点坐标，s 为斜切项；当前标定的 s 为 0。
    // K 对应 width、height 指定的成像尺寸；缩放、裁剪后须调整 K。
    cv::Mat cameraMatrix;
    // 畸变向量为 5x1 CV_64F，顺序 k1,k2,p1,p2,k3。
    // k1/k2/k3 为径向系数，p1/p2 为切向系数，均为无量纲。
    cv::Mat distCoeffs;
    double extraInfoDelay = 0.0; // extra_info 延迟（秒），启用相机前确认
};

// 统一配置：加载并保存输入、相机、队列与检测参数。
class DartCongfig
{
public:
    InputParams input;
    FrameQueueParams frameQueue;
    YOLOParams yolo;
    GuideLightParams guideLight;
    // 类型别名便于调用方表达相机侧别，字段定义统一维护。
    using LeftCameraParams = CameraParams;
    using RightCameraParams = CameraParams;

    LeftCameraParams leftCamera;    // YAML 节点 cameras.left；当前为 MV-CS200-10GC
    RightCameraParams rightCamera;  // YAML 节点 cameras.right；当前为 MV-CS016-10GC

    // 构造时调用 load；成功返回后，两侧参数均已完成解析和格式校验。
    // configPath 支持绝对路径；相对路径以进程工作目录为基准。
    explicit DartCongfig(const std::string& configPath = "config/dart.yaml");

    // 一次读取五个板块，全部校验通过后替换成员；失败保留原配置。
    // 媒体与模型相对路径统一以 YAML 所在目录为基准。
    // image/video 要求媒体路径非空；设备 IP 与曝光允许 TODO 占位。
    void load(const std::string& configPath);
};

#endif // DART_CONFIG_H
