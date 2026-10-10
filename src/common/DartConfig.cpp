#include "common/DartConfig.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

// 解析辅助函数仅在当前源文件内使用。prefix/name 用于生成带相机侧别的字段路径。
namespace {
// 文件输入的 mode 与媒体路径不允许为空。
std::string readNonEmptyString(const cv::FileNode& node, const std::string& field)
{
    if (!node.isString() || static_cast<std::string>(node).empty())
        throw std::runtime_error(field + ": expected a non-empty string");
    return static_cast<std::string>(node);
}

// 统一报告字段错误，例如 left_camera.image_width；load 再补充文件路径。
[[noreturn]] void invalid(const std::string& field, const std::string& reason)
{
    throw std::runtime_error(field + ": " + reason);
}

// 要求键存在；空字符串可作为设备 IP 的 TODO 占位值，由 readString 检查类型。
cv::FileNode required(const cv::FileNode& node, const char* key,
                      const std::string& prefix)
{
    const auto value = node[key];
    if (value.isNone()) invalid(prefix + "." + key, "missing required field");
    return value;
}

// 设备 IP 与本机网口 IP 按字符串读取，允许暂存空值。
std::string readString(const cv::FileNode& node, const char* key,
                       const std::string& prefix)
{
    const auto value = required(node, key, prefix);
    if (!value.isString()) invalid(prefix + "." + key, "expected a string");
    return static_cast<std::string>(value);
}

// 接受整数和浮点标量；统一检查有限性、非负性及目标类型上界。
// maximum 由调用方传入，防止后续转换到 float/int 时超出表示范围。
double readNumber(const cv::FileNode& node, const char* key,
                  const std::string& prefix, double maximum)
{
    const auto value = required(node, key, prefix);
    const std::string field = prefix + "." + key;
    if (!value.isInt() && !value.isReal()) invalid(field, "expected a number");
    const double result = static_cast<double>(value);
    if (!std::isfinite(result) || result < 0.0 || result > maximum)
        invalid(field, "expected a finite non-negative value within range");
    return result;
}

// 图像尺寸要求为正整数；转换前检查小数部分，避免截断配置值。
int readDimension(const cv::FileNode& node, const char* key,
                  const std::string& prefix)
{
    const double value = readNumber(node, key, prefix, std::numeric_limits<int>::max());
    if (value < 1.0 || std::floor(value) != value)
        invalid(prefix + "." + key, "expected a positive integer");
    return static_cast<int>(value);
}

// 读取 OpenCV 矩阵节点，统一转换为 CV_64F，并拒绝空矩阵、多通道及非有限值。
// 内参与畸变向量的具体尺寸由 readCamera 分别校验。
cv::Mat readMatrix(const cv::FileNode& node, const char* key,
                   const std::string& prefix)
{
    const std::string field = prefix + "." + key;
    cv::Mat result;
    try {
        required(node, key, prefix) >> result;
    } catch (const cv::Exception& error) {
        invalid(field, error.what());
    }
    if (result.empty() || result.channels() != 1 || result.dims != 2)
        invalid(field, "expected a non-empty single-channel matrix");
    result.convertTo(result, CV_64F);
    if (!cv::checkRange(result)) invalid(field, "matrix contains non-finite values");
    return result;
}

// 解析一侧相机的完整节点，返回通过校验的临时参数。
CameraParams readCamera(const cv::FileNode& node, const std::string& name)
{
    if (!node.isMap()) invalid(name, "missing camera section or expected a mapping");
    CameraParams camera;
    camera.deviceIp = readString(node, "device_ip", name);
    camera.netIp = readString(node, "net_ip", name);
    camera.exposure = static_cast<float>(readNumber(
        node, "exposure_us", name, std::numeric_limits<float>::max()));
    camera.gain = static_cast<float>(readNumber(
        node, "gain", name, std::numeric_limits<float>::max()));
    camera.extraInfoDelay = readNumber(
        node, "extra_info_delay_s", name, std::numeric_limits<double>::max());
    camera.width = readDimension(node, "image_width", name);
    camera.height = readDimension(node, "image_height", name);
    camera.cameraMatrix = readMatrix(node, "camera_matrix", name);
    camera.distCoeffs = readMatrix(node, "distortion_coefficients", name);

    // K 的结构为 [fx, s, cx; 0, fy, cy; 0, 0, 1]，fx、fy 必须为正。
    // 先检查尺寸，确保后续 at<double> 下标访问有效。
    const auto& k = camera.cameraMatrix;
    if (k.rows != 3 || k.cols != 3)
        invalid(name + ".camera_matrix", "expected a 3x3 matrix");
    if (k.at<double>(0, 0) <= 0.0 || k.at<double>(1, 1) <= 0.0 ||
        k.at<double>(1, 0) != 0.0 || k.at<double>(2, 0) != 0.0 ||
        k.at<double>(2, 1) != 0.0 || k.at<double>(2, 2) != 1.0)
        invalid(name + ".camera_matrix", "invalid pinhole intrinsics");

    // 当前标定采用五参数模型；接受 1x5 或 5x1，统一保存为独立的 5x1 矩阵。
    if (camera.distCoeffs.total() != 5 ||
        (camera.distCoeffs.rows != 1 && camera.distCoeffs.cols != 1))
        invalid(name + ".distortion_coefficients", "expected 5 coefficients: k1,k2,p1,p2,k3");
    camera.distCoeffs = camera.distCoeffs.reshape(1, 5).clone();
    return camera;
}

// 结构节点必须为映射，防止缺失节点被当作默认配置使用。
cv::FileNode section(const cv::FileNode& root, const char* key)
{
    const auto node = root[key];
    if (!node.isMap()) invalid(key, "expected a mapping");
    return node;
}

int readInteger(const cv::FileNode& node, const char* key,
                const std::string& prefix, int minimum)
{
    const double value = readNumber(node, key, prefix, std::numeric_limits<int>::max());
    if (value < minimum || std::floor(value) != value)
        invalid(prefix + "." + key, "integer value is outside the allowed range");
    return static_cast<int>(value);
}

std::string resolvePath(const std::string& value, const std::filesystem::path& base)
{
    if (value.empty()) return {};
    const std::filesystem::path path(value);
    return (path.is_absolute() ? path : base / path).lexically_normal().string();
}

InputParams readInput(const cv::FileNode& node, const std::filesystem::path& base)
{
    InputParams result;
    const auto mode = readNonEmptyString(required(node, "mode", "input"), "input.mode");
    if (mode == "camera") result.mode = InputMode::Camera;
    else if (mode == "video") result.mode = InputMode::Video;
    else if (mode == "image") result.mode = InputMode::Image;
    else invalid("input.mode", "expected camera, video or image");

    const auto path = [&](const char* key) {
        const auto value = readString(node, key, "input");
        if (result.mode != InputMode::Camera && value.empty())
            invalid(std::string("input.") + key, "expected a non-empty media path");
        return resolvePath(value, base);
    };
    result.leftPath = path("left_path");
    result.rightPath = path("right_path");
    return result;
}

FrameQueueParams readQueue(const cv::FileNode& node)
{
    FrameQueueParams result;
    result.capacity = static_cast<std::size_t>(readInteger(node, "capacity", "frame_queue", 1));
    result.maxTimestampDiffMs = readInteger(node, "max_timestamp_diff_ms", "frame_queue", 0);
    return result;
}

YOLOParams readYOLO(const cv::FileNode& node, const std::filesystem::path& base)
{
    YOLOParams result;
    result.modelPath = resolvePath(
        readNonEmptyString(required(node, "model_path", "yolo"), "yolo.model_path"), base);
    result.device = readNonEmptyString(required(node, "device", "yolo"), "yolo.device");
    result.inputSize = readInteger(node, "input_size", "yolo", 1);
    result.tileOverlapPx = readInteger(node, "tile_overlap_px", "yolo", 0);
    result.leftPaddingPx = readInteger(node, "left_padding_px", "yolo", 0);
    result.rightPaddingPx = readInteger(node, "right_padding_px", "yolo", 0);
    result.confidenceThreshold = static_cast<float>(readNumber(node, "confidence_threshold", "yolo", 1));
    result.nmsIouThreshold = static_cast<float>(readNumber(node, "nms_iou_threshold", "yolo", 1));
    result.warmupCount = readInteger(node, "warmup_count", "yolo", 1);
    result.validate();
    return result;
}

// 读取单通道差分阈值；范围和上下界顺序由GuideLightParams统一校验。
double readDifferenceBound(const cv::FileNode& node, const char* key)
{
    const auto value = required(node, key, "guide_light");
    if (!value.isInt() && !value.isReal())
        invalid(std::string("guide_light.") + key, "expected a numeric difference bound");
    return static_cast<double>(value);
}

GuideLightParams readGuideLight(const cv::FileNode& node)
{
    GuideLightParams result;
    result.leftDiffLower = readDifferenceBound(node, "left_diff_lower");
    result.leftDiffUpper = readDifferenceBound(node, "left_diff_upper");
    result.rightDiffLower = readDifferenceBound(node, "right_diff_lower");
    result.rightDiffUpper = readDifferenceBound(node, "right_diff_upper");
    result.validate();
    return result;
}

} // namespace

DartCongfig::DartCongfig(const std::string& configPath)
{
    load(configPath);
}

void DartCongfig::load(const std::string& configPath)
{
    try {
        cv::FileStorage file(configPath, cv::FileStorage::READ);
        if (!file.isOpened()) throw std::runtime_error("cannot open configuration file");
        const auto root = file.root();
        const auto base = std::filesystem::absolute(configPath).parent_path();
        auto nextInput = readInput(section(root, "input"), base);
        auto nextQueue = readQueue(section(root, "frame_queue"));
        const auto cameras = section(root, "cameras");
        auto left = readCamera(cameras["left"], "cameras.left");
        auto right = readCamera(cameras["right"], "cameras.right");
        auto nextYOLO = readYOLO(section(root, "yolo"), base);
        auto nextGuideLight = readGuideLight(section(root, "guide_light"));

        // 全部板块校验通过后提交，任何解析错误都保留之前的配置。
        input = std::move(nextInput);
        frameQueue = nextQueue;
        leftCamera = std::move(left);
        rightCamera = std::move(right);
        yolo = std::move(nextYOLO);
        guideLight = nextGuideLight;
    } catch (const std::exception& error) {
        throw std::runtime_error("DartCongfig [" + configPath + "]: " + error.what());
    }
}
