#include "common/DartConfig.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

// 解析辅助函数仅在当前源文件内使用。prefix/name 用于生成带相机侧别的字段路径。
namespace {
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
DartCongfig::CameraParams readCamera(const cv::FileNode& node,
                                   const std::string& name)
{
    if (!node.isMap()) invalid(name, "missing camera section or expected a mapping");
    DartCongfig::CameraParams camera;
    camera.deviceIp = readString(node, "device_ip", name);
    camera.netIp = readString(node, "net_ip", name);
    camera.exposure = static_cast<float>(readNumber(
        node, "exposure", name, std::numeric_limits<float>::max()));
    camera.gain = static_cast<float>(readNumber(
        node, "gain", name, std::numeric_limits<float>::max()));
    camera.extraInfoDelay = readNumber(
        node, "extra_info_delay", name, std::numeric_limits<double>::max());
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
} // namespace

DartCongfig::DartCongfig(const std::string& configPath)
{
    load(configPath);
}

void DartCongfig::load(const std::string& configPath)
{
    try {
        // FileStorage 在离开作用域时释放文件资源，矩阵数据由返回的 cv::Mat 持有。
        cv::FileStorage file(configPath, cv::FileStorage::READ);
        if (!file.isOpened()) throw std::runtime_error("cannot open configuration file");
        auto left = readCamera(file["left_camera"], "left_camera");
        auto right = readCamera(file["right_camera"], "right_camera");
        // 两侧均通过校验后再更新；任一侧解析失败时，原有成员保持原值。
        leftCamera = std::move(left);
        rightCamera = std::move(right);
    } catch (const cv::Exception& error) {
        // 补充配置文件路径，供入口统一输出加载失败原因。
        throw std::runtime_error("DartCongfig [" + configPath + "]: " + error.what());
    } catch (const std::runtime_error& error) {
        // 补充配置文件路径，供入口统一输出加载失败原因。
        throw std::runtime_error("DartCongfig [" + configPath + "]: " + error.what());
    }
}
