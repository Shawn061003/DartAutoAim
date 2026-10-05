#ifndef DART_CONFIG_H
#define DART_CONFIG_H

#include <string>
#include <opencv2/core.hpp>

// 双相机配置：将 YAML 中的设备设置与标定结果加载到左右相机成员。
class DartCongfig
{
public:
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

    // 类型别名便于调用方表达相机侧别，字段定义统一维护。
    using LeftCameraParams = CameraParams;
    using RightCameraParams = CameraParams;

    LeftCameraParams leftCamera;    // YAML 节点 left_camera；当前为 MV-CS200-10GC
    RightCameraParams rightCamera;  // YAML 节点 right_camera；当前为 MV-CS016-10GC

    // 构造时调用 load；成功返回后，两侧参数均已完成解析和格式校验。
    // configPath 支持绝对路径；相对路径以进程工作目录为基准。
    explicit DartCongfig(const std::string& configPath = "config/dart.yaml");

    // 读取 OpenCV YAML（%YAML:1.0、!!opencv-matrix）。
    // 校验必填字段、数值范围、内参布局和五参数畸变向量。
    // 两侧均通过校验后更新成员；失败抛出 std::runtime_error，保留原配置。
    // 异常消息包含配置路径，字段校验错误还包含相机节点及字段名。
    // 允许 IP 为空、曝光为 0，以便先使用标定数据；上机前须补齐 TODO。
    void load(const std::string& configPath);
};

#endif // DART_CONFIG_H
