#pragma once

// 图像坐标约定：以全图左上角为原点，x向右、y向下。

#include "common/DartConfig.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <opencv2/core.hpp>

// 所有采集、感知和解算模块共用；Unknown表示尚未标明来源，不能直接进入检测。
enum class CameraSide { Unknown, Left, Right };

// 采集模块提供的单帧数据，供感知与解算模块共享。
struct CameraFrame {
    cv::Mat image;                  // CV_8UC3 BGR全图，使用期间保持有效且不被修改
    std::uint64_t frame_id = 0;      // 相机帧号
    std::int64_t timestamp_ms = 0;   // 采集时间戳（毫秒），左右使用同一时间基准
    CameraSide camera_side = CameraSide::Unknown; // 来源相机，随帧传递
};

// 队列完整定义保留在 AcquireFrame.h，在 .cpp 中包含，避免头文件循环依赖。
class StereoFrameQueue;

// 单相机设备：管理 SDK 句柄、参数和单次取图；采集循环由 CameraInput 驱动。
// 生命周期操作由同一控制线程调用，停止前须等待使用该设备的取图线程退出。
class CameraDevice {
public:
    // 保存已解析的相机参数和固定侧别；side 必须是 Left 或 Right。
    // 构造只保存、校验配置，连接设备在 StartGrabbing 中完成。
    CameraDevice(const CameraParams& params, CameraSide side);
    ~CameraDevice() noexcept;

    // SDK 句柄由当前对象独占，禁止复制或移动设备对象。
    CameraDevice(const CameraDevice&) = delete;
    CameraDevice& operator=(const CameraDevice&) = delete;
    CameraDevice(CameraDevice&&) = delete;
    CameraDevice& operator=(CameraDevice&&) = delete;

    // 查找并打开设备、设置参数、启动采集；失败时清理已取得的资源并抛异常。
    void StartGrabbing();

    // 最多等待 timeoutMs 毫秒；超时返回 nullopt，未启动或设备错误抛异常。
    // 由一个采集线程调用。成功帧独立持有 BGR 全图，释放 SDK 缓存后仍有效。
    // 填写帧号和固定侧别；timestamp_ms 使用主机 steady_clock 收帧时间，不代表同步曝光。
    std::optional<CameraFrame> GrabImage(std::uint32_t timeoutMs);

    // 停止采集、关闭设备并销毁句柄；允许重复调用，兼容初始化中途失败。
    // 析构调用本接口；调用前须结束外部取图循环。
    void StopGrabbing() noexcept;

private:
    // 按设备 IP 和本机网口 IP 定位相机，创建句柄并打开设备。
    void FindCamera();

    // 设置曝光、增益、连续采集方式及网口传输参数，按设备节点检查支持范围。
    // 首版采集全图；标定尺寸用于核对输出，不直接作为硬件 ROI 设置。
    void SetCamera();

    // 关闭已打开的设备并销毁句柄；供停止流程和初始化失败清理流程调用。
    void CloseCamera() noexcept;

    CameraParams params_;                 // 复用统一配置，采集类不解析 YAML
    const CameraSide side_;               // 绑定当前设备的左右身份
    void* handle_ = nullptr;              // MVS 设备句柄，SDK 头文件放在 .cpp 中
    bool sdkInitialized_ = false;         // 当前设备持有一次 SDK 初始化引用
    bool opened_ = false;                 // 已成功打开设备
    bool grabbing_ = false;               // 已成功启动 SDK 采集
};

// 双相机输入：持有左右设备、队列和两个采集线程，分别向队列发布帧。
// 下游通过 frameQueue().GetFrame() 配对取帧；本类只负责生产图像。
class CameraInput {
public:
    // queueCapacity 为每侧容量，必须大于 0；grabTimeoutMs 为单次取图超时，必须大于 0。
    // 构造准备设备对象和队列，StartCamera 显式启动实际采集。
    CameraInput(const CameraParams& leftParams, const CameraParams& rightParams,
                std::size_t queueCapacity, std::uint32_t grabTimeoutMs = 100);
    // 在 .cpp 中定义析构：先停止并回收线程，再释放设备与队列。
    ~CameraInput() noexcept;

    CameraInput(const CameraInput&) = delete;
    CameraInput& operator=(const CameraInput&) = delete;
    CameraInput(CameraInput&&) = delete;
    CameraInput& operator=(CameraInput&&) = delete;

    // 启动两台设备及各自的采集线程，然后返回；持续采集在后台循环中进行。
    // 任一设备或线程启动失败时，停止已启动部分、回收资源并向调用方抛异常。
    // 再次启动前须 StopCamera；启动与停止清空队列，队列引用保持有效。
    // StartCamera/StopCamera 由同一控制线程串行调用。
    void StartCamera();

    // 请求两个循环退出，等待有限超时的取图结束并回收线程，随后停止、关闭设备。
    // 允许重复调用；析构调用本接口。由采集线程之外的控制线程调用。
    void StopCamera() noexcept;

    // 与 VideoInput 一致的队列入口；引用有效期不超过当前输入对象。
    // 调用队列成员的文件需包含 Acquisition/AcquireFrame.h。
    StereoFrameQueue& frameQueue() noexcept;

private:
    // 每个线程只操作一台设备：取图、按侧别入队；超时继续等待。
    // 捕获取图或入队异常，记录错误并请求两个循环退出，异常不逃出线程入口。
    void CaptureLoop(CameraDevice& device) noexcept;

    CameraDevice leftCamera_;
    CameraDevice rightCamera_;
    std::unique_ptr<StereoFrameQueue> queue_; // 完整类型在 .cpp 中可见，队列负责入队拷贝
    const std::uint32_t grabTimeoutMs_;
    std::atomic<bool> stopRequested_{true};
    std::thread leftThread_;
    std::thread rightThread_;
};
