#include "common/DartConfig.h"
#include "Acquisition/VideoInput.h"
#include "Acquisition/CameraInput.h"
#include "Acquisition/AcquireFrame.h"
#include "Acquisition/Quit.h"
#include "Perception/YOLOinfer.h"
#include "Perception/GuideLightDetect.h"
#include "Computation/StereoscopicMeasurement.h"
#include "Computation/LocationTransform.h"
#include "Computation/YawSolver.h"
#include "Communication/Communication.hpp"
#include "Visualization/Visualize.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <opencv2/highgui.hpp>

namespace {
// 默认配置定位到源码目录；显式 --config 路径以进程工作目录为基准。
const std::filesystem::path kProjectRoot{DART_PROJECT_ROOT};

// true 选择基地，false 选择前哨站；只在主线程更新。
bool aim_base = true;

// 配置由 DartCongfig 解析；通信对象在相机流程结束后释放。
CommunicationParams communicationParams;
std::unique_ptr<dart_mcu::Communication> communication;
bool preAimPending = false;
std::optional<bool> requestedAim;
std::uint64_t aimRevision = 0;
std::uint64_t lastCommandStatusSequence = 0;
std::chrono::steady_clock::time_point lastCommandAt{};

void printUsage(const char* program)
{
    std::cout << "Usage: " << program << " [--config path.yaml]\n"
              << "  Input mode and serial settings are configured in YAML.\n"
              << "  --help  Show this help.\n";
}

// 命令行只选择配置文件，输入模式和串口参数均由 YAML 提供。
std::string parseConfigPath(int argc, char* argv[])
{
    if (argc == 1) return (kProjectRoot / "config/dart.yaml").string();
    if (argc == 3 && std::string(argv[1]) == "--config" &&
        std::string(argv[2]).size() > 0 && std::string(argv[2]).rfind("--", 0) != 0)
        return argv[2];
    throw std::invalid_argument("Expected --config path.yaml; configure input/communication in YAML");
}

// 调用方只做数据转换；可视化模块不引用采集/检测模块的结构体。
YOLOInferenceVisualize::FrameVisualize makeYOLOVisualize(
    const CameraFrame& frame, const YOLOInference::YOLOResults& results)
{
    YOLOInferenceVisualize::FrameVisualize data{
        {frame.image, frame.frame_id, frame.timestamp_ms}, {}};
    data.targets.reserve(results.size());
    for (const auto& result : results)
        data.targets.push_back({result.roi, result.conf});
    return data;
}

GuideLightDetectVisualize::FrameVisualize makeGuideLightVisualize(
    const CameraFrame& frame, const GuideLightDetect::DetectResult& results)
{
    GuideLightDetectVisualize::FrameVisualize data{
        {frame.image, frame.frame_id, frame.timestamp_ms}, {}};
    data.targets.reserve(results.size());
    for (const auto& result : results)
        data.targets.push_back({result.roi,
            result.status == GuideLightDetect::DetectStatus::SUCCESS,
            result.CenterPoint, result.contours});
    return data;
}

StereoscopicMeasurementVisualize::FrameVisualize makeLocationVisualize(
    const std::vector<LocationTransform::LocationTransformResult>& locations)
{
    StereoscopicMeasurementVisualize::FrameVisualize data;
    data.targets.reserve(locations.size());
    using ClassVisualize = StereoscopicMeasurementVisualize::TargetClassVisualize;
    for (const auto& result : locations) {
        auto category = ClassVisualize::UNKNOWN;
        if (result.TargetName == TargetClasses::BASE) category = ClassVisualize::BASE;
        else if (result.TargetName == TargetClasses::OUTPOST) category = ClassVisualize::OUTPOST;
        data.targets.push_back({result.RealLocation * 0.001, category});
    }
    return data;
}

// 只在主线程更新瞄准模式；true 选择基地，false 选择前哨站。
// 返回 true 表示收到模式键，图片模式据此继续等待后续按键。
bool handleAimModeKey(int key)
{
    const bool modeKey = key == '1' || key == '2';
    if (modeKey) {
        const bool selectedBase = key == '1';
        if (!requestedAim || selectedBase != *requestedAim) {
            requestedAim = selectedBase;
            aim_base = selectedBase;
            ++aimRevision;
            preAimPending = communication != nullptr;
        }
        std::cout << "Aim mode: " << (aim_base ? "BASE" : "OUTPOST")
                  << " (aim_base=" << (aim_base ? "true" : "false") << ")" << std::endl;
    }
    // 相机循环传入 -1 时也处理待发送请求，移动时保留最近一次目标选择。
    if (preAimPending && communication) {
        const auto status = communication->latestStatus(
            std::chrono::milliseconds(communicationParams.statusMaxAgeMs));
        if (status && status->status.is_stationary == 1 &&
            status->sequence > lastCommandStatusSequence && status->received_at > lastCommandAt) {
            communication->send({0, aim_base ? 0.26 : 0.17, 0},
                std::chrono::milliseconds(communicationParams.writeTimeoutMs));
            lastCommandStatusSequence = status->sequence;
            lastCommandAt = std::chrono::steady_clock::now();
            preAimPending = false;
        }
    }
    return modeKey;
}

// 一对匹配帧共享同一条检测链；图像、YOLO ROI 和传统视觉结果保持原图坐标。
// 返回值保留两侧各自的帧号、时间戳和检测状态，供后续模块继续使用。
GuideLightDetect::StereoDetectResult processFrame(
    const StereoFrame& frame, YOLOInference& yolo, GuideLightDetect& guideLight)
{
    // 第一步：YOLO 为左右原图生成候选 ROI。
    const auto rois = yolo.RunYOLOInfer(frame.left, frame.right);
    // 第二步：在各 ROI 内提取绿色轮廓并拟合中心，完成本阶段视觉检测。
    const auto detections = guideLight.RunDetection(frame.left, frame.right, rois);
    // 一个 detection 窗口，上层 YOLO、下层传统视觉，每对帧只刷新一次。
    const int key = DetectionVisualize::Show(makeYOLOVisualize(frame.left, rois.left),
                             makeYOLOVisualize(frame.right, rois.right),
                             makeGuideLightVisualize(frame.left, detections.left),
                             makeGuideLightVisualize(frame.right, detections.right));
    handleAimModeKey(key);
    return detections;
}

// 校验发送所需的通信、目标选择、电控状态与帧时间；不满足条件时返回空值。
// 通过时返回本次使用的电控状态，供调用方在发送成功后记录其序号。
std::optional<dart_mcu::ReceivedStatus> checkSendConditions(
    const std::optional<double>& yaw, const StereoFrame& frame,
    std::uint64_t frameAimRevision, const CommunicationParams& params)
{
    // 先检查当前帧是否具备发送基础；任意条件成立都跳过本次发送。
    if (!communication ||                       // 通信对象尚未创建，无法读取状态或发送命令。
        preAimPending ||                        // 预瞄准请求仍在等待发送，优先完成该请求。
        !yaw ||                                 // 当前所选目标缺失或无法求解，没有可发送的角度。
        frameAimRevision != aimRevision)         // 本帧处理期间切换过目标，丢弃本帧调整结果。
        return std::nullopt;

    // 获取未超过 YAML 中 status_max_age_ms 的最新有效电控回传。
    // 尚未收到有效状态、状态过期、通信停止或故障时，接口返回空值。
    const auto status = communication->latestStatus(
        std::chrono::milliseconds(params.statusMaxAgeMs));

    // 将上次命令写入完成时间转换为毫秒，与左右帧的本机收帧时间戳比较。
    // 两者都使用 steady_clock；该时间仅表示串口写入完成，机构执行状态由回传提供。
    const auto lastCommandMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        lastCommandAt.time_since_epoch()).count();

    // 再检查镖架状态和数据时序；任意条件成立都等待后续状态或新帧。
    if (!status ||                              // 没有有效且未过期的电控状态，当前运动状态未知。
        status->status.is_stationary != 1 ||     // 电控报告镖架移动中，暂停发送调整。
        status->sequence <= lastCommandStatusSequence || // 回传已用于之前的发送，避免重复使用。
        status->received_at <= lastCommandAt ||  // 回传早于或等于上次写入完成时间，继续等待新回传。
        frame.left.timestamp_ms <= lastCommandMs ||  // 左图采集时间未晚于上次写入完成时间，跳过旧帧。
        frame.right.timestamp_ms <= lastCommandMs)   // 右图也必须来自上次写入完成之后，跳过旧帧。
        return std::nullopt;

    // 全部条件通过，返回本次检查使用的状态；调用方在发送成功后记录其序号。
    return status;
}

int runImageMode(const DartCongfig& config)
{
    // 图片模式单次处理一对；媒体路径和队列容量由同一份配置提供。
    VideoInput input(config.input, config.frameQueue.capacity);
    StereoFrameQueue& frames = input.frameQueue();
    if (!input.captureNext()) throw std::runtime_error("No image pair available");
    const auto frame = frames.GetFrame(config.frameQueue.maxTimestampDiffMs);
    // 图片后端同时提交两侧并赋予相同时间戳，此处应立即获得完整配对。
    if (!frame) throw std::runtime_error("Captured image pair could not be matched");

    // 检测器在输入有效后初始化，持有配置副本；连续处理时保留在循环外。
    std::cout << "Image pair acquired. Loading YOLO on " << config.yolo.device
              << "...\n" << std::flush;
    YOLOInference yolo(config.yolo);
    GuideLightDetect guideLight(config.guideLight);
    StereoscopicMeasurement measurement(config.leftCamera, config.rightCamera,
        config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
    LocationTransform locationtransform(config.locationTransform.R, config.locationTransform.T);

    const auto detections = processFrame(*frame, yolo, guideLight);
    std::cout << "Visual detection completed: left_frame=" << frame->left.frame_id
              << " right_frame=" << frame->right.frame_id << '\n';
    const auto measurements = measurement.RunMeasurement(detections);
    const auto locations = locationtransform.RunTransform(measurements);
    // 模式键只更新选择；其它按键或关闭全部窗口结束本次图片处理。
    const auto visual = makeLocationVisualize(locations);
    while (handleAimModeKey(StereoscopicMeasurementVisualize::Show(visual, 0))) {}
    cv::destroyAllWindows();
    return 0;
}

int runCameraMode(const DartCongfig& config)
{
    // 1. 注册退出信号并准备输入。先构造信号守卫，保证相机析构后才恢复信号处理。
    CameraStopSignals stopSignals;
    CameraInput input(config.leftCamera, config.rightCamera, config.frameQueue.capacity);
    StereoFrameQueue& frames = input.frameQueue();

    // 2. 模型只加载、预热一次，完成后再启动相机，避免初始化期间积压图像。
    std::cout << "Loading camera perception on " << config.yolo.device << "...\n" << std::flush;
    YOLOInference yolo(config.yolo);
    GuideLightDetect guideLight(config.guideLight);
    StereoscopicMeasurement stereoscopicmeasurement(config.leftCamera, config.rightCamera, config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
    LocationTransform locationtransform(config.locationTransform.R, config.locationTransform.T);
    YawSolver yawsolver;

    if (stopSignals.stopRequested()) return 0;
    input.StartCamera();
    std::cout << "Camera perception started. Press Ctrl+C to stop.\n" << std::flush;

    // 3. GetFrame 每次仅尝试一次配对；缺帧或时间差过大时等待下一次尝试。
    while (!stopSignals.stopRequested()) {
        // 串口故障中止本次运行；没有图像时仍处理待发送的预瞄准。
        if (communication && !communication->isRunning())
            throw std::runtime_error("MCU communication stopped: " + communication->lastError());
        handleAimModeKey(-1);
        const auto frame = frames.GetFrame(config.frameQueue.maxTimestampDiffMs);
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 4. 完成左右帧的 YOLO ROI 提取和引导灯检测，保持原图坐标及帧来源。
        const auto frameAimRevision = aimRevision;
        const auto detections = processFrame(*frame, yolo, guideLight);
        // 5. 双目测距；空结果刷新为空面板，不沿用之前的目标。
        const auto measurements = stereoscopicmeasurement.RunMeasurement(detections);
        // 6. 坐标转换。
        const auto locations = locationtransform.RunTransform(measurements);
        // 只提交位置画面，窗口事件和目标切换统一在 processFrame 中处理。
        StereoscopicMeasurementVisualize::Show(makeLocationVisualize(locations), -1);

        if (stopSignals.stopRequested()) break;

        // 7. 仅求解当前选择的目标；目标缺失或无效时返回空值。
        const auto yaw = yawsolver.RunSolve(locations, aim_base);

        // 8. 检查发送条件，通过后直接下发本帧 yaw。
        const auto status = checkSendConditions(yaw, *frame, frameAimRevision, config.communication);
        if (status) {
            // yaw 单位 rad、右正左负；本阶段开火许可与伸缩目标均为 0。
            communication->send({0, *yaw, 0},
                std::chrono::milliseconds(config.communication.writeTimeoutMs));
            lastCommandStatusSequence = status->sequence;
            lastCommandAt = std::chrono::steady_clock::now();
        }
    }

    // 9. 正常退出显式停止；初始化或检测抛异常时由 CameraInput 析构完成同样清理。
    input.StopCamera();
    std::cout << "Camera perception stopped.\n";
    return 0;
}
} // namespace

// 主入口只负责解析运行参数、选择对应配置与处理流程、统一报告异常。
// 返回码：0 表示处理完成、相机正常停止或帮助已显示；1 表示错误；2 表示输入后端待实现。
int main(int argc, char* argv[])
{
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }
    try {
        const auto configPath = parseConfigPath(argc, argv);
        std::cout << "Configuration: " << configPath << '\n';
        const DartCongfig config(configPath);
        communicationParams = config.communication;
        communication.reset();
        preAimPending = false;
        requestedAim.reset();
        aimRevision = lastCommandStatusSequence = 0;
        lastCommandAt = {};
        if (config.input.mode == InputMode::Camera && !config.communication.port.empty()) {
            communication = std::make_unique<dart_mcu::Communication>();
            communication->start(config.communication.port, config.communication.baud);
        }
        aim_base = true; // 每次启动默认瞄准基地，运行期间由 1/2 键切换。
        std::cout << "Aim mode: BASE. Focus a visualization window; 1=BASE, 2=OUTPOST." << std::endl;
        switch (config.input.mode) {
        case InputMode::Image:
            return runImageMode(config);
        case InputMode::Video:
            // TODO: 接入视频解码循环，复用配置与 processFrame。
            std::cerr << "Video decoding is not implemented yet.\n";
            return 2;
        case InputMode::Camera: {
            const int result = runCameraMode(config);
            communication.reset(); // 停止接收线程并关闭串口。
            return result;
        }
        }
    } catch (const std::invalid_argument& error) {
        communication.reset();
        std::cerr << error.what() << '\n';
        printUsage(argv[0]);
    } catch (const std::exception& error) {
        communication.reset();
        std::cerr << error.what() << '\n';
    }
    return 1;
}
