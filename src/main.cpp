#include "common/DartConfig.h"
#include "Acquisition/VideoInput.h"
#include "Acquisition/CameraInput.h"
#include "Acquisition/AcquireFrame.h"
#include "Acquisition/Quit.h"
#include "Perception/YOLOinfer.h"
#include "Perception/GuideLightDetect.h"
#include "Computation/StereoscopicMeasurement.h"
#include "Computation/LocationTransform.h"
#include "Visualization/Visualize.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
// 默认配置定位到源码目录；显式 --config 路径以进程工作目录为基准。
const std::filesystem::path kProjectRoot{DART_PROJECT_ROOT};

void printUsage(const char* program)
{
    std::cout << "Usage: " << program << " [--config path.yaml]\n"
              << "  Default: config/dart.yaml\n"
              << "  Select camera/video/image through YAML input.mode.\n"
              << "  --help  Show this help.\n";
}

// 命令行只选择配置文件，运行模式和算法参数统一由该 YAML 提供。
std::string parseConfigPath(int argc, char* argv[])
{
    if (argc == 1) return (kProjectRoot / "config/dart.yaml").string();
    if (argc == 3 && std::string(argv[1]) == "--config" &&
        std::string(argv[2]).size() > 0 && std::string(argv[2]).rfind("--", 0) != 0)
        return argv[2];
    throw std::invalid_argument("Expected --config path.yaml; set input.mode in YAML");
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
    DetectionVisualize::Show(makeYOLOVisualize(frame.left, rois.left),
                             makeYOLOVisualize(frame.right, rois.right),
                             makeGuideLightVisualize(frame.left, detections.left),
                             makeGuideLightVisualize(frame.right, detections.right));
    return detections;
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
    // 两个窗口均创建后再等待按键；GTK3 关闭全部窗口也可结束图片模式。
    StereoscopicMeasurementVisualize::Show(
        makeLocationVisualize(locations), 0);
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

    if (stopSignals.stopRequested()) return 0;
    input.StartCamera();
    std::cout << "Camera perception started. Press Ctrl+C to stop.\n" << std::flush;

    // 3. GetFrame 每次仅尝试一次配对；缺帧或时间差过大时等待下一次尝试。
    while (!stopSignals.stopRequested()) {
        const auto frame = frames.GetFrame(config.frameQueue.maxTimestampDiffMs);
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 4. 完成左右帧的 YOLO ROI 提取和引导灯检测，保持原图坐标及帧来源。
        const auto detections = processFrame(*frame, yolo, guideLight);
        // 5. 双目测距；空结果刷新为空面板，不沿用之前的目标。
        const auto measurements = stereoscopicmeasurement.RunMeasurement(detections);
        // 6. 坐标转换。
        const auto locations = locationtransform.RunTransform(measurements);
        StereoscopicMeasurementVisualize::Show(makeLocationVisualize(locations), 1);
    }

    // 7. 正常退出显式停止；初始化或检测抛异常时由 CameraInput 析构完成同样清理。
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
        switch (config.input.mode) {
        case InputMode::Image:
            return runImageMode(config);
        case InputMode::Video:
            // TODO: 接入视频解码循环，复用配置与 processFrame。
            std::cerr << "Video decoding is not implemented yet.\n";
            return 2;
        case InputMode::Camera:
            return runCameraMode(config);
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        printUsage(argv[0]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    return 1;
}
