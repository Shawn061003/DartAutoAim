// ----------------------------------------------------------------
// 文件职责：组织配置、采集、检测、解算、显示与串口发送
// ----------------------------------------------------------------
// main：读取配置并分派 image / camera 模式，统一处理退出和异常。
// processFrame：两种模式共用的 YOLO → 引导灯检测 → 显示/按键流程。
// runImageMode / runCameraMode：分别组织单对图片处理和持续相机处理。
// checkSendConditions / canAimAtBase：分别负责发送前检查和BASE稳定后新帧放行。

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
#include "Visualization/VisualizeAdapter.h"

#include <algorithm>
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
// ----------------------------------------------------------------
// 共享运行状态：目标选择、待发预瞄准与上一条运动命令
// ----------------------------------------------------------------
// 下列业务状态只由主线程读写；串口接收线程通过Communication接口提供状态快照。
// 默认配置定位到源码目录；显式 --config 路径以进程工作目录为基准。
const std::filesystem::path kProjectRoot{DART_PROJECT_ROOT};

// true 选择基地，false 选择前哨站；只在主线程更新。
bool aim_base = true;

// 配置由 DartCongfig 解析；通信对象在相机流程结束后释放。
CommunicationParams communicationParams;
std::unique_ptr<dart_mcu::Communication> communication;
bool preAimPending = false; // 已请求、尚未发送的预瞄准；移动期间保留，稍后继续尝试。
std::optional<bool> requestedAim; // 最近一次按键请求；空值表示尚未收到目标选择按键。
std::uint64_t aimRevision = 0; // 目标请求版本；变化后，旧版本开始处理的图像失效。
std::uint64_t lastCommandStatusSequence = 0; // 上次发送使用的本地回传序号，不是电控任务编号。
std::chrono::steady_clock::time_point lastCommandAt{}; // 上次完整写入命令的时刻，不代表机构到位。

// ----------------------------------------------------------------
// 命令行入口：打印帮助与确定 YAML 配置路径
// ----------------------------------------------------------------
// 帮助只描述配置入口；相机、模型、串口等业务参数统一放在YAML中。
void printUsage(const char* program)
{
    std::cout << "Usage: " << program << " [--config path.yaml]\n"
              << "  Input mode and serial settings are configured in YAML.\n"
              << "  --help  Show this help.\n";
}

// 命令行只选择配置文件，输入模式和串口参数均由 YAML 提供。
std::string parseConfigPath(int argc, char* argv[])
{
    // 不带参数使用项目默认配置；显式指定时只接受 --config <路径>。
    if (argc == 1) return (kProjectRoot / "config/dart.yaml").string();
    if (argc == 3 && std::string(argv[1]) == "--config" &&
        std::string(argv[2]).size() > 0 && std::string(argv[2]).rfind("--", 0) != 0)
        return argv[2];
    throw std::invalid_argument("Expected --config path.yaml; configure input/communication in YAML");
}

// ----------------------------------------------------------------
// 目标切换与预瞄准：接收 1/2 按键，并在允许时发送一次预瞄准
// ----------------------------------------------------------------
// true选择基地，false选择前哨站；不在串口接收线程修改这些业务状态。
// 返回true仅表示收到模式键，图片模式据此继续等待；不表示命令已成功发送。
bool handleAimModeKey(int key)
{
    // step1  更新最近一次目标请求；相同请求不重复触发预瞄准。
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

    // step2  尝试执行待发预瞄准；静止、回传更新且晚于上一条命令时才允许发送。
    // 相机循环传入-1时也会走这里；条件不足时保留请求，下一轮继续尝试。
    if (preAimPending && communication) {
        const auto status = communication->latestStatus(
            std::chrono::milliseconds(communicationParams.statusMaxAgeMs));
        if (status && status->status.is_stationary == 1 &&
            status->sequence > lastCommandStatusSequence && status->received_at > lastCommandAt) {
            // 固定预瞄准角度单位为rad；本阶段开火许可和伸缩目标均为0。
            communication->send({0, aim_base ? 0.26 : 0.17, 0},
                std::chrono::milliseconds(communicationParams.writeTimeoutMs));
            // step3  完整写入成功后，记录使用过的回传和命令时间，再清除待发标记。
            lastCommandStatusSequence = status->sequence;
            lastCommandAt = std::chrono::steady_clock::now();
            preAimPending = false;
        }
    }
    return modeKey;
}

// ----------------------------------------------------------------
// 共用帧处理：YOLO 提取 ROI → 引导灯中心检测 → 显示与按键
// ----------------------------------------------------------------
// 输入为上游配好的左右双帧；输出保持全图坐标、各侧帧号和时间戳，供测距继续使用。
GuideLightDetect::StereoDetectResult processFrame(
    const StereoFrame& frame, YOLOInference& yolo, GuideLightDetect& guideLight)
{
    // step1  YOLO为左右原图生成候选ROI。
    const auto rois = yolo.RunYOLOInfer(frame.left, frame.right);

    // step2  在ROI中提取绿色轮廓并拟合中心，保留成功/失败状态。
    const auto detections = guideLight.RunDetection(frame.left, frame.right, rois);

    // step3  每对帧刷新一次detection窗口，并把按键交给目标切换逻辑。
    // 显示的数据转换在VisualizeAdapter中完成，此处只组织调用。
    const int key = visualization::showDetection(frame, rois, detections);
    handleAimModeKey(key);
    return detections;
}

// ----------------------------------------------------------------
// 最终发送检查：确认当前 yaw、目标版本、电控状态与帧时间仍有效
// ----------------------------------------------------------------
// 本函数只检查，不发送；不满足条件返回空值，通过时返回本次使用的状态快照。
// 调用方只在send成功后提交该快照序号，避免同一条回传重复授权发送。
std::optional<dart_mcu::ReceivedStatus> checkSendConditions(
    const std::optional<double>& yaw, const StereoFrame& frame,
    std::uint64_t frameAimRevision, const CommunicationParams& params)
{
    // step1  检查通信对象、待发预瞄准、yaw有效性及目标版本。
    if (!communication ||                       // 通信对象尚未创建，无法读取状态或发送命令。
        preAimPending ||                        // 预瞄准请求仍在等待发送，优先完成该请求。
        !yaw ||                                 // 当前所选目标缺失或无法求解，没有可发送的角度。
        frameAimRevision != aimRevision)         // 本帧处理期间切换过目标，丢弃本帧调整结果。
        return std::nullopt;

    // step2  获取未超过YAML中status_max_age_ms的最新有效电控回传。
    // 尚未收到有效状态、状态过期、通信停止或故障时，接口返回空值。
    const auto status = communication->latestStatus(
        std::chrono::milliseconds(params.statusMaxAgeMs));

    // step3  将上次命令写入完成时间转换为毫秒，与双帧的主机收帧时间比较。
    // 两者都使用 steady_clock；该时间仅表示串口写入完成，机构执行状态由回传提供。
    const auto lastCommandMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        lastCommandAt.time_since_epoch()).count();

    // step4  发送前再次确认镖架静止、回传未被复用、图像晚于上一条命令。
    // 这里检查的是镖架状态；场地BASE灯的稳定性由上游UpdateStability单独判断。
    if (!status ||                              // 没有有效且未过期的电控状态，当前运动状态未知。
        status->status.is_stationary != 1 ||     // 电控报告镖架移动中，暂停发送调整。
        status->sequence <= lastCommandStatusSequence || // 回传已用于之前的发送，避免重复使用。
        status->received_at <= lastCommandAt ||  // 回传早于或等于上次写入完成时间，继续等待新回传。
        frame.left.timestamp_ms <= lastCommandMs ||  // 左图采集时间未晚于上次写入完成时间，跳过旧帧。
        frame.right.timestamp_ms <= lastCommandMs)   // 右图也必须来自上次写入完成之后，跳过旧帧。
        return std::nullopt;

    // step5  返回本次放行所用的回传，供调用方成功发送后记录。
    return status;
}

// ----------------------------------------------------------------
// BASE精调放行：稳定标签成立后，继续等待确认时刻之后的新双帧
// ----------------------------------------------------------------
// 不在这里重复计算2px位移；stable由测距模块提供，此处只维护stableConfirmedAtMs。
bool canAimAtBase(const std::vector<LocationTransform::LocationTransformResult>& locations,
                  const StereoFrame& frame, std::optional<std::int64_t>& stableConfirmedAtMs)
{
    // step1  按类别寻找BASE，避免依赖结果数组顺序。
    const auto base = std::find_if(locations.begin(), locations.end(), [](const auto& location) {
        return location.TargetName == TargetClasses::BASE;
    });
    // step2  目标丢失或重新运动时，撤销之前的稳定确认时刻。
    if (base == locations.end() || !base->stable) {
        stableConfirmedAtMs.reset();
        return false;
    }

    // step3  首次判稳只记录当前确认时刻，本帧不放行。
    if (!stableConfirmedAtMs) {
        stableConfirmedAtMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return false; // 首次判稳只记录时刻，不能使用确认之前采集的图像。
    }

    // step4  后续仍稳定时，要求左右图都在确认之后采集；使用较早一侧进行检查。
    return std::min(frame.left.timestamp_ms, frame.right.timestamp_ms) > *stableConfirmedAtMs;
}

// ----------------------------------------------------------------
// 图片模式：处理单对图片，显示结果并等待退出
// ----------------------------------------------------------------
int runImageMode(const DartCongfig& config)
{
    // ----------------------------------------------------------------
    // step1  读取左右图片，并从队列取得匹配双帧
    // ----------------------------------------------------------------
    // 图片后端只产生一对图像；路径和队列容量均由YAML提供。
    VideoInput input(config.input, config.frameQueue.capacity);
    StereoFrameQueue& frames = input.frameQueue();
    if (!input.captureNext()) throw std::runtime_error("No image pair available");
    const auto frame = frames.GetFrame(config.frameQueue.maxTimestampDiffMs);
    // 图片后端同时提交两侧并赋予相同时间戳，此处应立即获得完整配对。
    if (!frame) throw std::runtime_error("Captured image pair could not be matched");

    // ----------------------------------------------------------------
    // step2  图片读取成功后，初始化检测、测距和坐标转换模块
    // ----------------------------------------------------------------
    std::cout << "Image pair acquired. Loading YOLO on " << config.yolo.device
              << "...\n" << std::flush;
    YOLOInference yolo(config.yolo);
    GuideLightDetect guideLight(config.guideLight);
    StereoscopicMeasurement measurement(config.leftCamera, config.rightCamera,
        config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
    LocationTransform locationtransform(config.locationTransform.R, config.locationTransform.T);

    // ----------------------------------------------------------------
    // step3  完成检测、双目测距和镖口坐标转换
    // ----------------------------------------------------------------
    // 单张图片只输出测距结果，不能积累用于BASE判稳的五次连续观测。
    const auto detections = processFrame(*frame, yolo, guideLight);
    std::cout << "Visual detection completed: left_frame=" << frame->left.frame_id
              << " right_frame=" << frame->right.frame_id << '\n';
    const auto measurements = measurement.RunMeasurement(detections);
    const auto locations = locationtransform.RunTransform(measurements);

    // ----------------------------------------------------------------
    // step4  显示位置结果，等待按键或关闭窗口后清理
    // ----------------------------------------------------------------
    // 模式键只更新选择；其它按键或关闭全部窗口结束本次图片处理。
    while (handleAimModeKey(visualization::showLocations(locations, 0))) {}
    cv::destroyAllWindows();
    return 0;
}

// ----------------------------------------------------------------
// 相机模式：持续采集，按观察资格 → 稳定性 → yaw放行的顺序处理
// ----------------------------------------------------------------
int runCameraMode(const DartCongfig& config)
{
    // ----------------------------------------------------------------
    // step1  准备退出信号、双相机输入和帧队列
    // ----------------------------------------------------------------
    // 先构造信号守卫，保证相机析构后才恢复原来的信号处理。
    CameraStopSignals stopSignals;
    CameraInput input(config.leftCamera, config.rightCamera, config.frameQueue.capacity);
    StereoFrameQueue& frames = input.frameQueue();

    // ----------------------------------------------------------------
    // step2  初始化检测、测距、坐标转换和 yaw 解算模块
    // ----------------------------------------------------------------
    // 模型仅在此处加载、预热一次，完成初始化后再启动相机。
    std::cout << "Loading camera perception on " << config.yolo.device << "...\n" << std::flush;
    YOLOInference yolo(config.yolo);
    GuideLightDetect guideLight(config.guideLight);
    StereoscopicMeasurement stereoscopicmeasurement(config.leftCamera, config.rightCamera, config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
    LocationTransform locationtransform(config.locationTransform.R, config.locationTransform.T);
    YawSolver yawsolver;

    // ----------------------------------------------------------------
    // step3  准备本轮观察状态和稳定性清理操作
    // ----------------------------------------------------------------
    // 分别保存镖架静止边界、BASE稳定确认时刻以及目标/命令版本。
    std::optional<std::int64_t> stationaryAfterMs;
    std::optional<std::int64_t> baseStableAfterMs;
    auto observationCommandAt = lastCommandAt;
    auto observationAimRevision = aimRevision;
    const auto milliseconds = [](auto time) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
    };
    const auto resetStability = [&] {
        stereoscopicmeasurement.ResetStability();
        baseStableAfterMs.reset();
    };

    // ----------------------------------------------------------------
    // step4  启动双相机采集
    // ----------------------------------------------------------------
    // 初始化期间已收到退出信号时，直接结束，不再启动设备。
    if (stopSignals.stopRequested()) return 0;
    input.StartCamera();
    std::cout << "Camera perception started. Press Ctrl+C to stop.\n" << std::flush;

    // 持续执行 step5～step14，直到收到退出信号。
    while (!stopSignals.stopRequested()) {
        // ----------------------------------------------------------------
        // step5  检查通信并处理待发送的预瞄准
        // ----------------------------------------------------------------
        // 串口故障中止运行；暂时没有图像时，也继续处理预瞄准请求。
        if (communication && !communication->isRunning())
            throw std::runtime_error("MCU communication stopped: " + communication->lastError());
        handleAimModeKey(-1);

        // ----------------------------------------------------------------
        // step6  目标或运动命令变化时，开启新一轮观察
        // ----------------------------------------------------------------
        // 清空旧的稳定性记录和静止边界，更新当前观察对应的版本。
        if (lastCommandAt != observationCommandAt || aimRevision != observationAimRevision) {
            resetStability();
            stationaryAfterMs.reset();
            observationCommandAt = lastCommandAt;
            observationAimRevision = aimRevision;
        }

        // ----------------------------------------------------------------
        // step7  确认镖架静止，记录允许使用图像的时间边界
        // ----------------------------------------------------------------
        // 移动中、状态未知或回传早于本轮命令时，清空观察状态。
        if (communication) {
            const auto motor = communication->latestStatus(
                std::chrono::milliseconds(config.communication.statusMaxAgeMs));
            if (!motor || motor->status.is_stationary != 1 || motor->received_at <= lastCommandAt) {
                resetStability();
                stationaryAfterMs.reset();
            } else if (!stationaryAfterMs) {
                // 仅在本轮首次确认静止时记录边界，后续心跳不持续推迟它。
                stationaryAfterMs = milliseconds(motor->received_at);
            }
        }

        // ----------------------------------------------------------------
        // step8  从队列取得时间配对的左右双帧
        // ----------------------------------------------------------------
        // 每次仅尝试一次配对；缺帧或时间差过大时，等待下一轮。
        const auto frame = frames.GetFrame(config.frameQueue.maxTimestampDiffMs);
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ----------------------------------------------------------------
        // step9  执行 YOLO 和引导灯检测
        // ----------------------------------------------------------------
        // 保留全图坐标、帧来源；检测显示窗口同时接收目标切换按键。
        const auto frameAimRevision = aimRevision;
        const auto detections = processFrame(*frame, yolo, guideLight);

        // ----------------------------------------------------------------
        // step10  判断本帧能否用于观察，并选择搜索或测距
        // ----------------------------------------------------------------
        // 处理期间状态须保持一致，双帧须来自静止确认之后；右图无有效目标时搜索。
        const bool sameContext = frameAimRevision == aimRevision &&
            observationCommandAt == lastCommandAt && !preAimPending;
        const bool capturedAfterStop = !communication || (stationaryAfterMs &&
            std::min(frame->left.timestamp_ms, frame->right.timestamp_ms) > *stationaryAfterMs);
        const bool freshObservation = sameContext && capturedAfterStop;
        const bool needSearch = std::none_of(detections.right.begin(), detections.right.end(),
            [](const auto& result) {
                return result.status == GuideLightDetect::DetectStatus::SUCCESS;
            });
        const bool canObserveStability = freshObservation && !needSearch;

        // ----------------------------------------------------------------
        // step11  双目测距、稳定性更新与镖口坐标转换
        // ----------------------------------------------------------------
        // 测距只输出几何结果；允许观察时更新BASE窗口，否则清空。
        std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult> measurements;
        if (!needSearch) measurements = stereoscopicmeasurement.RunMeasurement(detections);
        if (canObserveStability)
            stereoscopicmeasurement.UpdateStability(detections, measurements);
        else
            resetStability();
        const auto locations = locationtransform.RunTransform(measurements);

        // ----------------------------------------------------------------
        // step12  统一判断是否允许解算，并生成本帧 yaw
        // ----------------------------------------------------------------
        // BASE精调须稳定且使用确认后的新双帧；OUTPOST和搜索不等待BASE窗口。
        const bool canSolveYaw = freshObservation &&
            (needSearch || !aim_base || canAimAtBase(locations, *frame, baseStableAfterMs));
        std::optional<double> yaw;
        if (canSolveYaw) {
            if (needSearch) {
                const double searchYaw = yawsolver.RunSearch(detections.left);
                if (searchYaw != 0.0) yaw = searchYaw;
            } else {
                yaw = yawsolver.RunSolve(locations, aim_base);
            }
        }

        // ----------------------------------------------------------------
        // step13  刷新空间位置显示，并检查退出信号
        // ----------------------------------------------------------------
        // 搜索或测距无结果时显示空面板，清除上一帧位置。
        // 窗口事件和目标切换统一在 processFrame 中处理。
        visualization::showLocations(locations, -1);
        if (stopSignals.stopRequested()) break;

        // ----------------------------------------------------------------
        // step14  最终检查通信状态，发送 yaw 并记录命令
        // ----------------------------------------------------------------
        // 发送前再次校验回传有效期、镖架状态及数据时序，成功后记录序号和时间。
        const auto status = checkSendConditions(yaw, *frame, frameAimRevision, config.communication);
        if (status) {
            // yaw 单位 rad、右正左负；本阶段开火许可与伸缩目标均为 0。
            communication->send({0, *yaw, 0},
                std::chrono::milliseconds(config.communication.writeTimeoutMs));
            lastCommandStatusSequence = status->sequence;
            lastCommandAt = std::chrono::steady_clock::now();
        }
    }

    // ----------------------------------------------------------------
    // step15  停止双相机采集并结束本次运行
    // ----------------------------------------------------------------
    // 正常退出显式停止；异常路径由CameraInput析构完成设备清理。
    input.StopCamera();
    std::cout << "Camera perception stopped.\n";
    return 0;
}
} // namespace

// ----------------------------------------------------------------
// 程序入口：加载配置 → 准备运行状态 → 启动对应模式 → 统一处理异常
// ----------------------------------------------------------------
// 返回码：0表示正常完成/退出或显示帮助；1表示参数或运行错误；2表示视频后端尚未实现。
int main(int argc, char* argv[])
{
    // ----------------------------------------------------------------
    // step1  优先处理帮助请求，无需读取配置或打开设备
    // ----------------------------------------------------------------
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }
    try {
        // ----------------------------------------------------------------
        // step2  解析配置路径并读取完整YAML参数
        // ----------------------------------------------------------------
        const auto configPath = parseConfigPath(argc, argv);
        std::cout << "Configuration: " << configPath << '\n';
        const DartCongfig config(configPath);

        // ----------------------------------------------------------------
        // step3  初始化本次运行的通信、目标请求与命令历史
        // ----------------------------------------------------------------
        communicationParams = config.communication;
        communication.reset();
        preAimPending = false;
        requestedAim.reset();
        aimRevision = lastCommandStatusSequence = 0;
        lastCommandAt = {};

        // ----------------------------------------------------------------
        // step4  仅相机模式且串口路径非空时，启动后台通信
        // ----------------------------------------------------------------
        if (config.input.mode == InputMode::Camera && !config.communication.port.empty()) {
            communication = std::make_unique<dart_mcu::Communication>();
            communication->start(config.communication.port, config.communication.baud);
        }

        // ----------------------------------------------------------------
        // step5  设置默认瞄准目标，并输出运行提示
        // ----------------------------------------------------------------
        aim_base = true; // 每次启动默认瞄准基地，运行期间由 1/2 键切换。
        std::cout << "Aim mode: BASE. Focus a visualization window; 1=BASE, 2=OUTPOST." << std::endl;

        // ----------------------------------------------------------------
        // step6  按输入模式运行；相机模式结束后释放通信对象
        // ----------------------------------------------------------------
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
        // step7  参数异常：释放通信对象，报告错误并输出帮助。
        communication.reset();
        std::cerr << error.what() << '\n';
        printUsage(argv[0]);
    } catch (const std::exception& error) {
        // step7  其它运行异常：释放通信对象并报告错误。
        communication.reset();
        std::cerr << error.what() << '\n';
    }
    return 1;
}
