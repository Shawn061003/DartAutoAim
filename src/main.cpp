// ----------------------------------------------------------------
// 文件职责：组织配置、采集、检测、解算、显示与串口发送
// ----------------------------------------------------------------
// main：读取配置并分派 image / camera 模式，统一处理退出和异常。
// processFrame：两种模式共用的 YOLO → 引导灯检测 → 显示/按键流程。
// runImageMode / runCameraMode：分别组织单对图片处理和持续相机处理。
// checkSendConditions：负责发送前的电控状态和帧时序检查。

#include "common/DartConfig.h"
#include "Acquisition/VideoInput.h"
#include "Acquisition/CameraInput.h"
#include "Acquisition/AcquireFrame.h"
#include "Acquisition/Quit.h"
#include "Perception/YOLOinfer.h"
#include "Perception/GuideLightDetect.h"
#include "Computation/StereoscopicMeasurement.h"
#include "Computation/LocationTransform.h"
#include "Computation/KalmanFilter.h"
#include "Computation/YawSolver.h"
#include "Communication/Communication.hpp"
#include "Visualization/VisualizeAdapter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <limits>
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

// 当前角度估计，单位rad、右转增大；以初始安装角和成功下发的相对yaw累计。
// 电控当前仅回传静止状态，此值未经过编码器角度反馈校正。
double current_angle = std::atan(4.0 / 482.5);
constexpr double kMaxYawAngleRad = 17.0 * CV_PI / 180.0; // 累计角度上限，等于17度允许发送。

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
// 统一yaw下发：先检查累计角度上限，写入成功后再提交角度与命令历史
// ----------------------------------------------------------------
// 调用方负责镖架静止和回传时序检查；预瞄准、搜索、精调均通过这里发送。
bool sendYawCommand(double relativeYaw, const dart_mcu::ReceivedStatus& status,
                    const CommunicationParams& params)
{
    const double nextAngle = current_angle + relativeYaw;
    if (!std::isfinite(nextAngle) || nextAngle > kMaxYawAngleRad) {
        std::cerr << "[Yaw] command rejected: resulting angle=" << nextAngle * 180.0 / CV_PI
                  << " deg, maximum=17 deg\n";
        return false;
    }
    if (relativeYaw <= 0.0001745329){
        return false;
    }
    communication->send({0, relativeYaw, 0}, std::chrono::milliseconds(params.writeTimeoutMs));
    // 被拒绝或send抛异常时均不提交；此处记录的是成功写入的命令角度估计。
    current_angle = nextAngle;
    lastCommandStatusSequence = status.sequence;
    lastCommandAt = std::chrono::steady_clock::now();
    return true;
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
            // 预设为相对初始参考方向的绝对角度：基地15.8度，前哨站10.5度。
            // 电控接收相对yaw，右转为正，因此发送“预设角度 - 当前角度”。
            const double preset_angle = (aim_base ? 15.8 : 10.5) * CV_PI / 180.0;
            const double relative_yaw = preset_angle - current_angle;
            // step3  统一检查17度上限；发送成功后才清除待发标记。
            if (sendYawCommand(relative_yaw, *status, communicationParams))
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
// 逐帧诊断：保留双帧来源、匹配几何和KF输出，便于比较滤波前后的数据
// ----------------------------------------------------------------
const char* targetClassName(TargetClasses target)
{
    switch (target) {
    case TargetClasses::BASE: return "BASE";
    case TargetClasses::OUTPOST: return "OUTPOST";
    default: return "UNKNOWN";
    }
}

void printMeasurements(const StereoFrame& frame,
    const GuideLightDetect::StereoDetectResult& detections,
    const std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult>& measurements,
    const std::vector<LocationTransform::LocationTransformResult>& locations)
{
    const auto targetName = aim_base ? TargetClasses::BASE : TargetClasses::OUTPOST;
    const auto successCount = [](const auto& results) {
        return static_cast<std::size_t>(std::count_if(results.begin(), results.end(), [](const auto& result) {
            return result.status == GuideLightDetect::DetectStatus::SUCCESS;
        }));
    };
    std::printf("[Measure][before-KF] frames=(%llu,%llu) timestamps_ms=(%lld,%lld) "
                "selected=%s detections_success/total=L:%zu/%zu,R:%zu/%zu matched=%zu\n",
        static_cast<unsigned long long>(frame.left.frame_id),
        static_cast<unsigned long long>(frame.right.frame_id),
        static_cast<long long>(frame.left.timestamp_ms), static_cast<long long>(frame.right.timestamp_ms),
        targetClassName(targetName), successCount(detections.left), detections.left.size(),
        successCount(detections.right), detections.right.size(), measurements.size());
    for (const auto& measurement : measurements) {
        const auto& p = measurement.RealLocation;
        std::printf("[Match] target=%s selected=%d pixels_L=(%.3f,%.3f) pixels_R=(%.3f,%.3f) "
                    "reprojection_px=(L:%.6f,R:%.6f,max:%.6f) right_camera_mm=(%.3f,%.3f,%.3f)\n",
            targetClassName(measurement.TargetName), measurement.TargetName == targetName ? 1 : 0,
            measurement.leftPixel.x, measurement.leftPixel.y, measurement.rightPixel.x, measurement.rightPixel.y,
            measurement.leftReprojectionErrorPx, measurement.rightReprojectionErrorPx,
            measurement.reprojectionErrorPx, p.x, p.y, p.z);
    }
    for (const auto& location : locations) {
        const auto& p = location.RealLocation;
        const double yawDeg = p.x == 0.0 && p.y == 0.0
            ? std::numeric_limits<double>::quiet_NaN() : std::atan2(p.x, p.y) * 180.0 / CV_PI;
        std::printf("[Measure][raw] target=%s selected=%d dart_mm=(%.3f,%.3f,%.3f) yaw_deg=%.6f\n",
            targetClassName(location.TargetName), location.TargetName == targetName ? 1 : 0,
            p.x, p.y, p.z, yawDeg);
    }
}

void printKalmanResult(const StereoFrame& frame,
    const std::optional<DartKalmanFilter::KalmanFilterResult>& result,
    const KalmanFilterParams& params, int lostFrames, bool canUpdate)
{
    const auto targetName = aim_base ? TargetClasses::BASE : TargetClasses::OUTPOST;
    if (!result) {
        std::printf("[KF][after] frames=(%llu,%llu) target=%s result=none can_update=%d "
                    "lost_frames=%d/%d reason=%s\n",
            static_cast<unsigned long long>(frame.left.frame_id),
            static_cast<unsigned long long>(frame.right.frame_id), targetClassName(targetName),
            canUpdate ? 1 : 0, lostFrames, params.lostResetFrames,
            lostFrames > 0 ? "selected_target_missing" : "update_not_allowed");
        return;
    }
    const auto& r = *result;
    const auto& p = r.XkHat;
    const double yawDeg = p.x == 0.0 && p.y == 0.0
        ? std::numeric_limits<double>::quiet_NaN() : std::atan2(p.x, p.y) * 180.0 / CV_PI;
    std::printf("[KF][after] frames=(%llu,%llu) target=%s dart_mm=(%.3f,%.3f,%.3f) "
                "yaw_deg=%.6f P_diag_mm2=(%.6f,%.6f,%.6f)\n",
        static_cast<unsigned long long>(frame.left.frame_id),
        static_cast<unsigned long long>(frame.right.frame_id), targetClassName(r.TargetName),
        p.x, p.y, p.z, yawDeg, r.Pk(0, 0), r.Pk(1, 1), r.Pk(2, 2));
    std::printf("[KF][gate] age=%zu/%d yaw_sigma_deg=%.6f/%.6f NIS=%.6f/%.6f "
                "yaw_residual_mean_deg=%+.6f abs_limit_deg=%.6f "
                "pass=(age:%d,precision:%d,innovation:%d,drift:%d) stable=%d required_streak=%d\n",
        r.age, params.minAge, r.yawSigmaDeg, params.maxYawErrorDeg, r.nis, params.maxNis,
        r.yawResidualMeanDeg, params.maxYawErrorDeg,
        r.age >= static_cast<std::size_t>(params.minAge) ? 1 : 0,
        std::isfinite(r.yawSigmaDeg) && r.yawSigmaDeg <= params.maxYawErrorDeg ? 1 : 0,
        std::isfinite(r.nis) && r.nis >= 0.0 && r.nis <= params.maxNis ? 1 : 0,
        std::isfinite(r.yawResidualMeanDeg) && std::abs(r.yawResidualMeanDeg) <= params.maxYawErrorDeg ? 1 : 0,
        r.stable ? 1 : 0, params.stableFrames);
}

// ----------------------------------------------------------------
// 当前目标滤波：按aim_base选择测量，连续缺失达到阈值时清除KF历史
// ----------------------------------------------------------------
// 每对已处理双帧调用一次；不以空轮询计数，也不在目标缺失时输出旧位置。
std::optional<DartKalmanFilter::KalmanFilterResult> filterSelectedTarget(
    const std::vector<LocationTransform::LocationTransformResult>& locations,
    DartKalmanFilter& kalmanFilter, int& lostFrames, int lostResetFrames, bool canUpdate)
{
    const auto targetName = aim_base ? TargetClasses::BASE : TargetClasses::OUTPOST;
    const auto target = std::find_if(locations.begin(), locations.end(),
        [targetName](const auto& location) { return location.TargetName == targetName; });
    if (target == locations.end()) {
        kalmanFilter.InvalidateStability();
        // 计数达到阈值后保持饱和，仅在首次达到阈值时重置。
        if (lostFrames < lostResetFrames && ++lostFrames == lostResetFrames) {
            std::printf("[KF][reset] target=%s reason=consecutive_missing lost_frames=%d\n",
                targetClassName(targetName), lostFrames);
            kalmanFilter.Reset();
        }
        return std::nullopt;
    }

    lostFrames = 0;
    if (!canUpdate) return std::nullopt;
    return kalmanFilter.RunKalmanFilter(*target);
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
    // 使用本帧检测结果完成双目测距和坐标转换。
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
// 相机模式：持续采集，完成检测、测距、yaw解算和发送检查
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
    // step2  初始化检测、测距、坐标转换、KF和 yaw 解算模块
    // ----------------------------------------------------------------
    // 模型仅在此处加载、预热一次，完成初始化后再启动相机。
    std::cout << "Loading camera perception on " << config.yolo.device << "...\n" << std::flush;
    YOLOInference yolo(config.yolo);
    GuideLightDetect guideLight(config.guideLight);
    StereoscopicMeasurement stereoscopicmeasurement(config.leftCamera, config.rightCamera, config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
    LocationTransform locationtransform(config.locationTransform.R, config.locationTransform.T);
    YawSolver yawsolver;
    DartKalmanFilter kalmanFilter(config.kalmanFilter);
    int kalmanLostFrames = 0;
    auto kalmanCommandAt = lastCommandAt;
    auto kalmanAimRevision = aimRevision;

    // ----------------------------------------------------------------
    // step3  启动双相机采集
    // ----------------------------------------------------------------
    // 初始化期间已收到退出信号时，直接结束，不再启动设备。
    if (stopSignals.stopRequested()) return 0;
    input.StartCamera();
    std::cout << "Camera perception started. Press Ctrl+C to stop.\n" << std::flush;

    // 持续执行 step4～step12，直到收到退出信号。
    while (!stopSignals.stopRequested()) {
        // ----------------------------------------------------------------
        // step4  检查通信并处理待发送的预瞄准
        // ----------------------------------------------------------------
        // 串口故障中止运行；暂时没有图像时，也继续处理预瞄准请求。
        if (communication && !communication->isRunning())
            throw std::runtime_error("MCU communication stopped: " + communication->lastError());
        handleAimModeKey(-1);

        // ----------------------------------------------------------------
        // step5  从队列取得时间配对的左右双帧
        // ----------------------------------------------------------------
        // 每次仅尝试一次配对；缺帧或时间差过大时，等待下一轮。
        const auto frame = frames.GetFrame(config.frameQueue.maxTimestampDiffMs);
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ----------------------------------------------------------------
        // step6  执行 YOLO 和引导灯检测
        // ----------------------------------------------------------------
        // 保留全图坐标、帧来源；检测显示窗口同时接收目标切换按键。
        const auto frameAimRevision = aimRevision;
        const auto detections = processFrame(*frame, yolo, guideLight);

        // ----------------------------------------------------------------
        // step7  根据右图检测结果选择搜索或测距
        // ----------------------------------------------------------------
        // 右图集合为空或全部为FAILED时进入搜索。
        const bool needSearch = std::none_of(detections.right.begin(), detections.right.end(),
            [](const auto& result) {
                return result.status == GuideLightDetect::DetectStatus::SUCCESS;
            });

        // ----------------------------------------------------------------
        // step8  双目测距与镖口坐标转换
        // ----------------------------------------------------------------
        // 右图有有效目标时执行测距；搜索时使用空位置结果。
        std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult> measurements;
        if (!needSearch) measurements = stereoscopicmeasurement.RunMeasurement(detections);
        const auto locations = locationtransform.RunTransform(measurements);

        // 按类别打印本帧全部匹配，不假设BASE/OUTPOST与数组下标固定对应。
        printMeasurements(*frame, detections, measurements, locations);

        // ----------------------------------------------------------------
        // step9  卡尔曼滤波定位
        // ----------------------------------------------------------------
        // 切换目标或下发运动命令后，旧位置不再属于当前观测上下文。
        if (kalmanCommandAt != lastCommandAt || kalmanAimRevision != aimRevision) {
            std::printf("[KF][reset] reason=target_or_command_changed\n");
            kalmanFilter.Reset();
            kalmanLostFrames = 0;
            kalmanCommandAt = lastCommandAt;
            kalmanAimRevision = aimRevision;
        }
        bool canUpdateKalman = !preAimPending && frameAimRevision == aimRevision;
        if (communication) {
            const auto motor = communication->latestStatus(
                std::chrono::milliseconds(config.communication.statusMaxAgeMs));
            const auto commandMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                lastCommandAt.time_since_epoch()).count();
            canUpdateKalman = canUpdateKalman && motor && motor->status.is_stationary == 1 &&
                motor->received_at > lastCommandAt && frame->left.timestamp_ms > commandMs &&
                frame->right.timestamp_ms > commandMs;
        }
        // 镖架运动、状态未知或图像已过时，不融合该帧，避免历史跨坐标系累积。
        if (!canUpdateKalman) kalmanFilter.Reset();
        const auto filteredTarget = filterSelectedTarget(locations, kalmanFilter,
            kalmanLostFrames, config.kalmanFilter.lostResetFrames, canUpdateKalman);
        printKalmanResult(*frame, filteredTarget, config.kalmanFilter, kalmanLostFrames, canUpdateKalman);
        std::vector<LocationTransform::LocationTransformResult> filteredLocations;
        if (filteredTarget)
            filteredLocations.push_back({filteredTarget->XkHat, filteredTarget->TargetName});

        // ----------------------------------------------------------------
        // step10  根据搜索或滤波结果生成本帧 yaw
        // ----------------------------------------------------------------
        // 目标缺失时不生成yaw，最终发送条件由checkSendConditions统一检查。
        std::optional<double> yaw;
        if (needSearch) {
            const double searchYaw = yawsolver.RunSearch(detections.left);
            if (searchYaw != 0.0) yaw = searchYaw;
        } else if (filteredTarget) {
            yaw = yawsolver.RunSolve(filteredLocations, aim_base);
            if (yaw)
                std::printf("[YawSolver] target at %.6f deg, requested yaw %+.6f deg\n",
                            (current_angle + *yaw) * 180.0 / CV_PI, *yaw * 180.0 / CV_PI);
        }

        // ----------------------------------------------------------------
        // step11  刷新空间位置显示，并检查退出信号
        // ----------------------------------------------------------------
        // 只显示当前所选目标的KF输出；本帧无滤波结果时清空，不显示原始或历史位置。
        // 窗口事件和目标切换统一在 processFrame 中处理。
        visualization::showLocations(filteredLocations, -1);
        if (stopSignals.stopRequested()) break;

        // ----------------------------------------------------------------
        // step12  最终检查通信状态，发送 yaw 并记录命令
        // ----------------------------------------------------------------
        // 发送前再次校验回传有效期、镖架状态及数据时序，成功后记录序号和时间。
        const auto status = checkSendConditions(yaw, *frame, frameAimRevision, config.communication);
        // 视觉精调须通过KF门控；搜索分支没有三维观测，仍使用原有发送条件。
        if (status && (needSearch || (filteredTarget && filteredTarget->stable))) {
            // yaw 单位 rad、右正左负；本阶段开火许可与伸缩目标均为 0。
            sendYawCommand(*yaw, *status, config.communication);
        }
    }

    // ----------------------------------------------------------------
    // step13  停止双相机采集并结束本次运行
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
        current_angle = std::atan(4.0 / 482.5); // 每次启动从已知初始安装角开始累计。

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
