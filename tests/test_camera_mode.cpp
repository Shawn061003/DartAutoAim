// 直接验证主函数中的相机分支；设备、感知和 GUI 边界使用替身，不访问硬件或窗口。
#define main cameraModeProgramMain
#include "../src/main.cpp"
#undef main

#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <unistd.h>
#include <opencv2/calib3d.hpp>

namespace {
int yoloLoads = 0, yoloCalls = 0, detectionCalls = 0, stopCalls = 0, yoloShows = 0, guideShows = 0;
int measurementShows = 0;
int detectionKey = -1, measurementKey = -1;
bool active = false, failStart = false, failDetection = false, emptyResults = false;
bool controlRun = false;
float controlMotionPerFrame = 0;
int controlFrameLimit = 2;
StereoFrameQueue* controlQueue = nullptr;
GuideLightDetect::StereoDetectResult controlDetections;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void reset()
{
    yoloLoads = yoloCalls = detectionCalls = stopCalls = yoloShows = guideShows = 0;
    measurementShows = 0;
    detectionKey = measurementKey = -1;
    active = failStart = failDetection = emptyResults = false;
}
void priorSignalHandler(int) {}

// 使用伪终端模拟电控，只验证主程序的状态门控与下发内容，不访问真实设备。
class ControlMcu {
public:
    ControlMcu()
    {
        master_ = ::posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        require(master_ >= 0, "Cannot open control PTY");
        if (::grantpt(master_) != 0 || ::unlockpt(master_) != 0) {
            ::close(master_);
            throw std::runtime_error("Cannot initialize control PTY");
        }
        char name[256]{};
        if (::ptsname_r(master_, name, sizeof(name)) != 0) {
            ::close(master_);
            throw std::runtime_error("Cannot get control PTY name");
        }
        port = name;
    }
    ~ControlMcu() { if (master_ >= 0) ::close(master_); }
    void disconnect() { ::close(master_); master_ = -1; }

    template<class Predicate>
    static void waitFor(Predicate predicate)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!predicate()) {
            require(std::chrono::steady_clock::now() < deadline, "Control status wait timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void publish(bool stationary)
    {
        const auto previous = communication->latestStatus(std::chrono::seconds(10));
        const auto sequence = previous ? previous->sequence : 0;
        const dart_mcu::ReceiveFrame status = stationary
            ? dart_mcu::ReceiveFrame{0x42,0x52,3,4,1,0,0,0,0x9D}
            : dart_mcu::ReceiveFrame{0x42,0x52,3,4,0,0,0,0,0x12};
        require(::write(master_, status.data(), status.size()) ==
                static_cast<ssize_t>(status.size()), "Cannot publish control status");
        waitFor([&] {
            const auto next = communication->latestStatus(std::chrono::seconds(10));
            return next && next->sequence > sequence;
        });
    }
    bool hasCommand()
    {
        pollfd descriptor{master_, POLLIN, 0};
        return ::poll(&descriptor, 1, 10) > 0 && (descriptor.revents & POLLIN);
    }
    void expectCommand(double yaw)
    {
        dart_mcu::Frame frame{};
        std::size_t count = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (count < frame.size()) {
            require(std::chrono::steady_clock::now() < deadline, "Control command wait timed out");
            pollfd descriptor{master_, POLLIN, 0};
            if (::poll(&descriptor, 1, 20) <= 0) continue;
            const auto received = ::read(master_, frame.data() + count, frame.size() - count);
            require(received > 0, "Cannot read control command");
            count += static_cast<std::size_t>(received);
        }
        dart_mcu::Command command;
        require(dart_mcu::decode_command(frame.data(), frame.size(), command), "Invalid control command");
        require(command.fire_allowed == 0 && command.stretch_target_mm == 0,
                "Aim command changed fire/stretch fields");
        require(std::abs(command.yaw_error_rad - yaw) < 1e-12, "Control yaw value/sign changed");
        require(!hasCommand(), "One request produced duplicate commands");
    }
    std::string port;
private:
    int master_ = -1;
};

void testSelectedTargetFilter()
{
    KalmanFilterParams params;
    params.Q = cv::Matx33d::eye();
    params.R = cv::Matx33d::eye() * 4.0;
    params.P0 = cv::Matx33d::eye() * 9.0;
    DartKalmanFilter kf(params);
    int lostFrames = 0;
    LocationTransform::LocationTransformResult base{{10, 20, 30}, TargetClasses::BASE};
    LocationTransform::LocationTransformResult outpost{{100, 200, 300}, TargetClasses::OUTPOST};
    aim_base = true;
    auto filtered = filterSelectedTarget({outpost, base}, kf, lostFrames, 5, true);
    require(filtered && filtered->TargetName == TargetClasses::BASE &&
            filtered->XkHat == base.RealLocation, "KF selected wrong target or depended on ordering");
    base.RealLocation = {24, 34, 44};
    filtered = filterSelectedTarget({outpost, base}, kf, lostFrames, 5, true);
    require(filtered && cv::norm(filtered->XkHat - cv::Point3d(20, 30, 40)) < 1e-10,
            "Selected target did not use filtered coordinates");

    for (int i = 1; i <= 4; ++i) {
        require(!filterSelectedTarget({outpost}, kf, lostFrames, 5, true) && lostFrames == i,
                "Another target hid the selected target loss or returned stale coordinates");
    }
    base.RealLocation = {38, 48, 58};
    filtered = filterSelectedTarget({base}, kf, lostFrames, 5, true);
    require(filtered && lostFrames == 0 && filtered->XkHat.x > 20 && filtered->XkHat.x < 38,
            "Short loss reset KF history or failed to clear the loss counter");
    for (int i = 1; i <= 5; ++i)
        require(!filterSelectedTarget({}, kf, lostFrames, 5, true) && lostFrames == i,
                "Continuous loss did not reach reset threshold");
    base.RealLocation = {70, 80, 90};
    filtered = filterSelectedTarget({base}, kf, lostFrames, 5, true);
    require(filtered && filtered->XkHat == base.RealLocation && lostFrames == 0,
            "KF did not reinitialize immediately after five missing frames");

    aim_base = false;
    filtered = filterSelectedTarget({base, outpost}, kf, lostFrames, 5, true);
    require(filtered && filtered->TargetName == TargetClasses::OUTPOST &&
            filtered->XkHat == outpost.RealLocation, "OUTPOST reused BASE filter history");
    for (int i = 0; i < 20; ++i)
        require(!filterSelectedTarget({base}, kf, lostFrames, 5, true), "Missing OUTPOST returned BASE");
    require(lostFrames == 5, "Loss count did not saturate");
    require(!filterSelectedTarget({outpost}, kf, lostFrames, 5, false) && lostFrames == 0,
            "Disallowed frame updated KF or counted visible target as missing");
    filtered = filterSelectedTarget({outpost}, kf, lostFrames, 5, true);
    require(filtered && filtered->XkHat == outpost.RealLocation,
            "Disallowed measurement polluted KF history");
    aim_base = true;
}

void testYawAngleLimit()
{
    ControlMcu mcu;
    communicationParams = CommunicationParams{};
    communication = std::make_unique<dart_mcu::Communication>();
    communication->start(mcu.port, communicationParams.baud);
    lastCommandStatusSequence = 0;
    lastCommandAt = {};
    current_angle = 16.0 * CV_PI / 180.0;
    mcu.publish(true);
    auto status = communication->latestStatus(std::chrono::seconds(1));
    require(status.has_value(), "Missing angle-limit fixture status");
    const double originalAngle = current_angle;
    require(!sendYawCommand(2.0 * CV_PI / 180.0, *status, communicationParams),
            "Command exceeding 17 degrees was accepted");
    require(!mcu.hasCommand() && current_angle == originalAngle &&
            lastCommandStatusSequence == 0 && lastCommandAt == std::chrono::steady_clock::time_point{},
            "Rejected yaw wrote bytes or changed command history");

    // 等于上限仍可下发；确认16+1度不会被误当成越界。
    const double oneDegree = CV_PI / 180.0;
    require(sendYawCommand(oneDegree, *status, communicationParams), "17 degree boundary was rejected");
    mcu.expectCommand(oneDegree);
    require(std::abs(current_angle - 17.0 * CV_PI / 180.0) < 1e-12,
            "Accepted command did not accumulate angle");
    current_angle = 17.0 * CV_PI / 180.0;
    mcu.publish(true);
    status = communication->latestStatus(std::chrono::seconds(1));
    require(sendYawCommand(0.0, *status, communicationParams), "Exactly 17 degrees was rejected");
    mcu.expectCommand(0.0);

    const auto previousSequence = lastCommandStatusSequence;
    const auto previousAt = lastCommandAt;
    require(!sendYawCommand(oneDegree, *status, communicationParams) && !mcu.hasCommand(),
            "Second positive command crossed limit");
    require(!sendYawCommand(std::numeric_limits<double>::infinity(), *status, communicationParams) &&
            !sendYawCommand(std::numeric_limits<double>::quiet_NaN(), *status, communicationParams),
            "Nonfinite command bypassed upper limit");
    require(current_angle == 17.0 * CV_PI / 180.0 && lastCommandStatusSequence == previousSequence &&
            lastCommandAt == previousAt && !mcu.hasCommand(), "Invalid command changed state");

    // 允许向较小角度调整；不要因为当前处于上限就禁止反向恢复。
    mcu.publish(true);
    status = communication->latestStatus(std::chrono::seconds(1));
    require(sendYawCommand(-oneDegree, *status, communicationParams), "Negative correction was rejected");
    mcu.expectCommand(-oneDegree);
    const double beforeFailure = current_angle;
    const auto sequenceBeforeFailure = lastCommandStatusSequence;
    const auto timeBeforeFailure = lastCommandAt;
    mcu.disconnect();
    ControlMcu::waitFor([&] { return !communication->isRunning(); });
    bool failed = false;
    try { sendYawCommand(-oneDegree, *status, communicationParams); }
    catch (const std::runtime_error&) { failed = true; }
    require(failed && current_angle == beforeFailure && lastCommandStatusSequence == sequenceBeforeFailure &&
            lastCommandAt == timeBeforeFailure, "Send failure committed angle or history");
    communication.reset();
}

void testAimControl()
{
    ControlMcu mcu;
    communicationParams = CommunicationParams{};
    communication = std::make_unique<dart_mcu::Communication>();
    communication->start(mcu.port, communicationParams.baud);
    aim_base = true;
    const double initialAngle = std::atan(4.0 / 482.5);
    const double baseAngle = 15.8 * CV_PI / 180.0;
    const double outpostAngle = 10.5 * CV_PI / 180.0;
    current_angle = initialAngle;
    requestedAim.reset();
    preAimPending = false;
    lastCommandStatusSequence = aimRevision = 0;
    lastCommandAt = {};

    handleAimModeKey('1');
    require(preAimPending && !mcu.hasCommand(), "Unknown status allowed pre-aim");
    require(current_angle == initialAngle, "Unknown status changed current angle");
    mcu.publish(false);
    handleAimModeKey(-1);
    require(preAimPending && !mcu.hasCommand(), "Moving state allowed pre-aim");
    handleAimModeKey('2');
    handleAimModeKey('1');
    handleAimModeKey('2');
    require(!aim_base && !mcu.hasCommand(), "Pending target did not keep latest selection");
    mcu.publish(true);
    handleAimModeKey(-1);
    mcu.expectCommand(outpostAngle - initialAngle);
    require(std::abs(current_angle - outpostAngle) < 1e-12, "OUTPOST angle was not accumulated");
    require(!preAimPending, "Queued pre-aim was not cleared");
    handleAimModeKey('2');
    require(!preAimPending && !mcu.hasCommand(), "Repeated mode key reissued command");
    handleAimModeKey('1');
    require(preAimPending && !mcu.hasCommand(), "One status authorized two commands");
    mcu.publish(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(communicationParams.statusMaxAgeMs + 20));
    handleAimModeKey(-1);
    require(preAimPending && !mcu.hasCommand(), "Expired status allowed pre-aim");
    require(std::abs(current_angle - outpostAngle) < 1e-12, "Expired status changed current angle");
    mcu.publish(true);
    handleAimModeKey(-1);
    mcu.expectCommand(baseAngle - outpostAngle);
    require(std::abs(current_angle - baseAngle) < 1e-12, "BASE switch reused an absolute command");

    // 切回前哨站应发送负相对角度，验证连续切换会使用上次累计的方向。
    mcu.publish(true);
    handleAimModeKey('2');
    mcu.expectCommand(outpostAngle - baseAngle);
    require(std::abs(current_angle - outpostAngle) < 1e-12, "Reverse switch did not update angle");

    mcu.disconnect();
    ControlMcu::waitFor([&] { return !communication->isRunning(); });
    require(!communication->latestStatus(std::chrono::seconds(1)), "Disconnected status remained valid");
    communication.reset();
    requestedAim.reset();
    preAimPending = false;
}
}
CameraDevice::CameraDevice(const CameraParams& params, CameraSide side)
    : params_(params), side_(side) {}
CameraDevice::~CameraDevice() noexcept = default;

CameraInput::CameraInput(const CameraParams& left, const CameraParams& right,
                         std::size_t capacity, std::uint32_t timeout)
    : leftCamera_(left, CameraSide::Left), rightCamera_(right, CameraSide::Right),
      queue_(std::make_unique<StereoFrameQueue>(capacity)), grabTimeoutMs_(timeout) {}
CameraInput::~CameraInput() noexcept { StopCamera(); }
StereoFrameQueue& CameraInput::frameQueue() noexcept { return *queue_; }
void CameraInput::StartCamera()
{
    require(yoloLoads == 1, "Camera started before model initialization");
    active = true;
    if (failStart) throw std::runtime_error("injected startup failure");
    CameraFrame left{cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 0)), 1, 0, CameraSide::Left};
    CameraFrame right{left.image, 9, 100, CameraSide::Right};
    if (controlRun) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        right.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    queue_->push(left, right);
    // 第一次 GetFrame 丢弃旧左帧并返回 nullopt；下一次应取得这张新左帧。
    left.frame_id = 2;
    left.timestamp_ms = right.timestamp_ms;
    queue_->pushLeft(left);
    controlQueue = queue_.get();
}
void CameraInput::StopCamera() noexcept { ++stopCalls; active = false; }

struct YOLOInference::ModelRuntime {};
YOLOInference::YOLOInference(const YOLOParams& params) : params_(params) { ++yoloLoads; }
YOLOInference::~YOLOInference() = default;
YOLOInference::StereoYOLOResult YOLOInference::RunYOLOInfer(
    const CameraFrame& left, const CameraFrame& right)
{
    ++yoloCalls;
    require(active, "Perception ran without active camera");
    require(left.frame_id == 2 && right.frame_id == 9 && left.timestamp_ms == right.timestamp_ms,
            "Unmatched/stale frames entered perception");
    require(left.camera_side == CameraSide::Left && right.camera_side == CameraSide::Right,
            "Camera side was lost");
    if (emptyResults) return {};
    return {{{{0, 0, 8, 8}, 0.9, left.frame_id, left.timestamp_ms, left.camera_side}},
            {{{0, 0, 8, 8}, 0.9, right.frame_id, right.timestamp_ms, right.camera_side}}};
}
GuideLightDetect::GuideLightDetect(const GuideLightParams&) {}
GuideLightDetect::StereoDetectResult GuideLightDetect::RunDetection(
    const CameraFrame& left, const CameraFrame& right, const YOLOInference::StereoYOLOResult& rois)
{
    ++detectionCalls;
    require(yoloCalls == detectionCalls, "Perception stages ran out of order");
    if (failDetection) throw std::runtime_error("injected perception failure");
    if (!emptyResults)
        require(rois.left.at(0).frame_id == left.frame_id && rois.right.at(0).frame_id == right.frame_id,
                "YOLO results were not forwarded to traditional vision");
    // 用真实退出信号验证本次检测完成后正常退出；FAILED 和空集合均不能阻碍退出。
    if (!controlRun || detectionCalls == controlFrameLimit) std::raise(SIGINT);
    if (controlRun) {
        auto detections = controlDetections;
        for (auto& d : detections.left) {
            d.timestamp_ms = left.timestamp_ms;
            d.frame_id = left.frame_id;
            d.CenterPoint.x += controlMotionPerFrame*detectionCalls;
        }
        for (auto& d : detections.right) {
            d.timestamp_ms = right.timestamp_ms;
            d.frame_id = right.frame_id;
            d.CenterPoint.x += controlMotionPerFrame*detectionCalls;
        }
        return detections;
    }
    StereoDetectResult result;
    if (!emptyResults) {
        GuideLightDetectResult failed;
        failed.status = DetectStatus::FAILED;
        result.left.push_back(failed);
    }
    return result;
}

int DetectionVisualize::Show(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                              const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                              const GuideLightDetectVisualize::FrameVisualize& left,
                              const GuideLightDetectVisualize::FrameVisualize& right, int delayMs)
{
    ++yoloShows;
    ++guideShows;
    require(delayMs > 0, "Continuous camera visualization must not wait indefinitely");
    require(yoloLeft.frame.frame_id == 2 && yoloRight.frame.frame_id == 9,
            "Visualization received mismatched frame metadata");
    require(yoloLeft.targets.size() == (emptyResults ? 0u : 1u) &&
            yoloRight.targets.size() == (emptyResults ? 0u : 1u), "Visualization lost YOLO results");
    require(detectionCalls == guideShows, "Guide-light visualization ran before detection");
    require(left.frame.frame_id == 2 && right.frame.frame_id == 9,
            "Guide-light visualization received mismatched frames");
    if (controlRun) {
        require(left.targets.size() == 1 && right.targets.size() == 1 &&
                left.targets.front().success && right.targets.front().success,
                "Successful control detections lost");
    } else {
        require(left.targets.size() == (emptyResults ? 0u : 1u) && right.targets.empty(),
                "Guide-light visualization lost empty/failed results");
        if (!emptyResults)
            require(!left.targets.front().success, "Failed detection displayed as successful");
    }
    return detectionKey;
}

int StereoscopicMeasurementVisualize::Show(const FrameVisualize& data, int delayMs)
{
    ++measurementShows;
    require(delayMs == -1, "Measurement visualization must not read keyboard input");
    require(guideShows == measurementShows, "Measurement shown before detection");
    if (controlRun) {
        const auto motor = communication->latestStatus(
            std::chrono::milliseconds(communicationParams.statusMaxAgeMs));
        const bool hasFilteredTarget = motor && motor->status.is_stationary == 1 &&
            motor->received_at > lastCommandAt;
        require(data.targets.size() == (hasFilteredTarget ? 1u : 0u),
                "Location display used raw or stale data without a current KF result");
        if (measurementShows < controlFrameLimit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            CameraFrame left{cv::Mat(8, 8, CV_8UC3, cv::Scalar(0, 0, 0)), 2, now, CameraSide::Left};
            CameraFrame right{left.image, 9, now, CameraSide::Right};
            controlQueue->push(left, right);
        }
    } else {
        require(data.targets.empty(), "Failed/empty detections created a false measurement");
    }
    return measurementKey;
}

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "Expected config path");
        testYawAngleLimit();
        testAimControl();
        testSelectedTargetFilter();
        require(communication == nullptr && !preAimPending, "Control session leaked state");
        DartCongfig config(argv[1]);
        aim_base = true;
        require(!handleAimModeKey(-1) && aim_base, "No key changed aim mode");
        require(handleAimModeKey('2') && !aim_base, "2 did not select OUTPOST");
        require(!handleAimModeKey('x') && !aim_base, "Unrelated key changed aim mode");
        require(handleAimModeKey('1') && aim_base, "1 did not select BASE");
        // 已知3D点经过测距、镖口坐标转换，再送入独立显示接口。
        const cv::Point3d expectedRight(500, -800, 24000);
        const cv::Matx33d rotation = config.stereoscopicmeasurement.R;
        const cv::Vec3d translation = config.stereoscopicmeasurement.T;
        const cv::Point3d expectedLeft(rotation.t() * (cv::Vec3d(expectedRight) - translation));
        std::vector<cv::Point2d> leftPixel, rightPixel;
        cv::projectPoints(std::vector<cv::Point3d>{expectedLeft}, cv::Vec3d(), cv::Vec3d(),
                          config.leftCamera.cameraMatrix, config.leftCamera.distCoeffs, leftPixel);
        cv::projectPoints(std::vector<cv::Point3d>{expectedRight}, cv::Vec3d(), cv::Vec3d(),
                          config.rightCamera.cameraMatrix, config.rightCamera.distCoeffs, rightPixel);
        GuideLightDetect::StereoDetectResult synthetic;
        GuideLightDetect::GuideLightDetectResult detected;
        detected.status = GuideLightDetect::DetectStatus::SUCCESS;
        detected.CenterPoint = leftPixel[0];
        synthetic.left.push_back(detected);
        detected.CenterPoint = rightPixel[0];
        synthetic.right.push_back(detected);
        StereoscopicMeasurement estimator(config.leftCamera, config.rightCamera,
            config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
        const auto measured = estimator.RunMeasurement(synthetic);
        require(measured.size() == 1 && measured[0].TargetName == TargetClasses::BASE,
                "Synthetic BASE did not pass triangulation");
        require(cv::norm(measured[0].RealLocation - expectedRight) < 1.0,
                "Triangulated coordinates changed");
        require(measured[0].leftReprojectionErrorPx < 0.01 && measured[0].rightReprojectionErrorPx < 0.01,
                "Measurement reprojection errors missing or inconsistent");
        const cv::Mat dartRotation = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 0, 1, 0, -1, 0);
        const cv::Mat dartTranslation = (cv::Mat_<double>(3, 1) << 240, -18.83, 231.45);
        const LocationTransform locationtransform(dartRotation, dartTranslation);
        const auto locations = locationtransform.RunTransform(measured);
        const auto visual = visualization::makeLocationVisualize(locations);
        require(visual.targets.size() == 1, "Location adapter lost target");
        require(cv::norm(visual.targets[0].position_m - cv::Point3d(0.740, 23.98117, 1.03145)) < 0.001,
                "Visualization must use transformed dart coordinates in meters");
        require(visual.targets[0].target_class == StereoscopicMeasurementVisualize::TargetClassVisualize::BASE,
                "Location adapter lost target class");
        require(visualization::makeLocationVisualize({}).targets.empty(), "Empty locations created a stale target");
        // 真实运行主循环的 yaw 发送分支，比较当前目标求解值和电控实际收到的命令。
        controlDetections = synthetic;
        const LocationTransform actualTransform(config.locationTransform.R, config.locationTransform.T);
        for (int scenario = 0; scenario < 4; ++scenario) {
            const bool stationary = scenario != 0;
            controlMotionPerFrame = scenario == 2 ? 0.8f : 0.1f;
            // 模拟同样的连续观测，确定门控放行时应发送的滤波yaw。
            controlFrameLimit = config.kalmanFilter.minAge + config.kalmanFilter.stableFrames;
            DartKalmanFilter expectedFilter(config.kalmanFilter);
            std::optional<double> expectedYaw;
            for (int frame = 1; frame < controlFrameLimit; ++frame) {
                auto latest = synthetic;
                for (auto& d : latest.left) d.CenterPoint.x += controlMotionPerFrame*frame;
                for (auto& d : latest.right) d.CenterPoint.x += controlMotionPerFrame*frame;
                const auto expectedLocations = actualTransform.RunTransform(estimator.RunMeasurement(latest));
                require(expectedLocations.size() == 1, "Synthetic KF target missing");
                const auto filtered = expectedFilter.RunKalmanFilter(expectedLocations.front());
                if (filtered.stable) {
                    expectedYaw = YawSolver{}.RunSolve({{filtered.XkHat, filtered.TargetName}}, true);
                    break;
                }
            }
            if (scenario == 1 || scenario == 3)
                require(expectedYaw.has_value(), "Stationary synthetic target did not pass KF gate");
            reset();
            controlRun = true;
            ControlMcu mcu;
            communicationParams = config.communication;
            communication = std::make_unique<dart_mcu::Communication>();
            communication->start(mcu.port, communicationParams.baud);
            requestedAim.reset();
            preAimPending = false;
            aim_base = true;
            // 第4种场景：镖架静止，但本次视觉精调会使累计角度超过17度。
            const double initialAngle = scenario == 3 ? 16.0 * CV_PI / 180.0 : std::atan(4.0 / 482.5);
            if (scenario == 3)
                require(initialAngle + *expectedYaw > 17.0 * CV_PI / 180.0, "Fixture does not cross angle limit");
            current_angle = initialAngle;
            aimRevision = lastCommandStatusSequence = 0;
            lastCommandAt = {};
            mcu.publish(stationary);
            require(runCameraMode(config) == 0, "Control camera run failed");
            // 仅镖架静止且KF稳定时发送，不能在首次初始化后提前下发。
            if (stationary && scenario != 3 && expectedYaw) {
                mcu.expectCommand(*expectedYaw);
                require(std::abs(current_angle - (initialAngle + *expectedYaw)) < 1e-12,
                        "Visual correction was not accumulated in current angle");
                const double angleAfterCorrection = current_angle;
                mcu.publish(true);
                handleAimModeKey('2');
                const double outpostAngle = 10.5 * CV_PI / 180.0;
                mcu.expectCommand(outpostAngle - angleAfterCorrection);
                require(std::abs(current_angle - outpostAngle) < 1e-12,
                        "Pre-aim switch ignored previous visual correction");
            } else {
                require(!mcu.hasCommand(), "Moving launcher, unstable KF or over-limit angle allowed yaw command");
                require(current_angle == initialAngle, "Blocked yaw changed current angle");
            }
            communication.reset();
            controlRun = false;
        }
        config.frameQueue.capacity = 4;
        config.frameQueue.maxTimestampDiffMs = 5;
        const auto oldHandler = std::signal(SIGINT, priorSignalHandler);
        for (bool empty : {false, true}) {
            reset();
            emptyResults = empty;
            // 目标切换只接受 processFrame 的键值，位置显示的返回值不影响选择。
            aim_base = empty;
            detectionKey = empty ? '2' : '1';
            measurementKey = empty ? '1' : '2';
            require(runCameraMode(config) == 0, "Camera mode did not stop normally");
            require(aim_base == !empty, "Target switching did not stay in processFrame");
            require(yoloLoads == 1 && yoloCalls == 1 && detectionCalls == 1 && yoloShows == 1 && guideShows == 1
                    && measurementShows == 1,
                    "Perception/visualization not completed once");
            require(!active && stopCalls >= 1, "Camera not stopped on normal exit");
            const auto restored = std::signal(SIGINT, priorSignalHandler);
            require(restored == priorSignalHandler, "Signal handler not restored");
        }
        for (bool startupFailure : {false, true}) {
            reset();
            failStart = startupFailure;
            failDetection = !startupFailure;
            bool caught = false;
            try { runCameraMode(config); }
            catch (const std::runtime_error&) { caught = true; }
            require(caught && !active && stopCalls >= 1, "Exception path did not release camera");
            require(std::signal(SIGINT, priorSignalHandler) == priorSignalHandler,
                    "Exception path did not restore signals");
        }
        std::signal(SIGINT, oldHandler);
        std::cout << "Camera mode pairing, perception, signal exit and cleanup: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        communication.reset();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
