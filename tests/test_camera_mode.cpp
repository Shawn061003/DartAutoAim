// 直接验证主函数中的相机分支；设备、感知和 GUI 边界使用替身，不访问硬件或窗口。
#define main cameraModeProgramMain
#include "../src/main.cpp"
#undef main

#include <csignal>
#include <cstdlib>
#include <fcntl.h>
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
class ControlMcu;
ControlMcu* controlMcu = nullptr;
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

void testBaseReadyToAim()
{
    StereoFrame frame;
    LocationTransform::LocationTransformResult base;
    base.TargetName = TargetClasses::BASE;
    base.stable = true;
    std::optional<std::int64_t> confirmedAt;
    require(!canAimAtBase({base}, frame, confirmedAt) && confirmedAt,
            "First stable result must only record confirmation time");
    frame.left.timestamp_ms = *confirmedAt;
    frame.right.timestamp_ms = *confirmedAt + 1;
    require(!canAimAtBase({base}, frame, confirmedAt), "Stale left frame allowed aiming");
    std::swap(frame.left.timestamp_ms, frame.right.timestamp_ms);
    require(!canAimAtBase({base}, frame, confirmedAt), "Stale right frame allowed aiming");
    frame.left.timestamp_ms = frame.right.timestamp_ms = *confirmedAt + 1;
    require(canAimAtBase({base}, frame, confirmedAt), "Fresh stable stereo pair was rejected");
    base.stable = false;
    require(!canAimAtBase({base}, frame, confirmedAt) && !confirmedAt,
            "Unstable BASE retained confirmation time");
    base.stable = true;
    require(!canAimAtBase({base}, frame, confirmedAt), "BASE restabilization reused old confirmation");
    require(!canAimAtBase({}, frame, confirmedAt) && !confirmedAt,
            "Missing BASE retained confirmation time");
}

void testAimControl()
{
    ControlMcu mcu;
    communicationParams = CommunicationParams{};
    communication = std::make_unique<dart_mcu::Communication>();
    communication->start(mcu.port, communicationParams.baud);
    aim_base = true;
    requestedAim.reset();
    preAimPending = false;
    lastCommandStatusSequence = aimRevision = 0;
    lastCommandAt = {};

    handleAimModeKey('1');
    require(preAimPending && !mcu.hasCommand(), "Unknown status allowed pre-aim");
    mcu.publish(false);
    handleAimModeKey(-1);
    require(preAimPending && !mcu.hasCommand(), "Moving state allowed pre-aim");
    handleAimModeKey('2');
    handleAimModeKey('1');
    handleAimModeKey('2');
    require(!aim_base && !mcu.hasCommand(), "Pending target did not keep latest selection");
    mcu.publish(true);
    handleAimModeKey(-1);
    mcu.expectCommand(0.17);
    require(!preAimPending, "Queued pre-aim was not cleared");
    handleAimModeKey('2');
    require(!preAimPending && !mcu.hasCommand(), "Repeated mode key reissued command");
    handleAimModeKey('1');
    require(preAimPending && !mcu.hasCommand(), "One status authorized two commands");
    mcu.publish(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(communicationParams.statusMaxAgeMs + 20));
    handleAimModeKey(-1);
    require(preAimPending && !mcu.hasCommand(), "Expired status allowed pre-aim");
    mcu.publish(true);
    handleAimModeKey(-1);
    mcu.expectCommand(0.26);

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
    if (!controlRun || detectionCalls == 7) std::raise(SIGINT);
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
        require(data.targets.size() == 1, "Control target was not measured");
        if (measurementShows <= 6)
            require(!controlMcu->hasCommand(), "BASE sent before five observations and a subsequent fresh frame");
        if (measurementShows < 7) {
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
        testBaseReadyToAim();
        testAimControl();
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
        const auto visual = makeLocationVisualize(locations);
        require(visual.targets.size() == 1, "Location adapter lost target");
        require(cv::norm(visual.targets[0].position_m - cv::Point3d(0.740, 23.98117, 1.03145)) < 0.001,
                "Visualization must use transformed dart coordinates in meters");
        require(visual.targets[0].target_class == StereoscopicMeasurementVisualize::TargetClassVisualize::BASE,
                "Location adapter lost target class");
        require(makeLocationVisualize({}).targets.empty(), "Empty locations created a stale target");
        // 真实运行主循环的 yaw 发送分支，比较当前目标求解值和电控实际收到的命令。
        controlDetections = synthetic;
        const LocationTransform actualTransform(config.locationTransform.R, config.locationTransform.T);
        for (int scenario = 0; scenario < 3; ++scenario) {
            const bool stationary = scenario != 0;
            controlMotionPerFrame = scenario == 2 ? 0.8f : 0.1f;
            auto latest = synthetic;
            for (auto& d : latest.left) d.CenterPoint.x += controlMotionPerFrame*6;
            for (auto& d : latest.right) d.CenterPoint.x += controlMotionPerFrame*6;
            auto expectedEstimator = StereoscopicMeasurement(config.leftCamera, config.rightCamera,
                config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
            const auto expectedYaw = YawSolver{}.RunSolve(
                actualTransform.RunTransform(expectedEstimator.RunMeasurement(latest)), true);
            require(expectedYaw.has_value(), "Synthetic control yaw missing");
            reset();
            controlRun = true;
            ControlMcu mcu;
            controlMcu = &mcu;
            communicationParams = config.communication;
            communication = std::make_unique<dart_mcu::Communication>();
            communication->start(mcu.port, communicationParams.baud);
            requestedAim.reset();
            preAimPending = false;
            aim_base = true;
            aimRevision = lastCommandStatusSequence = 0;
            lastCommandAt = {};
            mcu.publish(stationary);
            require(runCameraMode(config) == 0, "Control camera run failed");
            if (scenario == 1) mcu.expectCommand(*expectedYaw);
            else require(!mcu.hasCommand(), "Moving launcher/BASE allowed yaw command");
            controlMcu = nullptr;
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
