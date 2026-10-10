// 直接验证主函数中的相机分支；设备、感知和 GUI 边界使用替身，不访问硬件或窗口。
#define main cameraModeProgramMain
#include "../src/main.cpp"
#undef main

#include <csignal>

namespace {
int yoloLoads = 0, yoloCalls = 0, detectionCalls = 0, stopCalls = 0, yoloShows = 0, guideShows = 0;
bool active = false, failStart = false, failDetection = false, emptyResults = false;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void reset()
{
    yoloLoads = yoloCalls = detectionCalls = stopCalls = yoloShows = guideShows = 0;
    active = failStart = failDetection = emptyResults = false;
}
void priorSignalHandler(int) {}
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
    queue_->push(left, right);
    // 第一次 GetFrame 丢弃旧左帧并返回 nullopt；下一次应取得这张新左帧。
    left.frame_id = 2;
    left.timestamp_ms = 100;
    queue_->pushLeft(left);
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
    std::raise(SIGINT);
    StereoDetectResult result;
    if (!emptyResults) {
        GuideLightDetectResult failed;
        failed.status = DetectStatus::FAILED;
        result.left.push_back(failed);
    }
    return result;
}

void YOLOInferenceVisualize::Show(const FrameVisualize& left, const FrameVisualize& right, int delayMs)
{
    ++yoloShows;
    require(delayMs > 0, "Continuous camera visualization must not wait indefinitely");
    require(left.frame.frame_id == 2 && right.frame.frame_id == 9,
            "Visualization received mismatched frame metadata");
    require(left.targets.size() == (emptyResults ? 0u : 1u) &&
            right.targets.size() == (emptyResults ? 0u : 1u), "Visualization lost YOLO results");
}
void GuideLightDetectVisualize::Show(const FrameVisualize& left, const FrameVisualize& right, int delayMs)
{
    ++guideShows;
    require(delayMs > 0, "Continuous guide-light visualization must not wait indefinitely");
    require(detectionCalls == guideShows, "Guide-light visualization ran before detection");
    require(left.frame.frame_id == 2 && right.frame.frame_id == 9,
            "Guide-light visualization received mismatched frames");
    require(left.targets.size() == (emptyResults ? 0u : 1u) && right.targets.empty(),
            "Guide-light visualization lost empty/failed results");
    if (!emptyResults)
        require(!left.targets.front().success, "Failed detection displayed as successful");
}

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "Expected config path");
        DartCongfig config(argv[1]);
        config.frameQueue.capacity = 4;
        config.frameQueue.maxTimestampDiffMs = 5;
        const auto oldHandler = std::signal(SIGINT, priorSignalHandler);
        for (bool empty : {false, true}) {
            reset();
            emptyResults = empty;
            require(runCameraMode(config) == 0, "Camera mode did not stop normally");
            require(yoloLoads == 1 && yoloCalls == 1 && detectionCalls == 1 && yoloShows == 1 && guideShows == 1,
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
        std::cerr << error.what() << '\n';
        return 1;
    }
}
