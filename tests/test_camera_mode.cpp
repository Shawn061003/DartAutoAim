// 直接验证主函数中的相机分支；设备、感知和 GUI 边界使用替身，不访问硬件或窗口。
#define main cameraModeProgramMain
#include "../src/main.cpp"
#undef main

#include <csignal>
#include <opencv2/calib3d.hpp>

namespace {
int yoloLoads = 0, yoloCalls = 0, detectionCalls = 0, stopCalls = 0, yoloShows = 0, guideShows = 0;
int measurementShows = 0;
bool active = false, failStart = false, failDetection = false, emptyResults = false;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void reset()
{
    yoloLoads = yoloCalls = detectionCalls = stopCalls = yoloShows = guideShows = 0;
    measurementShows = 0;
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

void DetectionVisualize::Show(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
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
    require(left.targets.size() == (emptyResults ? 0u : 1u) && right.targets.empty(),
            "Guide-light visualization lost empty/failed results");
    if (!emptyResults)
        require(!left.targets.front().success, "Failed detection displayed as successful");
}

void StereoscopicMeasurementVisualize::Show(const FrameVisualize& data, int delayMs)
{
    ++measurementShows;
    require(delayMs > 0, "Continuous measurement visualization must not block");
    require(guideShows == measurementShows, "Measurement shown before detection");
    require(data.left_frame_id == 2 && data.right_frame_id == 9, "Measurement lost frame metadata");
    require(data.targets.empty(), "Failed/empty detections created a false measurement");
    require(data.left_camera_m.x < 0 && cv::norm(data.left_camera_m) < 1.0,
            "Left camera has wrong coordinate frame or unit");
}

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "Expected config path");
        DartCongfig config(argv[1]);
        // 用已知3D点生成左右畸变像素，检查误差由测距器传出，并正确转换到显示接口。
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
        const StereoscopicMeasurement estimator(config.leftCamera, config.rightCamera,
            config.stereoscopicmeasurement.R, config.stereoscopicmeasurement.T);
        const auto measured = estimator.RunMeasurement(synthetic);
        require(measured.size() == 1 && measured[0].TargetName == TargetClasses::BASE,
                "Synthetic BASE did not pass triangulation");
        require(cv::norm(measured[0].RealLocation - expectedRight) < 1.0,
                "Triangulated coordinates changed");
        require(measured[0].leftReprojectionErrorPx < 0.01 && measured[0].rightReprojectionErrorPx < 0.01,
                "Measurement reprojection errors missing or inconsistent");
        StereoFrame sampleFrame;
        sampleFrame.left.frame_id = 17;
        sampleFrame.right.frame_id = 18;
        const auto visual = makeMeasurementVisualize(sampleFrame, config.stereoscopicmeasurement.T, measured);
        require(cv::norm(visual.targets[0].position_m - expectedRight * 0.001) < 0.001,
                "Measurement adapter failed millimeter-to-meter conversion");
        require(cv::norm(visual.left_camera_m - cv::Point3d(translation) * 0.001) < 1e-12,
                "Measurement adapter used the wrong camera transform");
        require(visual.targets[0].target_class == StereoscopicMeasurementVisualize::TargetClassVisualize::BASE
                && visual.targets[0].left_reprojection_error_px == measured[0].leftReprojectionErrorPx
                && visual.targets[0].right_reprojection_error_px == measured[0].rightReprojectionErrorPx,
                "Measurement adapter lost target class or reprojection errors");
        config.frameQueue.capacity = 4;
        config.frameQueue.maxTimestampDiffMs = 5;
        const auto oldHandler = std::signal(SIGINT, priorSignalHandler);
        for (bool empty : {false, true}) {
            reset();
            emptyResults = empty;
            require(runCameraMode(config) == 0, "Camera mode did not stop normally");
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
        std::cerr << error.what() << '\n';
        return 1;
    }
}
