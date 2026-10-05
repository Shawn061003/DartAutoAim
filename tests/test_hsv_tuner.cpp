#include "Acquisition/VideoInput.h"
#include "Perception/GuideLightDetect.h"
#include "Visualization/Visualize.h"
#include "common/DartConfig.h"

#include <array>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <string>

#ifndef DART_PROJECT_ROOT
#define DART_PROJECT_ROOT "."
#endif

namespace {
constexpr const char* kLeftWindow = "Left HSV - Q/Esc: quit, D: reset";
constexpr const char* kRightWindow = "Right HSV - Q/Esc: quit, D: reset";
constexpr std::array<const char*, 6> kSliderNames{
    "Lower H", "Lower S", "Lower V", "Upper H", "Upper S", "Upper V"};
using Bounds = std::array<int, 6>;

Bounds toBounds(const GuideLightParams& params, CameraSide side)
{
    const auto& lower = side == CameraSide::Left ? params.leftHsvLower : params.rightHsvLower;
    const auto& upper = side == CameraSide::Left ? params.leftHsvUpper : params.rightHsvUpper;
    return {int(lower[0]), int(lower[1]), int(lower[2]), int(upper[0]), int(upper[1]), int(upper[2])};
}

GuideLightParams toParams(const Bounds& left, const Bounds& right)
{
    GuideLightParams params;
    params.leftHsvLower = cv::Scalar(left[0], left[1], left[2]);
    params.leftHsvUpper = cv::Scalar(left[3], left[4], left[5]);
    params.rightHsvLower = cv::Scalar(right[0], right[1], right[2]);
    params.rightHsvUpper = cv::Scalar(right[3], right[4], right[5]);
    params.validate();
    return params;
}

void setSliders(const char* window, const Bounds& values)
{
    for (std::size_t i = 0; i < values.size(); ++i)
        cv::setTrackbarPos(kSliderNames[i], window, values[i]);
}

void createSliders(const char* window, const Bounds& values)
{
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    for (std::size_t i = 0; i < values.size(); ++i)
        cv::createTrackbar(kSliderNames[i], window, nullptr, i % 3 == 0 ? 179 : 255);
    setSliders(window, values);
}

Bounds readSliders(const char* window, const Bounds& previous)
{
    Bounds values;
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = cv::getTrackbarPos(kSliderNames[i], window);
    // 仅修正当前相机的另一端，避免上下界交叉；左右滑块没有联动。
    for (std::size_t i = 0; i < 3; ++i) {
        if (values[i] > values[i + 3]) {
            if (values[i] != previous[i]) {
                values[i + 3] = values[i];
                cv::setTrackbarPos(kSliderNames[i + 3], window, values[i + 3]);
            } else {
                values[i] = values[i + 3];
                cv::setTrackbarPos(kSliderNames[i], window, values[i]);
            }
        }
    }
    return values;
}

GuideLightDetectVisualize::FrameVisualize makeVisualize(
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

GuideLightDetect::StereoDetectResult detectPair(
    const StereoFrame& frame, const YOLOInference::StereoYOLOResult& rois,
    const Bounds& leftBounds, const Bounds& rightBounds)
{
    // 同一个检测器持有左右参数，由RunDetection/GetContours按侧别选择。
    GuideLightDetect detector(toParams(leftBounds, rightBounds));
    return detector.RunDetection(frame.left, frame.right, rois);
}

cv::Mat drawPair(const StereoFrame& frame, const GuideLightDetect::StereoDetectResult& results)
{
    return GuideLightDetectVisualize::Draw(makeVisualize(frame.left, results.left),
                                          makeVisualize(frame.right, results.right));
}

void showPair(const cv::Mat& canvas)
{
    // 复用可视化模块的左右画面、ROI放大图、轮廓和中心，各侧窗口挂自己的滑块。
    const int half = canvas.cols / 2;
    cv::imshow(kLeftWindow, canvas(cv::Rect(0, 0, half, canvas.rows)));
    cv::imshow(kRightWindow, canvas(cv::Rect(half, 0, canvas.cols - half, canvas.rows)));
}

struct Options {
    std::string configPath = (std::filesystem::path(DART_PROJECT_ROOT) / "config/dart.yaml").string();
    bool checkOnly = false;
};

Options parseOptions(int argc, char** argv)
{
    Options result;
    bool hasConfig = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--config" && !hasConfig) {
            if (++i == argc || std::string(argv[i]).empty()
                || std::string(argv[i]).rfind("--", 0) == 0)
                throw std::invalid_argument("--config requires a YAML path.");
            result.configPath = argv[i];
            hasConfig = true;
        } else if (arg == "--check" && !result.checkOnly) {
            result.checkOnly = true;
        } else {
            throw std::invalid_argument("Usage: dart_hsv_tuner [--config path.yaml] [--check]");
        }
    }
    return result;
}

DartCongfig loadImageConfig(const std::string& path)
{
    DartCongfig config(path);
    // 在读取左右图片、构造YOLO和创建任何窗口之前拒绝非Image模式。
    if (config.input.mode != InputMode::Image)
        throw std::invalid_argument("dart_hsv_tuner requires input.mode=image; camera/video are not supported.");
    return config;
}

int runTuner(const Options& options)
{
    const auto config = loadImageConfig(options.configPath);
    VideoInput input(config.input, config.frameQueue.capacity);
    if (!input.captureNext()) throw std::runtime_error("No image pair available.");
    const auto frame = input.frameQueue().GetFrame(config.frameQueue.maxTimestampDiffMs);
    if (!frame) throw std::runtime_error("Captured image pair could not be matched.");

    YOLOInference yolo(config.yolo);
    const auto rois = yolo.RunYOLOInfer(frame->left, frame->right);
    // 初始化和重置均使用YAML中各侧自己的HSV参数。
    const Bounds initialLeft = toBounds(config.guideLight, CameraSide::Left);
    const Bounds initialRight = toBounds(config.guideLight, CameraSide::Right);
    Bounds left = initialLeft, right = initialRight;
    auto canvas = drawPair(*frame, detectPair(*frame, rois, left, right));
    // 无窗口检查走同一读图/推理/检测/绘制链路，不写文件、不打印结果。
    if (options.checkOnly) return 0;

    createSliders(kLeftWindow, left);
    createSliders(kRightWindow, right);
    showPair(canvas);
    while (true) {
        const int key = cv::waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') break;
        // GTK3可能不支持VISIBLE并返回-1；用AUTOSIZE辅助识别已关闭的窗口。
        const auto closed = [](const char* name) {
            return cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) == 0.0
                || cv::getWindowProperty(name, cv::WND_PROP_AUTOSIZE) < 0.0;
        };
        if (closed(kLeftWindow) || closed(kRightWindow)) break;
        if (key == 'd' || key == 'D') {
            setSliders(kLeftWindow, initialLeft);
            setSliders(kRightWindow, initialRight);
        }
        const auto nextLeft = readSliders(kLeftWindow, left);
        const auto nextRight = readSliders(kRightWindow, right);
        if (nextLeft != left || nextRight != right) {
            left = nextLeft;
            right = nextRight;
            canvas = drawPair(*frame, detectPair(*frame, rois, left, right));
            showPair(canvas);
        }
    }
    cv::destroyAllWindows();
    return 0;
}
} // namespace

#ifndef DART_HSV_TUNER_TEST
int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: dart_hsv_tuner [--config path.yaml] [--check]\n"
                     "Requires input.mode=image. Each camera has independent Lower/Upper H/S/V sliders.\n"
                     "D: reset both cameras from YAML; Q/Esc: quit. No files are written.\n"
                     "--check: run the image pipeline without windows or result output.\n";
        return 0;
    }
    try {
        return runTuner(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
