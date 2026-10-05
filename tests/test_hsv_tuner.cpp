#include "Perception/GuideLightDetect.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DART_PROJECT_ROOT
#define DART_PROJECT_ROOT "."
#endif

// 滑块参数直接传给生产GetContours，不复制圆度选优算法。
class GuideLightDetectTestAccess {
public:
    static std::vector<cv::Point2f> GetContours(
        const cv::Mat& image, const cv::Rect& roi,
        const cv::Scalar& lower, const cv::Scalar& upper)
    {
        GuideLightDetect detector;
        detector.hsv_lower_ = lower;
        detector.hsv_upper_ = upper;
        return detector.GetContours(image, roi);
    }
};

namespace {
constexpr const char* kWindow = "YOLO - HSV - Contours";
constexpr const char* kOverview = "YOLO ROIs";
const std::array<const char*, 6> kSliderNames{
    "H low", "S low", "V low", "H high", "S high", "V high"};
using Bounds = std::array<int, 6>;
constexpr Bounds kInitialBounds{60, 102, 81, 86, 212, 255};
void SetSliders(const Bounds& values)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        cv::setTrackbarPos(kSliderNames[i], kWindow, values[i]);
    }
}

Bounds ReadSliders(const Bounds& previous)
{
    Bounds values;
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = cv::getTrackbarPos(kSliderNames[i], kWindow);
    }
    // 上下界交叉时，让另一端跟随刚修改的一端，始终保持low <= high。
    for (std::size_t i = 0; i < 3; ++i) {
        if (values[i] > values[i + 3]) {
            if (values[i] != previous[i]) {
                values[i + 3] = values[i];
                cv::setTrackbarPos(kSliderNames[i + 3], kWindow, values[i + 3]);
            } else {
                values[i] = values[i + 3];
                cv::setTrackbarPos(kSliderNames[i], kWindow, values[i]);
            }
        }
    }
    return values;
}

struct RoiPreview {
    cv::Mat mask, filtered, overlay;
    std::vector<cv::Point2f> contour; // 全图坐标
    int candidates = 0;
    double circularity = 0;
};
struct Preview {
    cv::Mat canvas, overview;
    std::vector<RoiPreview> rois;
};

void PutPanel(cv::Mat& canvas, const cv::Mat& source, int column, int row,
              const std::string& label)
{
    constexpr int width = 260, height = 200, rowHeight = 250, header = 55;
    cv::Mat bgr;
    if (source.channels() == 1) cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
    else bgr = source;
    const double scale = std::min(double(width) / source.cols, double(height) / source.rows);
    const cv::Size size(std::max(1, int(std::lround(source.cols * scale))),
                        std::max(1, int(std::lround(source.rows * scale))));
    cv::Mat resized;
    cv::resize(bgr, resized, size, 0, 0,
               source.channels() == 1 || scale >= 1 ? cv::INTER_NEAREST : cv::INTER_AREA);
    const int y = header + row * rowHeight;
    resized.copyTo(canvas(cv::Rect(column * width + (width - size.width) / 2,
                                  y + 25 + (height - size.height) / 2, size.width, size.height)));
    cv::putText(canvas, label, {column * width + 6, y + 18},
                cv::FONT_HERSHEY_SIMPLEX, 0.48, {255, 255, 255}, 1, cv::LINE_AA);
}

Preview MakePreview(const cv::Mat& image, const YOLOInference::YOLOResults& detections,
                    const Bounds& bounds)
{
    Preview result;
    result.overview = image.clone();
    result.canvas = cv::Mat(55 + 250 * int(detections.size()), 1040,
                            CV_8UC3, cv::Scalar(35, 35, 35));
    const cv::Scalar lower(bounds[0], bounds[1], bounds[2]);
    const cv::Scalar upper(bounds[3], bounds[4], bounds[5]);
    cv::putText(result.canvas, cv::format("HSV low=(%d,%d,%d) high=(%d,%d,%d)",
                bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
                {10, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1, cv::LINE_AA);
    cv::putText(result.canvas, "Yellow: candidates | Red: selected | S: save | D: reset | Q / Esc: quit",
                {10, 44}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {180, 220, 180}, 1, cv::LINE_AA);
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        const auto& roi = detection.roi;
        RoiPreview p;
        cv::Mat hsv;
        // 所有计算使用原图像素，只有窗口显示时放大ROI。
        cv::cvtColor(image(roi), hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, lower, upper, p.mask);
        cv::bitwise_and(image(roi), image(roi), p.filtered, p.mask);
        p.overlay = image(roi).clone();
        std::vector<std::vector<cv::Point>> candidates;
        cv::findContours(p.mask.clone(), candidates, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        p.candidates = int(candidates.size());
        cv::drawContours(p.overlay, candidates, -1, {0, 255, 255}, 1);
        p.contour = GuideLightDetectTestAccess::GetContours(image, roi, lower, upper);
        if (!p.contour.empty()) {
            std::vector<cv::Point> local;
            for (const auto& point : p.contour)
                local.emplace_back(cvRound(point.x) - roi.x, cvRound(point.y) - roi.y);
            cv::drawContours(p.overlay, std::vector<std::vector<cv::Point>>{local}, 0, {0, 0, 255}, 1);
            const double perimeter = cv::arcLength(p.contour, true);
            p.circularity = 4 * CV_PI * cv::contourArea(p.contour) / (perimeter * perimeter);
        }
        cv::rectangle(result.overview, roi, {0, 255, 0}, 2);
        cv::putText(result.overview, cv::format("ROI %d conf=%.3f", int(i + 1), detection.conf),
                    {roi.x, std::max(18, roi.y - 8)}, cv::FONT_HERSHEY_SIMPLEX,
                    0.55, {0, 255, 0}, 1, cv::LINE_AA);
        PutPanel(result.canvas, image(roi), 0, int(i), cv::format("ROI %d | conf %.3f", int(i + 1), detection.conf));
        PutPanel(result.canvas, p.mask, 1, int(i), "HSV mask");
        PutPanel(result.canvas, p.filtered, 2, int(i), "Filtered");
        PutPanel(result.canvas, p.overlay, 3, int(i), "External contours");
        cv::putText(result.canvas, cv::format("ROI %d: candidates=%d selected_points=%d circularity=%.4f %s",
                    int(i + 1), p.candidates, int(p.contour.size()), p.circularity,
                    p.contour.empty() ? "(no valid contour)" : ""),
                    {10, 55 + int(i) * 250 + 242}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
                    {220, 220, 220}, 1, cv::LINE_AA);
        result.rois.push_back(std::move(p));
    }
    return result;
}

void SavePreview(const std::filesystem::path& imagePath, const std::filesystem::path& modelPath,
                 const std::string& camera, const cv::Mat& image,
                 const YOLOInference::YOLOResults& detections, const Bounds& bounds, const Preview& preview)
{
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto directory = std::filesystem::path(DART_PROJECT_ROOT) / "outputs" / "hsv_tuner"
                         / std::to_string(stamp);
    std::filesystem::create_directories(directory);
    const auto saveImage = [&](const std::string& name, const cv::Mat& mat) {
        if (!cv::imwrite((directory / name).string(), mat))
            throw std::runtime_error("Cannot save image: " + name);
    };
    saveImage("preview.png", preview.canvas);
    saveImage("yolo_rois.png", preview.overview);
    cv::FileStorage config((directory / "hsv.yaml").string(), cv::FileStorage::WRITE);
    if (!config.isOpened()) throw std::runtime_error("Cannot save HSV settings.");
    config << "image" << imagePath.string() << "model" << modelPath.string() << "camera" << camera;
    config << "hsv_lower" << "[" << bounds[0] << bounds[1] << bounds[2] << "]";
    config << "hsv_upper" << "[" << bounds[3] << bounds[4] << bounds[5] << "]";
    config << "coordinate_system" << "full_image_x_right_y_down";
    config << "detections" << "[";
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& roi = detections[i].roi;
        const auto& p = preview.rois[i];
        const auto prefix = "roi_" + std::to_string(i + 1);
        saveImage(prefix + "_original.png", image(roi));
        saveImage(prefix + "_mask.png", p.mask);
        saveImage(prefix + "_filtered.png", p.filtered);
        saveImage(prefix + "_contours.png", p.overlay);
        // 此状态只说明轮廓提取结果；中心提取尚未实现。
        config << "{" << "roi" << roi << "confidence" << detections[i].conf
               << "contour_status" << (p.contour.empty() ? "FAILED" : "SUCCESS")
               << "candidate_count" << p.candidates << "circularity" << p.circularity
               << "contour_full_image" << p.contour << "}";
    }
    config << "]";
    std::cout << "Saved: " << directory << std::endl;
}
} // namespace

// YOLO只在启动时运行一次；滑块更新仅重算每个YOLO ROI的传统视觉。
// 默认输入是右相机图片。另一侧使用空白占位帧，结果不作为双目测量数据。
// --preview-only用于无窗口运行同一流程并保存结果。
int main(int argc, char** argv)
{
    try {
        auto imagePath = std::filesystem::path(DART_PROJECT_ROOT)
            / "Input/InputImage/Image_20261003164303751.bmp";
        auto modelPath = std::filesystem::path(DART_PROJECT_ROOT) / "weights/GuideLightDetect.onnx";
        std::string camera = "right", device = "CPU";
        bool previewOnly = false, hasImagePath = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: dart_hsv_tuner [image_path] [--camera right|left] "
                             "[--model model.onnx] [--device CPU] [--preview-only]\n"
                             "S: save to outputs/hsv_tuner; D: reset HSV; Q/Esc: quit.\n";
                return 0;
            }
            if (arg == "--preview-only") previewOnly = true;
            else if (arg == "--camera" || arg == "--model" || arg == "--device") {
                if (++i == argc) throw std::invalid_argument("Missing value for " + arg);
                if (arg == "--camera") camera = argv[i];
                else if (arg == "--model") modelPath = argv[i];
                else device = argv[i];
            } else if (!hasImagePath && arg.rfind("--", 0) != 0) {
                imagePath = arg;
                hasImagePath = true;
            } else throw std::invalid_argument("Unexpected argument: " + arg);
        }
        if (camera != "left" && camera != "right")
            throw std::invalid_argument("--camera must be left or right.");
        const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot read image: " + imagePath.string());
        std::cout << "Image: " << imagePath << " (" << image.cols << 'x' << image.rows
                  << "), camera=" << camera << "\nLoading YOLO on " << device << "..." << std::endl;
        YOLOInference yolo(modelPath.string(), device);
        CameraFrame left, right;
        left.image = camera == "left" ? image : cv::Mat::zeros(3648, 5472, CV_8UC3);
        right.image = camera == "right" ? image : cv::Mat::zeros(1080, 1440, CV_8UC3);
        const auto yoloResult = yolo.RunYOLOInfer(left, right);
        const auto& detections = camera == "left" ? yoloResult.left : yoloResult.right;
        std::cout << "YOLO ROIs: " << detections.size()
                  << ". Other camera is a blank placeholder; its results are ignored.\n" << std::flush;
        if (detections.empty()) throw std::runtime_error("YOLO found no ROI; cannot tune ROI contours.");
        Bounds bounds = kInitialBounds;
        auto preview = MakePreview(image, detections, bounds);
        for (std::size_t i = 0; i < detections.size(); ++i) {
            std::cout << "ROI " << i + 1 << ": " << detections[i].roi << " conf=" << detections[i].conf
                      << " candidates=" << preview.rois[i].candidates
                      << " contour_points=" << preview.rois[i].contour.size() << '\n';
        }
        if (previewOnly) {
            SavePreview(imagePath, modelPath, camera, image, detections, bounds, preview);
            return 0;
        }
        cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
        cv::namedWindow(kOverview, cv::WINDOW_NORMAL);
        cv::resizeWindow(kOverview, 720, 540);
        cv::imshow(kOverview, preview.overview);
        for (std::size_t i = 0; i < bounds.size(); ++i)
            cv::createTrackbar(kSliderNames[i], kWindow, nullptr, i % 3 == 0 ? 179 : 255);
        SetSliders(bounds);
        bool dirty = true;
        while (true) {
            const auto current = ReadSliders(bounds);
            if (current != bounds) { bounds = current; dirty = true; }
            if (dirty) {
                preview = MakePreview(image, detections, bounds);
                cv::imshow(kWindow, preview.canvas);
                std::cout << "lower=(" << bounds[0] << ',' << bounds[1] << ',' << bounds[2]
                          << ") upper=(" << bounds[3] << ',' << bounds[4] << ',' << bounds[5]
                          << ")\n" << std::flush;
                dirty = false;
            }
            const int key = cv::waitKey(30);
            const double visible = cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE);
            // 某些HighGUI后端对此属性返回-1，不能将其视为窗口已关闭。
            if (key == 27 || key == 'q' || key == 'Q' || visible == 0.0) break;
            if (key == 's' || key == 'S')
                SavePreview(imagePath, modelPath, camera, image, detections, bounds, preview);
            if (key == 'd' || key == 'D') { SetSliders(kInitialBounds); dirty = true; }
        }
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
