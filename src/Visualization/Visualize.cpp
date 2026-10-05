#include "Visualization/Visualize.h"

#include <algorithm>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

namespace {
constexpr int kPanelWidth = 640;
constexpr int kImageHeight = 480;
constexpr int kHeaderHeight = 52;
constexpr int kInsetWidth = 180;
constexpr int kInsetHeight = 144;
constexpr int kLabelHeight = 24;
const cv::Scalar kRed(0, 0, 255); // OpenCV 使用 BGR 顺序。
const cv::Scalar kOrange(0, 170, 255);
const cv::Scalar kText(235, 235, 235);

void label(cv::Mat& image, const std::string& text, cv::Point origin,
           const cv::Scalar& color = kText)
{
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv::LINE_AA);
}

// 等比例缩放并居中；返回实际图像区域，所有标注使用该区域对应的坐标变换。
cv::Rect fitImage(const cv::Mat& source, cv::Mat& destination)
{
    const double scale = std::min(static_cast<double>(destination.cols) / source.cols,
                                  static_cast<double>(destination.rows) / source.rows);
    const cv::Size size(std::max(1, cvRound(source.cols * scale)),
                        std::max(1, cvRound(source.rows * scale)));
    const cv::Rect area((destination.cols - size.width) / 2,
                        (destination.rows - size.height) / 2, size.width, size.height);
    cv::resize(source, destination(area), size, 0, 0,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return area;
}

cv::Point mapPoint(const cv::Point2f& point, const cv::Rect& source,
                   const cv::Rect& destination)
{
    return {cvRound((point.x - source.x) * destination.width / source.width) + destination.x,
            cvRound((point.y - source.y) * destination.height / source.height) + destination.y};
}

bool inside(const cv::Point2f& point, const cv::Rect& region)
{
    return std::isfinite(point.x) && std::isfinite(point.y)
        && point.x >= region.x && point.y >= region.y
        && point.x < region.x + region.width && point.y < region.y + region.height;
}

void drawTarget(cv::Mat& canvas, const YOLOInferenceVisualize::TargetVisualize& target,
                const cv::Rect& source, const cv::Rect& destination)
{
    const auto roi = target.roi & source;
    if (roi.empty()) return;
    cv::rectangle(canvas, mapPoint(cv::Point2f(roi.tl()), source, destination),
                  mapPoint(cv::Point2f(roi.br() - cv::Point(1, 1)), source, destination),
                  kRed, 2, cv::LINE_AA);
}

void drawTarget(cv::Mat& canvas, const GuideLightDetectVisualize::TargetVisualize& target,
                const cv::Rect& source, const cv::Rect& destination)
{
    const auto color = target.success ? kRed : kOrange;
    // 仅绘制有限且在当前图像范围内的点，防止失败结果中的 NaN 进入整数坐标。
    for (std::size_t i = 0; i < target.contours.size(); ++i) {
        const auto& a = target.contours[i];
        const auto& b = target.contours[(i + 1) % target.contours.size()];
        if (inside(a, source) && inside(b, source))
            cv::line(canvas, mapPoint(a, source, destination),
                     mapPoint(b, source, destination), color, 2, cv::LINE_AA);
    }
    if (target.success && inside(target.center, source))
        cv::drawMarker(canvas, mapPoint(target.center, source, destination),
                       kRed, cv::MARKER_CROSS, 16, 2, cv::LINE_AA);
}

std::string targetLabel(const YOLOInferenceVisualize::TargetVisualize& target)
{
    return cv::format("conf=%.3f", target.confidence);
}

std::string targetLabel(const GuideLightDetectVisualize::TargetVisualize& target)
{
    return target.success ? "SUCCESS" : "FAILED";
}

template <typename FrameVisualize>
cv::Mat drawPanel(const FrameVisualize& data, bool right)
{
    const auto& frame = data.frame;
    if (frame.image.empty() || frame.image.type() != CV_8UC3)
        throw std::invalid_argument("Visualize expects a non-empty CV_8UC3 BGR image");

    cv::Mat panel(kHeaderHeight + kImageHeight, kPanelWidth, CV_8UC3, cv::Scalar(24, 24, 24));
    label(panel, std::string(right ? "Right" : "Left") + "  frame="
          + std::to_string(frame.frame_id) + "  time=" + std::to_string(frame.timestamp_ms)
          + " ms", {10, 20});
    label(panel, "ROI count: " + std::to_string(data.targets.size()), {10, 42});

    cv::Mat imageArea = panel(cv::Rect(0, kHeaderHeight, kPanelWidth, kImageHeight));
    auto destination = fitImage(frame.image, imageArea);
    destination.y += kHeaderHeight;
    const cv::Rect fullImage(0, 0, frame.image.cols, frame.image.rows);
    for (std::size_t i = 0; i < data.targets.size(); ++i) {
        const auto& target = data.targets[i];
        drawTarget(panel, target, fullImage, destination);
        const auto roi = target.roi & fullImage;
        if (!roi.empty()) {
            auto origin = mapPoint(cv::Point2f(roi.tl()), fullImage, destination);
            origin.y = std::clamp(origin.y - 5, kHeaderHeight + 15, panel.rows - 5);
            origin.x = std::clamp(origin.x, 0, kPanelWidth - 160);
            label(panel, "#" + std::to_string(i) + " " + targetLabel(target), origin);
        }
    }
    if (data.targets.empty())
        label(panel, "No ROI", {kPanelWidth / 2 - 25, kHeaderHeight + kImageHeight / 2});

    // 当前 YOLO 每侧最多两项；放大图由上至下位于左画面左上角/右画面右上角。
    // 全图标注不限数量；超过可容纳数量时只省略额外预览，不隐去全图检测结果。
    const auto count = std::min<std::size_t>(data.targets.size(), 2);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& target = data.targets[i];
        const int x = right ? kPanelWidth - kInsetWidth - 8 : 8;
        const int y = kHeaderHeight + 8 + static_cast<int>(i) * (kInsetHeight + 8);
        cv::Mat inset = panel(cv::Rect(x, y, kInsetWidth, kInsetHeight));
        inset.setTo(cv::Scalar(32, 32, 32));
        label(inset, "#" + std::to_string(i) + " " + targetLabel(target), {5, 17});
        const auto roi = target.roi & fullImage;
        if (roi.empty()) {
            label(inset, "Invalid ROI", {10, 65}, kOrange);
        } else {
            cv::Mat zoom = inset(cv::Rect(0, kLabelHeight, kInsetWidth,
                                         kInsetHeight - kLabelHeight));
            const auto zoomArea = fitImage(frame.image(roi), zoom);
            drawTarget(zoom, target, roi, zoomArea);
        }
        cv::rectangle(inset, cv::Rect(0, 0, inset.cols, inset.rows), kText, 1);
    }
    return panel;
}

template <typename FrameVisualize>
cv::Mat drawStereo(const FrameVisualize& left, const FrameVisualize& right)
{
    cv::Mat canvas;
    cv::hconcat(drawPanel(left, false), drawPanel(right, true), canvas);
    cv::line(canvas, {kPanelWidth, 0}, {kPanelWidth, canvas.rows - 1}, kText, 1);
    return canvas;
}

void show(const char* title, const cv::Mat& canvas, int delay_ms)
{
    if (delay_ms < 0) throw std::invalid_argument("Visualize delay_ms must be >= 0");
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, canvas);
    if (delay_ms > 0) {
        cv::waitKey(delay_ms);
        return;
    }
    // 图片模式由 HighGUI 处理事件并等待按键，保证窗口持续显示。
    // GTK3 的 WND_PROP_VISIBLE 返回 -1，不能用该属性作为进入等待的条件。
    // GTK3 也会在所有 HighGUI 窗口关闭后结束等待；此时返回值为 -1。
    if (cv::waitKey(0) >= 0)
        cv::destroyWindow(title);
}
} // namespace

cv::Mat YOLOInferenceVisualize::Draw(const FrameVisualize& left, const FrameVisualize& right)
{
    return drawStereo(left, right);
}

void YOLOInferenceVisualize::Show(const FrameVisualize& left, const FrameVisualize& right,
                                 int delay_ms)
{
    show("YOLO inference", Draw(left, right), delay_ms);
}

cv::Mat GuideLightDetectVisualize::Draw(const FrameVisualize& left, const FrameVisualize& right)
{
    return drawStereo(left, right);
}

void GuideLightDetectVisualize::Show(const FrameVisualize& left, const FrameVisualize& right,
                                    int delay_ms)
{
    show("Guide light detection", Draw(left, right), delay_ms);
}
