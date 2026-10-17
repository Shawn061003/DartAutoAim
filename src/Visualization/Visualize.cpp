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

cv::Mat DetectionVisualize::Draw(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                                 const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                                 const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                                 const GuideLightDetectVisualize::FrameVisualize& guideRight)
{
    auto yolo = YOLOInferenceVisualize::Draw(yoloLeft, yoloRight);
    auto guide = GuideLightDetectVisualize::Draw(guideLeft, guideRight);
    label(yolo, "YOLO", {kPanelWidth - 80, 42});
    label(guide, "GUIDE LIGHT", {kPanelWidth - 120, 42});
    cv::Mat canvas;
    cv::vconcat(yolo, guide, canvas);
    cv::line(canvas, {0, yolo.rows}, {canvas.cols - 1, yolo.rows}, kText, 1);
    return canvas;
}

void DetectionVisualize::Show(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                              const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                              const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                              const GuideLightDetectVisualize::FrameVisualize& guideRight,
                              int delay_ms)
{
    show("detection", Draw(yoloLeft, yoloRight, guideLeft, guideRight), delay_ms);
}

namespace {
const cv::Scalar kMeasureText(55, 50, 40);
const cv::Scalar kTargetGreen(50, 170, 45);
const cv::Scalar kTargetName(175, 85, 30);
constexpr double kGridMeters = 0.5;

bool finitePosition(const cv::Point3d& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool validTarget(const StereoscopicMeasurementVisualize::TargetVisualize& target)
{
    return finitePosition(target.position_m) && target.position_m.z > 0.0;
}

const char* targetName(StereoscopicMeasurementVisualize::TargetClassVisualize target)
{
    using ClassVisualize = StereoscopicMeasurementVisualize::TargetClassVisualize;
    switch (target) {
    case ClassVisualize::BASE: return "BASE";
    case ClassVisualize::OUTPOST: return "OUTPOST";
    default: return "UNKNOWN";
    }
}

void measurementLabel(cv::Mat& image, const std::string& text, cv::Point origin,
                      const cv::Scalar& color = kMeasureText, double scale = 0.45)
{
    // 浅色描边使数值在网格和几何连线上仍可辨认。
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(250, 250, 250), 3, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv::LINE_AA);
}

std::string errorLabel(double error)
{
    return std::isfinite(error) && error >= 0.0 ? cv::format("%.3f px", error) : "N/A";
}

cv::Point plotPoint(const cv::Point2d& point, const cv::Rect2d& domain, const cv::Rect& plot)
{
    return {plot.x + cvRound((point.x - domain.x) * plot.width / domain.width),
            plot.y + plot.height - cvRound((point.y - domain.y) * plot.height / domain.height)};
}

void drawGrid(cv::Mat& canvas, const cv::Rect& plot, const cv::Rect2d& domain)
{
    cv::rectangle(canvas, plot, cv::Scalar(253, 253, 253), cv::FILLED);
    // 网格始终为0.5m，不随缩放改成其他物理间隔；过小的格子停止绘制，避免无界循环。
    if (kGridMeters * plot.width / domain.width >= 2.0) {
        for (double x = std::ceil(domain.x / kGridMeters) * kGridMeters;
             x < domain.x + domain.width; x += kGridMeters) {
            const int pixel = plotPoint({x, domain.y}, domain, plot).x;
            cv::line(canvas, {pixel, plot.y}, {pixel, plot.y + plot.height},
                     cv::Scalar(224, 226, 228), 1);
        }
        for (double y = std::ceil(domain.y / kGridMeters) * kGridMeters;
             y < domain.y + domain.height; y += kGridMeters) {
            const int pixel = plotPoint({domain.x, y}, domain, plot).y;
            cv::line(canvas, {plot.x, pixel}, {plot.x + plot.width, pixel},
                     cv::Scalar(224, 226, 228), 1);
        }
    } else {
        measurementLabel(canvas, "0.5 m grid below display resolution", plot.tl() + cv::Point(8, 18));
    }
    cv::rectangle(canvas, plot, cv::Scalar(160, 165, 170), 1);
}

void cameraLabel(cv::Mat& canvas, cv::Point camera, cv::Point text, const char* name)
{
    cv::line(canvas, camera, text + cv::Point(0, -5), cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    measurementLabel(canvas, name, text, kRed);
}
} // namespace

StereoscopicMeasurementVisualize::MetricsVisualize StereoscopicMeasurementVisualize::Metrics(
    const cv::Point3d& position_m, const cv::Point3d& left_camera_m)
{
    if (!finitePosition(position_m) || position_m.z <= 0.0 || !finitePosition(left_camera_m))
        throw std::invalid_argument("Measurement visualization requires finite positions and positive target depth");
    return {std::hypot(position_m.x - left_camera_m.x / 2.0,
                       position_m.z - left_camera_m.z / 2.0),
            std::hypot(position_m.x, position_m.z), -position_m.y};
}

cv::Mat StereoscopicMeasurementVisualize::Draw(const FrameVisualize& data)
{
    if (!finitePosition(data.left_camera_m))
        throw std::invalid_argument("Measurement visualization requires a finite left camera position");
    cv::Mat canvas(800, 1280, CV_8UC3, cv::Scalar(245, 247, 249));
    const cv::Rect topPlot(28, 238, 584, 482);
    const cv::Rect sidePlot(668, 238, 584, 482);
    const auto& left = data.left_camera_m;
    const cv::Point3d midpoint = left * 0.5;
    double xRadius = std::max(2.0, std::abs(left.x) + 1.0);
    double zMin = std::min(-1.0, left.z - 1.0);
    double zMax = std::max(27.0, left.z + 1.0);
    double hMin = std::min(-1.0, -left.y - 1.0);
    double hMax = std::max(3.0, -left.y + 1.0);
    std::size_t invalid = 0;
    for (const auto& target : data.targets) {
        if (!validTarget(target)) { ++invalid; continue; }
        const auto& p = target.position_m;
        xRadius = std::max(xRadius, std::abs(p.x - midpoint.x) + 1.0);
        zMax = std::max(zMax, p.z + 1.0);
        hMin = std::min(hMin, -p.y - 1.0);
        hMax = std::max(hMax, -p.y + 1.0);
    }
    zMin = std::floor(zMin / kGridMeters) * kGridMeters;
    zMax = std::ceil(zMax / kGridMeters) * kGridMeters;
    // 两视图共用等比例尺，保证每格真实对应0.5m x 0.5m。
    const double scale = std::min({topPlot.width / (2.0 * xRadius),
                                   topPlot.height / (zMax - zMin),
                                   sidePlot.width / (zMax - zMin),
                                   sidePlot.height / (hMax - hMin)});
    if (!(scale > 0.0) || !std::isfinite(scale))
        throw std::invalid_argument("Measurement visualization coordinate range is too large");
    const cv::Rect2d topDomain(midpoint.x - topPlot.width / (2.0 * scale), zMin,
                               topPlot.width / scale, topPlot.height / scale);
    const cv::Rect2d sideDomain(zMin, hMin, sidePlot.width / scale, sidePlot.height / scale);
    drawGrid(canvas, topPlot, topDomain);
    drawGrid(canvas, sidePlot, sideDomain);

    measurementLabel(canvas, "TOP VIEW  X / Z", {28, 32}, kMeasureText, 0.65);
    measurementLabel(canvas, "SIDE VIEW  Z / -Y", {668, 32}, kMeasureText, 0.65);
    measurementLabel(canvas, "Frames L=" + std::to_string(data.left_frame_id)
        + " R=" + std::to_string(data.right_frame_id), {28, 56});
    measurementLabel(canvas, "Right camera coordinates | units: m", {668, 56});
    measurementLabel(canvas, "Reprojection error (undistorted pixels)", {28, 84});
    cv::rectangle(canvas, {24, 94, 592, 98}, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(canvas, {24, 94, 592, 98}, cv::Scalar(200, 200, 200), 1);
    using ClassVisualize = TargetClassVisualize;
    int row = 0;
    for (const auto category : {ClassVisualize::BASE, ClassVisualize::OUTPOST}) {
        const auto it = std::find_if(data.targets.begin(), data.targets.end(), [category](const auto& t) {
            return validTarget(t) && t.target_class == category;
        });
        const int y = 115 + row * 43;
        if (it == data.targets.end()) {
            measurementLabel(canvas, std::string(targetName(category)) + ": not detected", {34, y});
        } else {
            const auto metrics = Metrics(it->position_m, left);
            measurementLabel(canvas, std::string(targetName(category)) + "  L: "
                + errorLabel(it->left_reprojection_error_px) + "  R: "
                + errorLabel(it->right_reprojection_error_px), {34, y});
            measurementLabel(canvas, cv::format("L0=%.3f m   L1=%.3f m   dH=%+.3f m",
                metrics.l0_m, metrics.l1_m, metrics.delta_h_m), {34, y + 19});
        }
        ++row;
    }
    measurementLabel(canvas, "L0: XZ distance to stereo midpoint", {668, 88});
    measurementLabel(canvas, "L1: XZ distance to right camera", {668, 111});
    measurementLabel(canvas, "dH = -Y: above right camera is positive", {668, 134});
    measurementLabel(canvas, "Red: cameras   Green: targets", {668, 157});
    measurementLabel(canvas, cv::format("Baseline: %.3f m   Grid: 0.5 m x 0.5 m", cv::norm(left)), {668, 180});
    measurementLabel(canvas, "Z forward (up) / X right", {28, 220});
    measurementLabel(canvas, "Z forward (right) / height up", {668, 220});

    const auto leftTop = plotPoint({left.x, left.z}, topDomain, topPlot);
    const auto rightTop = plotPoint({0, 0}, topDomain, topPlot);
    const auto midTop = plotPoint({midpoint.x, midpoint.z}, topDomain, topPlot);
    const auto leftSide = plotPoint({left.z, -left.y}, sideDomain, sidePlot);
    const auto rightSide = plotPoint({0, 0}, sideDomain, sidePlot);
    cv::line(canvas, leftTop, rightTop, cv::Scalar(100, 100, 100), 2, cv::LINE_AA);
    const int baselineY = rightSide.y;
    cv::line(canvas, {sidePlot.x, baselineY}, {sidePlot.x + sidePlot.width, baselineY},
             cv::Scalar(140, 140, 140), 1, cv::LINE_AA);

    for (const auto& target : data.targets) {
        if (!validTarget(target)) continue;
        const auto& p = target.position_m;
        const auto metrics = Metrics(p, left);
        const auto top = plotPoint({p.x, p.z}, topDomain, topPlot);
        const auto side = plotPoint({p.z, -p.y}, sideDomain, sidePlot);
        const auto foot = plotPoint({p.z, 0}, sideDomain, sidePlot);
        cv::line(canvas, midTop, top, cv::Scalar(155, 155, 155), 1, cv::LINE_AA);
        cv::line(canvas, rightTop, top, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
        const auto middle = (midTop + top) * 0.5;
        measurementLabel(canvas, cv::format("L0 %.3f m", metrics.l0_m), middle + cv::Point(-132, -4));
        measurementLabel(canvas, cv::format("L1 %.3f m", metrics.l1_m), middle + cv::Point(16, 18));
        cv::line(canvas, side, foot, cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
        // 两端短横线表示高度尺寸，即使高度为0也保留数值标签。
        for (const auto& endpoint : {side, foot})
            cv::line(canvas, endpoint + cv::Point(-4, 0), endpoint + cv::Point(4, 0), kMeasureText, 1);
        const int heightLabelX = std::clamp(side.x + 12, sidePlot.x + 4, sidePlot.x + sidePlot.width - 135);
        measurementLabel(canvas, cv::format("dH %+.3f m", metrics.delta_h_m),
                          {heightLabelX, (side.y + foot.y) / 2 + 18});
        for (const auto& point : {top, side}) {
            cv::circle(canvas, point, 6, kTargetGreen, cv::FILLED, cv::LINE_AA);
            measurementLabel(canvas, targetName(target.target_class), point + cv::Point(10, -10), kTargetName);
        }
    }
    cv::circle(canvas, leftTop, 4, kRed, cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, rightTop, 4, kRed, cv::FILLED, cv::LINE_AA);
    // 侧视图中两个光心可能几乎重合，左侧用空心圆、右侧用实心圆。
    cv::circle(canvas, leftSide, 8, kRed, 2, cv::LINE_AA);
    cv::circle(canvas, rightSide, 4, kRed, cv::FILLED, cv::LINE_AA);
    cameraLabel(canvas, leftTop, {topPlot.x + 118, 750}, "Left camera");
    cameraLabel(canvas, rightTop, {topPlot.x + 350, 750}, "Right camera");
    cameraLabel(canvas, leftSide, {sidePlot.x + 20, 750}, "Left (hollow)");
    cameraLabel(canvas, rightSide, {sidePlot.x + 205, 750}, "Right (filled)");
    measurementLabel(canvas, "Grid: 0.5 m x 0.5 m | equal axis scale", {28, 785});
    if (data.targets.empty() || invalid == data.targets.size())
        measurementLabel(canvas, "No valid measurement", {668, 785}, kTargetName);
    else if (invalid > 0)
        measurementLabel(canvas, "Invalid targets skipped: " + std::to_string(invalid), {668, 785});
    cv::line(canvas, {640, 0}, {640, canvas.rows - 1}, cv::Scalar(170, 175, 180), 1);
    return canvas;
}

void StereoscopicMeasurementVisualize::Show(const FrameVisualize& data, int delay_ms)
{
    show("measurement", Draw(data), delay_ms);
}
