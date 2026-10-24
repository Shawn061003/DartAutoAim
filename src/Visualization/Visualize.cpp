#include "Visualization/Visualize.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <opencv2/freetype.hpp>
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
const cv::Scalar kText(242, 235, 224);
const cv::Scalar kMuted(155, 140, 120);
const cv::Scalar kBackground(24, 17, 11);
const cv::Scalar kSurface(39, 29, 20);
const cv::Scalar kBorder(74, 57, 39);
const cv::Scalar kAccent(224, 208, 70);

cv::Ptr<cv::freetype::FreeType2> chineseFont()
{
    thread_local auto font = [] {
        std::string path;
        if (const char* configured = std::getenv("DART_VISUALIZATION_FONT")) {
            path = configured;
            if (!std::filesystem::is_regular_file(path))
                throw std::runtime_error("DART_VISUALIZATION_FONT 指定的字体文件不存在");
        } else {
            for (const auto* candidate : {
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/mnt/c/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttc"}) {
                if (std::filesystem::is_regular_file(candidate)) { path = candidate; break; }
            }
        }
        if (path.empty()) throw std::runtime_error("缺少中文字体，请用 DART_VISUALIZATION_FONT 指定中文字体文件");
        auto loaded = cv::freetype::createFreeType2();
        loaded->loadFontData(path, 0);
        return loaded;
    }();
    return font;
}

void label(cv::Mat& image, const std::string& text, cv::Point origin,
           const cv::Scalar& color = kText, int height = 16)
{
    chineseFont()->putText(image, text, origin, height, color, -1, cv::LINE_AA, true);
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
    return cv::format("置信度 %.0f%%", target.confidence * 100.0);
}

std::string targetLabel(const GuideLightDetectVisualize::TargetVisualize& target)
{
    return target.success ? "已定位" : "未定位";
}

template <typename FrameVisualize>
cv::Mat drawPanel(const FrameVisualize& data, bool right)
{
    const auto& frame = data.frame;
    if (frame.image.empty() || frame.image.type() != CV_8UC3)
        throw std::invalid_argument("Visualize expects a non-empty CV_8UC3 BGR image");

    cv::Mat panel(kHeaderHeight + kImageHeight, kPanelWidth, CV_8UC3, kBackground);
    cv::rectangle(panel, {0, 0, kPanelWidth, kHeaderHeight}, kSurface, cv::FILLED);
    cv::rectangle(panel, {0, 0, 3, kHeaderHeight}, kAccent, cv::FILLED);
    label(panel, right ? "右相机" : "左相机", {16, 23}, kText, 19);
    label(panel, "候选目标 " + std::to_string(data.targets.size()), {16, 44}, kMuted, 14);

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
        label(panel, "未发现目标", {kPanelWidth / 2 - 50, kHeaderHeight + kImageHeight / 2}, kMuted, 20);

    // 当前 YOLO 每侧最多两项；放大图由上至下位于左画面左上角/右画面右上角。
    // 全图标注不限数量；超过可容纳数量时只省略额外预览，不隐去全图检测结果。
    const auto count = std::min<std::size_t>(data.targets.size(), 2);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& target = data.targets[i];
        const int x = right ? kPanelWidth - kInsetWidth - 8 : 8;
        const int y = kHeaderHeight + 8 + static_cast<int>(i) * (kInsetHeight + 8);
        cv::Mat inset = panel(cv::Rect(x, y, kInsetWidth, kInsetHeight));
        inset.setTo(kSurface);
        label(inset, "#" + std::to_string(i) + " " + targetLabel(target), {5, 17});
        const auto roi = target.roi & fullImage;
        if (roi.empty()) {
            label(inset, "区域无效", {10, 65}, kOrange);
        } else {
            cv::Mat zoom = inset(cv::Rect(0, kLabelHeight, kInsetWidth,
                                         kInsetHeight - kLabelHeight));
            const auto zoomArea = fitImage(frame.image(roi), zoom);
            drawTarget(zoom, target, roi, zoomArea);
        }
        cv::rectangle(inset, cv::Rect(0, 0, inset.cols, inset.rows), kBorder, 1);
    }
    return panel;
}

template <typename FrameVisualize>
cv::Mat drawStereo(const FrameVisualize& left, const FrameVisualize& right)
{
    cv::Mat canvas;
    cv::hconcat(drawPanel(left, false), drawPanel(right, true), canvas);
    cv::line(canvas, {kPanelWidth, 0}, {kPanelWidth, canvas.rows - 1}, kBorder, 1);
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
    show("候选目标", Draw(left, right), delay_ms);
}

cv::Mat GuideLightDetectVisualize::Draw(const FrameVisualize& left, const FrameVisualize& right)
{
    return drawStereo(left, right);
}

void GuideLightDetectVisualize::Show(const FrameVisualize& left, const FrameVisualize& right,
                                    int delay_ms)
{
    show("引导灯检测", Draw(left, right), delay_ms);
}

cv::Mat DetectionVisualize::Draw(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                                 const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                                 const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                                 const GuideLightDetectVisualize::FrameVisualize& guideRight)
{
    auto yolo = YOLOInferenceVisualize::Draw(yoloLeft, yoloRight);
    auto guide = GuideLightDetectVisualize::Draw(guideLeft, guideRight);
    label(yolo, "目标识别", {kPanelWidth - 112, 32}, kAccent, 20);
    label(guide, "引导灯定位", {kPanelWidth - 132, 32}, kAccent, 20);
    cv::Mat canvas;
    cv::vconcat(yolo, guide, canvas);
    cv::line(canvas, {0, yolo.rows}, {canvas.cols - 1, yolo.rows}, kAccent, 1);
    return canvas;
}

void DetectionVisualize::Show(const YOLOInferenceVisualize::FrameVisualize& yoloLeft,
                              const YOLOInferenceVisualize::FrameVisualize& yoloRight,
                              const GuideLightDetectVisualize::FrameVisualize& guideLeft,
                              const GuideLightDetectVisualize::FrameVisualize& guideRight,
                              int delay_ms)
{
    show("目标检测", Draw(yoloLeft, yoloRight, guideLeft, guideRight), delay_ms);
}

namespace {
constexpr double kGridMeters = 0.5;

bool finitePosition(const cv::Point3d& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)
        && std::isfinite(std::hypot(point.x, point.y, point.z));
}

const char* targetName(StereoscopicMeasurementVisualize::TargetClassVisualize category)
{
    using ClassVisualize = StereoscopicMeasurementVisualize::TargetClassVisualize;
    switch (category) {
    case ClassVisualize::BASE: return "基地";
    case ClassVisualize::OUTPOST: return "前哨站";
    default: return "未知目标";
    }
}

cv::Scalar targetColor(StereoscopicMeasurementVisualize::TargetClassVisualize category)
{
    return category == StereoscopicMeasurementVisualize::TargetClassVisualize::OUTPOST
        ? cv::Scalar(240, 155, 195) : kAccent;
}

cv::Point plotPoint(const cv::Point2d& point, const cv::Rect2d& domain, const cv::Rect& plot)
{
    return {plot.x + cvRound((point.x - domain.x) * plot.width / domain.width),
            plot.y + plot.height - cvRound((point.y - domain.y) * plot.height / domain.height)};
}

void panel(cv::Mat& canvas, const cv::Rect& area)
{
    cv::rectangle(canvas, area, kSurface, cv::FILLED);
    cv::rectangle(canvas, area, kBorder, 1);
    cv::line(canvas, area.tl(), area.tl() + cv::Point(32, 0), kAccent, 2);
    cv::line(canvas, area.tl(), area.tl() + cv::Point(0, 12), kAccent, 2);
}

void drawGrid(cv::Mat& canvas, const cv::Rect& plot, const cv::Rect2d& domain)
{
    cv::rectangle(canvas, plot, kBackground, cv::FILLED);
    // 仍为0.5m等比例方格；过密时隐藏细网格，避免异常范围引起无界循环。
    if (kGridMeters * plot.width / domain.width >= 2.0) {
        for (double x = std::ceil(domain.x / kGridMeters) * kGridMeters;
             x < domain.x + domain.width; x += kGridMeters) {
            const int pixel = plotPoint({x, domain.y}, domain, plot).x;
            const bool major = std::abs(x / 5.0 - std::round(x / 5.0)) < 1e-8;
            cv::line(canvas, {pixel, plot.y}, {pixel, plot.y + plot.height},
                     major ? kBorder : cv::Scalar(38, 29, 20), 1);
            if (major && pixel > plot.x + 12 && pixel < plot.x + plot.width - 20)
                label(canvas, cv::format("%.0f", x), {pixel - 7, plot.y + plot.height + 20}, kMuted, 13);
        }
        for (double y = std::ceil(domain.y / kGridMeters) * kGridMeters;
             y < domain.y + domain.height; y += kGridMeters) {
            const int pixel = plotPoint({domain.x, y}, domain, plot).y;
            const bool major = std::abs(y / 5.0 - std::round(y / 5.0)) < 1e-8;
            cv::line(canvas, {plot.x, pixel}, {plot.x + plot.width, pixel},
                     major ? kBorder : cv::Scalar(38, 29, 20), 1);
            if (major && pixel > plot.y + 14 && pixel < plot.y + plot.height - 10)
                label(canvas, cv::format("%.0f", y), {plot.x + 5, pixel - 4}, kMuted, 13);
        }
    }
    cv::rectangle(canvas, plot, kBorder, 1);
}

void drawOrigin(cv::Mat& canvas, const cv::Point& point)
{
    cv::drawMarker(canvas, point, kText, cv::MARKER_CROSS, 12, 1, cv::LINE_AA);
    cv::circle(canvas, point, 7, kText, 1, cv::LINE_AA);
    label(canvas, "镖口", point + cv::Point(12, -10), kText, 16);
}

void drawTargetMarker(cv::Mat& canvas, cv::Point point,
                      StereoscopicMeasurementVisualize::TargetClassVisualize category)
{
    const auto color = targetColor(category);
    cv::circle(canvas, point, 12, color * 0.18, cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, point, 7, color, 1, cv::LINE_AA);
    cv::circle(canvas, point, 3, color, cv::FILLED, cv::LINE_AA);
    label(canvas, targetName(category), point + cv::Point(14, -12), color, 18);
}
} // namespace

StereoscopicMeasurementVisualize::MetricsVisualize StereoscopicMeasurementVisualize::Metrics(
    const cv::Point3d& position_m)
{
    if (!finitePosition(position_m))
        throw std::invalid_argument("位置可视化需要有限的镖口坐标");
    return {std::hypot(position_m.x, position_m.y), position_m.z};
}

cv::Mat StereoscopicMeasurementVisualize::Draw(const FrameVisualize& data)
{
    cv::Mat canvas(800, 1280, CV_8UC3, kBackground);
    label(canvas, "空间定位", {28, 47}, kText, 30);
    label(canvas, "镖口坐标系", {172, 44}, kMuted, 18);
    const auto validCount = std::count_if(data.targets.begin(), data.targets.end(), [](const auto& target) {
        return finitePosition(target.position_m);
    });
    cv::circle(canvas, {1070, 36}, 4, validCount > 0 ? kAccent : kMuted, cv::FILLED);
    label(canvas, validCount > 0 ? "已定位 " + std::to_string(validCount) + " 个目标" : "等待目标",
          {1086, 43}, validCount > 0 ? kAccent : kMuted, 17);
    cv::line(canvas, {24, 64}, {1256, 64}, kBorder, 1);

    int cardX = 24;
    for (const auto category : {TargetClassVisualize::BASE, TargetClassVisualize::OUTPOST}) {
        const cv::Rect card(cardX, 84, 604, 132);
        panel(canvas, card);
        const auto color = targetColor(category);
        label(canvas, targetName(category), {cardX + 18, 113}, color, 22);
        const auto found = std::find_if(data.targets.begin(), data.targets.end(), [category](const auto& target) {
            return target.target_class == category && finitePosition(target.position_m);
        });
        label(canvas, "水平距离", {cardX + 18, 143}, kMuted, 15);
        label(canvas, "相对高度", {cardX + 234, 143}, kMuted, 15);
        label(canvas, "横向偏移", {cardX + 428, 143}, kMuted, 15);
        if (found != data.targets.end()) {
            const auto metrics = Metrics(found->position_m);
            label(canvas, cv::format("%.2f", metrics.horizontal_distance_m),
                  {cardX + 18, 191}, kText, 36);
            label(canvas, cv::format("%+.2f", metrics.height_m),
                  {cardX + 234, 191}, kText, 32);
            label(canvas, cv::format("%+.2f", found->position_m.x),
                  {cardX + 428, 191}, kText, 32);
        } else {
            for (const int offset : {18, 234, 428})
                label(canvas, "—", {cardX + offset, 188}, kMuted, 30);
            label(canvas, "未发现", {cardX + 520, 111}, kMuted, 16);
        }
        cardX += 628;
    }

    panel(canvas, {24, 240, 604, 520});
    panel(canvas, {652, 240, 604, 520});
    label(canvas, "俯视图", {44, 273}, kText, 21);
    label(canvas, "侧视图", {672, 273}, kText, 21);
    label(canvas, "前向 Y ↑", {487, 271}, kMuted, 16);
    label(canvas, "高度 Z ↑", {1115, 271}, kMuted, 16);
    const cv::Rect topPlot(48, 298, 556, 410);
    const cv::Rect sidePlot(676, 298, 556, 410);

    double xRadius = 2.0, yMin = -1.0, yMax = 27.0, zMin = -1.0, zMax = 3.0;
    for (const auto& target : data.targets) {
        if (!finitePosition(target.position_m)) continue;
        const auto& p = target.position_m;
        xRadius = std::max(xRadius, std::abs(p.x) + 2.0);
        yMin = std::min(yMin, p.y - 2.0);
        yMax = std::max(yMax, p.y + 2.0);
        zMin = std::min(zMin, p.z - 2.0);
        zMax = std::max(zMax, p.z + 2.0);
    }
    const double scale = std::min({topPlot.width / (2.0 * xRadius),
        topPlot.height / (yMax - yMin), sidePlot.width / (yMax - yMin),
        sidePlot.height / (zMax - zMin)});
    if (!(scale > 0.0) || !std::isfinite(scale))
        throw std::invalid_argument("位置可视化坐标范围过大");
    const cv::Rect2d topDomain(-topPlot.width / (2.0 * scale), yMin,
                              topPlot.width / scale, topPlot.height / scale);
    const cv::Rect2d sideDomain(yMin, zMin, sidePlot.width / scale, sidePlot.height / scale);
    drawGrid(canvas, topPlot, topDomain);
    drawGrid(canvas, sidePlot, sideDomain);
    const auto topOrigin = plotPoint({0, 0}, topDomain, topPlot);
    const auto sideOrigin = plotPoint({0, 0}, sideDomain, sidePlot);
    cv::line(canvas, {topOrigin.x, topPlot.y}, {topOrigin.x, topPlot.y + topPlot.height}, kBorder, 1);
    cv::line(canvas, {sidePlot.x, sideOrigin.y}, {sidePlot.x + sidePlot.width, sideOrigin.y}, kBorder, 1);
    for (const auto& target : data.targets) {
        if (!finitePosition(target.position_m)) continue;
        const auto& p = target.position_m;
        const auto top = plotPoint({p.x, p.y}, topDomain, topPlot);
        const auto side = plotPoint({p.y, p.z}, sideDomain, sidePlot);
        const auto foot = plotPoint({p.y, 0}, sideDomain, sidePlot);
        const auto color = targetColor(target.target_class);
        cv::line(canvas, topOrigin, top, color * 0.45, 1, cv::LINE_AA);
        cv::line(canvas, sideOrigin, side, color * 0.45, 1, cv::LINE_AA);
        cv::line(canvas, side, foot, color * 0.65, 1, cv::LINE_AA);
        drawTargetMarker(canvas, top, target.target_class);
        drawTargetMarker(canvas, side, target.target_class);
    }
    drawOrigin(canvas, topOrigin);
    drawOrigin(canvas, sideOrigin);
    label(canvas, "横向 X →", {487, 745}, kMuted, 16);
    label(canvas, "前向 Y →", {1115, 745}, kMuted, 16);
    label(canvas, "单位：米  ·  网格：0.5 米", {28, 786}, kMuted, 15);
    const auto invalidCount = data.targets.size() - static_cast<std::size_t>(validCount);
    if (invalidCount > 0)
        label(canvas, "无效坐标已忽略", {1100, 786}, kOrange, 15);
    return canvas;
}

void StereoscopicMeasurementVisualize::Show(const FrameVisualize& data, int delay_ms)
{
    show("空间定位", Draw(data), delay_ms);
}
