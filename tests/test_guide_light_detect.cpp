#include "Perception/GuideLightDetect.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace {
void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void DrawDisk(cv::Mat& image, cv::Point2f center, float radius, cv::Vec3b color = {80, 200, 80})
{
    // 用已知的半像素圆心生成对称栅格轮廓，检查全图偏移及小数中心是否保留。
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const float dx = x - center.x, dy = y - center.y;
            if (dx * dx + dy * dy <= radius * radius)
                image.at<cv::Vec3b>(y, x) = color;
        }
    }
}

void CheckResult(const GuideLightDetect::GuideLightDetectResult& result,
                 const CameraFrame& frame, const cv::Rect& roi, bool success)
{
    Require(result.roi == roi, "ROI order or coordinates changed.");
    Require(result.camera_side == frame.camera_side, "Detection lost its source camera side.");
    Require(result.frame_id == frame.frame_id && result.timestamp_ms == frame.timestamp_ms,
            "Frame metadata was not preserved.");
    Require((result.status == GuideLightDetect::DetectStatus::SUCCESS) == success,
            "Unexpected detection status.");
    if (!success)
        Require(std::isnan(result.CenterPoint.x) && std::isnan(result.CenterPoint.y),
                "Failed detection must not expose a usable center.");
}
} // namespace

int main()
{
    try {
        GuideLightDetect detector;
        CameraFrame left{cv::Mat::zeros(100, 120, CV_8UC3), 42, 1000, CameraSide::Left};
        CameraFrame right{cv::Mat::zeros(120, 160, CV_8UC3), 43, 1002, CameraSide::Right};
        const cv::Point2f leftCenter(40.5f, 50.5f), rightCenter(110.5f, 80.5f);
        DrawDisk(left.image, leftCenter, 9, {56, 200, 27}); // HSV约(65,221,200)，仅左侧阈值接收
        DrawDisk(right.image, rightCenter, 11);
        // 2x2方块的外轮廓只有4点：轮廓可提取，但不能拟合椭圆。
        right.image(cv::Rect(11, 11, 2, 2)).setTo(cv::Scalar(80, 200, 80));
        const auto originalLeft = left.image.clone();
        const auto originalRight = right.image.clone();
        YOLOInference::StereoYOLOResult yolo;
        yolo.left = {{cv::Rect(0, 0, 20, 20), .9, 42, 1000, CameraSide::Left},
                     {cv::Rect(25, 35, 32, 32), .8, 42, 1000, CameraSide::Left}};
        yolo.right = {{cv::Rect(90, 60, 40, 40), .95, 43, 1002, CameraSide::Right},
                      {cv::Rect(5, 5, 20, 20), .7, 43, 1002, CameraSide::Right}};
        const auto result = detector.RunDetection(left, right, yolo);
        Require(result.left.size() == 2 && result.right.size() == 2,
                "Failed entries were dropped.");
        CheckResult(result.left[0], left, yolo.left[0].roi, false);
        CheckResult(result.left[1], left, yolo.left[1].roi, true);
        CheckResult(result.right[0], right, yolo.right[0].roi, true);
        CheckResult(result.right[1], right, yolo.right[1].roi, false);
        Require(result.left[0].contours.empty(), "Blank ROI produced a contour.");
        Require(result.right[1].contours.size() == 4, "Failed fit lost its diagnostic contour.");
        Require(cv::norm(result.left[1].CenterPoint - leftCenter) < 0.05,
                "Left center lost its global offset or fractional coordinates.");
        Require(cv::norm(result.right[0].CenterPoint - rightCenter) < 0.05,
                "Right center came from the wrong image or ROI.");
        yolo.left.clear();
        yolo.right.resize(1);
        const auto next = detector.RunDetection(left, right, yolo);
        Require(next.left.empty() && next.right.size() == 1, "Stale results survived the next call.");
        const auto empty = detector.RunDetection(left, right, {});
        Require(empty.left.empty() && empty.right.empty(), "Empty YOLO results produced detections.");
        Require(cv::norm(left.image, originalLeft, cv::NORM_INF) == 0
                && cv::norm(right.image, originalRight, cv::NORM_INF) == 0,
                "Detection modified an input image.");

        // 自定义 HSV 排除绿色，确认构造参数进入实际轮廓筛选。
        GuideLightParams params;
        params.rightHsvLower = cv::Scalar(0, 100, 100);
        params.rightHsvUpper = cv::Scalar(10, 255, 255);
        GuideLightDetect configured(params);
        const auto excluded = configured.RunDetection(left, right, yolo);
        Require(excluded.right.size() == 1 &&
                excluded.right[0].status == GuideLightDetect::DetectStatus::FAILED,
                "Configured HSV bounds were ignored.");
        // 同一检测器内两侧必须独立选阈值，交换参数后两侧都无法检测到各自颜色。
        yolo.left = {{cv::Rect(25, 35, 32, 32), .8, 42, 1000, CameraSide::Left}};
        const GuideLightParams defaults;
        GuideLightParams swapped = defaults;
        swapped.leftHsvLower = defaults.rightHsvLower;
        swapped.leftHsvUpper = defaults.rightHsvUpper;
        swapped.rightHsvLower = defaults.leftHsvLower;
        swapped.rightHsvUpper = defaults.leftHsvUpper;
        const auto wrongSide = GuideLightDetect(swapped).RunDetection(left, right, yolo);
        Require(wrongSide.left[0].status == GuideLightDetect::DetectStatus::FAILED
                && wrongSide.right[0].status == GuideLightDetect::DetectStatus::FAILED,
                "Camera-specific HSV bounds were not selected.");
        params.rightHsvLower[0] = 180;
        bool rejected = false;
        try { GuideLightDetect bad(params); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Invalid right HSV bounds accepted.");
        params = defaults;
        params.leftHsvUpper[1] = 256;
        rejected = false;
        try { GuideLightDetect bad(params); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Invalid left HSV bounds accepted.");
        auto mismatched = yolo;
        mismatched.right[0].camera_side = CameraSide::Left;
        rejected = false;
        try { detector.RunDetection(left, right, mismatched); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "YOLO/CameraFrame side mismatch was accepted.");
        auto unknown = left;
        unknown.camera_side = CameraSide::Unknown;
        rejected = false;
        try { detector.RunDetection(unknown, right, yolo); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Unknown camera side was used to select HSV.");
        rejected = false;
        try { detector.RunDetection(right, left, yolo); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Swapped input frames were accepted.");

        std::cout << "Guide light detection checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
