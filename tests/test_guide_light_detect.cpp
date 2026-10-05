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

void DrawDisk(cv::Mat& image, cv::Point2f center, float radius)
{
    // 用已知的半像素圆心生成对称栅格轮廓，检查全图偏移及小数中心是否保留。
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const float dx = x - center.x, dy = y - center.y;
            if (dx * dx + dy * dy <= radius * radius)
                image.at<cv::Vec3b>(y, x) = {80, 200, 80};
        }
    }
}

void CheckResult(const GuideLightDetect::GuideLightDetectResult& result,
                 const CameraFrame& frame, const cv::Rect& roi, bool success)
{
    Require(result.roi == roi, "ROI order or coordinates changed.");
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
        CameraFrame left{cv::Mat::zeros(100, 120, CV_8UC3), 42, 1000};
        CameraFrame right{cv::Mat::zeros(120, 160, CV_8UC3), 43, 1002};
        const cv::Point2f leftCenter(40.5f, 50.5f), rightCenter(110.5f, 80.5f);
        DrawDisk(left.image, leftCenter, 9);
        DrawDisk(right.image, rightCenter, 11);
        // 2x2方块的外轮廓只有4点：轮廓可提取，但不能拟合椭圆。
        right.image(cv::Rect(11, 11, 2, 2)).setTo(cv::Scalar(80, 200, 80));
        const auto originalLeft = left.image.clone();
        const auto originalRight = right.image.clone();
        YOLOInference::StereoYOLOResult yolo;
        yolo.left = {{cv::Rect(0, 0, 20, 20), .9, 42, 1000},
                     {cv::Rect(25, 35, 32, 32), .8, 42, 1000}};
        yolo.right = {{cv::Rect(90, 60, 40, 40), .95, 43, 1002},
                      {cv::Rect(5, 5, 20, 20), .7, 43, 1002}};
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
        std::cout << "Guide light detection checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
