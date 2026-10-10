#include "Perception/GuideLightDetect.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

class GuideLightDetectTestAccess {
public:
    static cv::Mat GreenDifference(GuideLightDetect& detector, const cv::Mat& image, CameraSide side)
    {
        return detector.GetGreenDifference(image, side);
    }
};

namespace {
void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void CheckGreenDifference(GuideLightDetect& detector)
{
    // 非连续输入覆盖逐行步长；固定样本覆盖饱和、白色抵消、负值和半整数。
    cv::Mat storage(4, 5, CV_8UC3, cv::Scalar(9, 9, 9));
    cv::Mat input = storage(cv::Rect(1, 1, 3, 2));
    input.at<cv::Vec3b>(0, 0) = {0, 255, 0};
    input.at<cv::Vec3b>(0, 1) = {255, 0, 255};
    input.at<cv::Vec3b>(0, 2) = {255, 255, 255};
    input.at<cv::Vec3b>(1, 0) = {41, 200, 40};
    input.at<cv::Vec3b>(1, 1) = {0, 0, 1};
    input.at<cv::Vec3b>(1, 2) = {40, 200, 40};
    const cv::Mat original = storage.clone();
    const cv::Mat expected = (cv::Mat_<float>(2, 3) << 255, -255, 0, 159.5f, -0.5f, 160);
    for (CameraSide side : {CameraSide::Left, CameraSide::Right}) {
        cv::Mat difference = GuideLightDetectTestAccess::GreenDifference(detector, input, side);
        Require(difference.type() == CV_32FC1 && difference.size() == input.size(),
                "Green difference changed the image dimensions or output type.");
        Require(cv::norm(difference, expected, cv::NORM_INF) == 0,
                "Green difference lost signs/fractions or used the wrong channel order.");
        difference.setTo(0);
        Require(cv::norm(storage, original, cv::NORM_INF) == 0,
                "Green difference modified or shared storage with its input.");
    }
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
        CheckGreenDifference(detector);
        CameraFrame left{cv::Mat::zeros(100, 120, CV_8UC3), 42, 1000, CameraSide::Left};
        CameraFrame right{cv::Mat::zeros(120, 160, CV_8UC3), 43, 1002, CameraSide::Right};
        const cv::Point2f leftCenter(40.5f, 50.5f), rightCenter(110.5f, 80.5f);
        DrawDisk(left.image, leftCenter, 9, {56, 200, 27}); // 差分D=158.5，验证浮点分割
        DrawDisk(right.image, rightCenter, 11);
        // 2x2噪点面积不足，轮廓筛选后返回FAILED和空轮廓。
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
        Require(result.right[1].contours.empty(), "Small noise survived the contour area gate.");
        Require(cv::norm(result.left[1].CenterPoint - leftCenter) < 0.05,
                "Left center lost its global offset or fractional coordinates.");
        Require(cv::norm(result.right[0].CenterPoint - rightCenter) < 0.05,
                "Right center came from the wrong image or ROI.");
        yolo.left.clear();
        yolo.right.resize(1);
        const auto next = detector.RunDetection(left, right, yolo);
        Require(next.left.empty() && next.right.size() == 1, "Stale results survived the next call.");
        // 无ROI无需图像数据，不应进入差分计算。
        CameraFrame emptyLeft, emptyRight;
        emptyLeft.camera_side = CameraSide::Left;
        emptyRight.camera_side = CameraSide::Right;
        const auto empty = detector.RunDetection(emptyLeft, emptyRight, {});
        Require(empty.left.empty() && empty.right.empty(), "Empty YOLO results produced detections.");
        Require(cv::norm(left.image, originalLeft, cv::NORM_INF) == 0
                && cv::norm(right.image, originalRight, cv::NORM_INF) == 0,
                "Detection modified an input image.");

        // 提高右侧下界，检查参数进入实际轮廓筛选。
        GuideLightParams params;
        params.rightDiffLower = 120.0;
        const auto excluded = GuideLightDetect(params).RunDetection(left, right, yolo);
        Require(excluded.right.size() == 1 &&
                excluded.right[0].status == GuideLightDetect::DetectStatus::FAILED,
                "Configured lower bound was ignored.");
        yolo.left = {{cv::Rect(25, 35, 32, 32), .8, 42, 1000, CameraSide::Left}};
        GuideLightParams selected;
        selected.leftDiffLower = 150.0;
        selected.leftDiffUpper = 160.0;
        selected.rightDiffLower = 110.0;
        selected.rightDiffUpper = 120.0;
        const auto accepted = GuideLightDetect(selected).RunDetection(left, right, yolo);
        Require(accepted.left[0].status == GuideLightDetect::DetectStatus::SUCCESS &&
                accepted.right[0].status == GuideLightDetect::DetectStatus::SUCCESS,
                "Camera-specific difference bounds failed.");
        GuideLightParams swapped = selected;
        swapped.leftDiffLower = selected.rightDiffLower;
        swapped.leftDiffUpper = selected.rightDiffUpper;
        swapped.rightDiffLower = selected.leftDiffLower;
        swapped.rightDiffUpper = selected.leftDiffUpper;
        const auto wrongSide = GuideLightDetect(swapped).RunDetection(left, right, yolo);
        Require(wrongSide.left[0].status == GuideLightDetect::DetectStatus::FAILED &&
                wrongSide.right[0].status == GuideLightDetect::DetectStatus::FAILED,
                "Camera-specific difference bounds were not selected.");
        params.rightDiffLower = 256.0;
        bool rejected = false;
        try { GuideLightDetect bad(params); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Invalid right difference bounds accepted.");
        params = GuideLightParams{};
        params.leftDiffUpper = 256.0;
        rejected = false;
        try { GuideLightDetect bad(params); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Invalid left difference bounds accepted.");
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
        Require(rejected, "Unknown camera side was used to select difference bounds.");
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
