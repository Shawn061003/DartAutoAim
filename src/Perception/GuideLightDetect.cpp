// 负责使用OpenCV传统视觉检测绿色引导灯

/*
输入：两图片上YOLO检测结果
输出：两图片上OpenCV检测结果
*/

#include "Perception/GuideLightDetect.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <limits>
#include <utility>

GuideLightDetect::GuideLightDetect(const GuideLightParams& params)
    : right_diff_lower_(params.rightDiffLower), right_diff_upper_(params.rightDiffUpper),
      left_diff_lower_(params.leftDiffLower), left_diff_upper_(params.leftDiffUpper)
{
    params.validate();
}

GuideLightDetect::StereoDetectResult GuideLightDetect::RunDetection(
    const CameraFrame& leftFrame, const CameraFrame& rightFrame,
    const YOLOInference::StereoYOLOResult& result)
{
    if (leftFrame.camera_side != CameraSide::Left || rightFrame.camera_side != CameraSide::Right)
        throw std::invalid_argument("GuideLightDetect requires Left/Right camera_side on the corresponding frames");
    const auto detectFrame = [this](const CameraFrame& frame,
                                   const YOLOInference::YOLOResults& targets) {
        DetectResult detections;
        detections.reserve(targets.size());
        if (targets.empty()) return detections;

        // 每侧每帧只计算一次单通道差分全图，供该侧所有ROI提取轮廓。
        const cv::Mat greenImage = GetGreenDifference(frame.image, frame.camera_side);
        for (const auto& target : targets) {
            if (target.camera_side != frame.camera_side)
                throw std::invalid_argument("YOLO result camera_side does not match its CameraFrame");
            GuideLightDetectResult detection;
            detection.roi = target.roi;
            detection.frame_id = frame.frame_id;
            detection.timestamp_ms = frame.timestamp_ms;
            detection.camera_side = frame.camera_side;
            detection.contours = GetContours(greenImage, target.roi, frame.camera_side);
            detection.CenterPoint = GetCenterPoint(detection.contours);
            if (std::isfinite(detection.CenterPoint.x)
                && std::isfinite(detection.CenterPoint.y)) {
                detection.status = DetectStatus::SUCCESS;
            }
            // 失败项也保留，保证输出与输入ROI一一对应；默认状态为FAILED。
            detections.push_back(std::move(detection));
        }
        return detections;
    };

    StereoDetectResult detections;
    detections.left = detectFrame(leftFrame, result.left);
    detections.right = detectFrame(rightFrame, result.right);
    return detections;
}

cv::Mat GuideLightDetect::GetGreenDifference(const cv::Mat& image, CameraSide /*side*/)
{
    if (image.empty() || image.type() != CV_8UC3)
        throw std::invalid_argument("Green difference requires a nonempty CV_8UC3 BGR image");

    // 直接写入独立的浮点单通道图，避免8位运算截断负值或丢失半整数。
    cv::Mat difference(image.size(), CV_32FC1);
    for (int y = 0; y < image.rows; ++y) {
        const auto* source = image.ptr<cv::Vec3b>(y);
        auto* output = difference.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x) {
            const auto& bgr = source[x];
            output[x] = static_cast<float>(bgr[1]) - 0.5f * (bgr[2] + bgr[0]);
        }
    }
    return difference;
}

std::vector<cv::Point2f> GuideLightDetect::GetContours(
    const cv::Mat& image, const cv::Rect& roi, CameraSide side)
{
    // 1. 输入已是浮点单通道差分全图，直接取出ROI。
    const cv::Mat difference = image(roi);

    // 2. 按相机侧别选择差分区间(lower,upper]，生成8位二值掩膜。
    const double lower = side == CameraSide::Left ? left_diff_lower_ : right_diff_lower_;
    const double upper = side == CameraSide::Left ? left_diff_upper_ : right_diff_upper_;
    cv::Mat mask, upperMask;
    cv::compare(difference, lower, mask, cv::CMP_GT);
    cv::compare(difference, upper, upperMask, cv::CMP_LE);
    cv::bitwise_and(mask, upperMask, mask);

    // 3. 只提取外轮廓，保留全部边界像素，供后续中心估计使用。
    std::vector<std::vector<cv::Point>> candidates;
    cv::findContours(mask, candidates, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 4. 圆度为4*pi*面积/周长平方，取圆度最高的候选。
    const std::vector<cv::Point>* bestContour = nullptr;
    double bestCircularity = 0.0;
    for (const auto& contour : candidates) {
        const double area = cv::contourArea(contour);
        const double perimeter = cv::arcLength(contour, true);
        // 单点、线段等退化轮廓不能用于后续圆形目标的中心估计。
        if (area <= 0.0 || perimeter <= 0.0) {
            continue;
        }
        const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity > bestCircularity) {
            bestCircularity = circularity;
            bestContour = &contour;
        }
    }

    // 5. 无有效候选时返回空集合；否则恢复为全图坐标。
    std::vector<cv::Point2f> result;
    if (bestContour == nullptr) {
        return result;
    }
    result.reserve(bestContour->size());
    for (const auto& point : *bestContour) {
        result.emplace_back(static_cast<float>(point.x + roi.x),
                            static_cast<float>(point.y + roi.y));
    }
    return result;
}

cv::Point2f GuideLightDetect::GetCenterPoint(
    const std::vector<cv::Point2f>& contours)
{
    // 用NaN区分失败与合法的图像坐标，避免将(0, 0)误当成检测结果。
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const cv::Point2f invalidCenter(nan, nan);
    if (contours.size() < 5) {
        return invalidCenter;
    }
    for (const auto& point : contours) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return invalidCenter;
        }
    }
    // 输入是有序外轮廓；共线或重复点构成的零面积轮廓无法用于椭圆拟合。
    if (cv::contourArea(contours) <= 0.0) {
        return invalidCenter;
    }

    try {
        const cv::RotatedRect ellipse = cv::fitEllipse(contours);
        if (!std::isfinite(ellipse.center.x) || !std::isfinite(ellipse.center.y)
            || !std::isfinite(ellipse.size.width) || !std::isfinite(ellipse.size.height)
            || ellipse.size.width <= 0.0f || ellipse.size.height <= 0.0f) {
            return invalidCenter;
        }
        // 输入已经是全图坐标，无须再加ROI偏移；整数边界也可拟合出小数中心。
        return ellipse.center;
    } catch (const cv::Exception&) {
        return invalidCenter;
    }
}
