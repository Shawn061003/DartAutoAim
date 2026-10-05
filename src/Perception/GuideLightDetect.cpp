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
    : right_hsv_lower_(params.rightHsvLower), right_hsv_upper_(params.rightHsvUpper),
      left_hsv_lower_(params.leftHsvLower), left_hsv_upper_(params.leftHsvUpper)
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
        for (const auto& target : targets) {
            if (target.camera_side != frame.camera_side)
                throw std::invalid_argument("YOLO result camera_side does not match its CameraFrame");
            GuideLightDetectResult detection;
            detection.roi = target.roi;
            detection.frame_id = frame.frame_id;
            detection.timestamp_ms = frame.timestamp_ms;
            detection.camera_side = frame.camera_side;
            detection.contours = GetContours(frame.image, target.roi, frame.camera_side);
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

std::vector<cv::Point2f> GuideLightDetect::GetContours(
    const cv::Mat& image, const cv::Rect& roi, CameraSide side)
{
    // 1. 在ROI内进行HSV筛选，得到高亮绿色区域的二值掩膜。
    cv::Mat hsv, mask;
    cv::cvtColor(image(roi), hsv, cv::COLOR_BGR2HSV);
    const auto& lower = side == CameraSide::Left ? left_hsv_lower_ : right_hsv_lower_;
    const auto& upper = side == CameraSide::Left ? left_hsv_upper_ : right_hsv_upper_;
    cv::inRange(hsv, lower, upper, mask);

    // 2. 只提取外轮廓，保留全部边界像素，供后续中心估计使用。
    std::vector<std::vector<cv::Point>> candidates;
    cv::findContours(mask, candidates, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 3. 圆度为4*pi*面积/周长平方，取圆度最高的候选。
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

    // 4. 无有效候选时返回空集合；否则恢复为全图坐标。
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
