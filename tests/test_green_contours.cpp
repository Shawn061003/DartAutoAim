// 验证差分区间轮廓路径；不依赖相机、模型、窗口或外部配置文件。
#include "Perception/GuideLightDetect.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

class GuideLightDetectTestAccess {
public:
    static std::vector<cv::Point2f> Contours(GuideLightDetect& detector, const cv::Mat& image,
                                           const cv::Rect& roi, CameraSide side)
    { return detector.GetContours(image, roi, side); }
};

void Require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main()
{
    try {
        GuideLightDetect detector;
        const cv::Rect roi(35, 25, 50, 50);
        const cv::Point center(60, 50);
        cv::Mat difference = cv::Mat::zeros(100, 140, CV_32FC1);
        // 非ROI区域放更大的目标，确认只处理指定区域。
        cv::circle(difference, {115, 50}, 20, 200, cv::FILLED);
        cv::circle(difference, center, 12, 0.5, cv::FILLED);
        cv::circle(difference, center, 5, -0.5, cv::FILLED);
        const cv::Mat originalDifference = difference.clone();
        const auto contour = GuideLightDetectTestAccess::Contours(detector, difference, roi, CameraSide::Left);
        Require(contour.size() > 5, "Positive fractional values disappeared before segmentation.");
        Require(cv::boundingRect(contour) == cv::Rect(48, 38, 25, 25),
                "ROI offset, external boundary, or threshold-zero behavior is wrong.");
        Require(cv::norm(cv::fitEllipse(contour).center - cv::Point2f(center)) < 0.05,
                "External ring contour produced the wrong global center.");
        for (float value : {0.0f, -0.5f}) {
            cv::Mat background(100, 140, CV_32FC1, cv::Scalar(value));
            Require(GuideLightDetectTestAccess::Contours(detector, background, roi, CameraSide::Right).empty(),
                    "Zero or negative background produced a contour.");
        }
        // 灰背景D=0、绿圈D=0.5、红色内孔D=-0.5，应与直接输入差分得到相同外轮廓。
        cv::Mat bgr(100, 140, CV_8UC3, cv::Scalar(50, 50, 50));
        cv::circle(bgr, center, 12, {50, 51, 51}, cv::FILLED);
        cv::circle(bgr, center, 5, {0, 0, 1}, cv::FILLED);
        const cv::Mat originalBgr = bgr.clone();
        Require(cv::norm(difference, originalDifference, cv::NORM_INF) == 0,
                "Contour extraction modified the difference image.");

        CameraFrame left{bgr, 1, 100, CameraSide::Left};
        CameraFrame right{bgr, 2, 101, CameraSide::Right};
        YOLOInference::StereoYOLOResult yolo;
        yolo.left = {{roi, .9, 1, 100, CameraSide::Left}};
        yolo.right = {{roi, .9, 2, 101, CameraSide::Right}};
        // D=0.5落在(0,0.5]内，验证小数阈值及上界包含关系。
        GuideLightParams params;
        params.leftDiffUpper = params.rightDiffUpper = 0.5;
        const auto result = GuideLightDetect(params).RunDetection(left, right, yolo);
        params.leftDiffLower = 0.5;
        const auto lowerExcluded = GuideLightDetect(params).RunDetection(left, right, yolo);
        Require(lowerExcluded.left[0].status == GuideLightDetect::DetectStatus::FAILED &&
                lowerExcluded.right[0].status == GuideLightDetect::DetectStatus::SUCCESS,
                "Exclusive lower bound or camera separation failed.");
        params.leftDiffLower = 0.0;
        params.rightDiffUpper = 0.25;
        const auto upperExcluded = GuideLightDetect(params).RunDetection(left, right, yolo);
        Require(upperExcluded.left[0].status == GuideLightDetect::DetectStatus::SUCCESS &&
                upperExcluded.right[0].status == GuideLightDetect::DetectStatus::FAILED,
                "Upper difference bound was ignored.");
        Require(result.left.size() == 1 && result.right.size() == 1 &&
                result.left[0].status == GuideLightDetect::DetectStatus::SUCCESS &&
                result.right[0].status == GuideLightDetect::DetectStatus::SUCCESS,
                "Existing RunDetection callers cannot use the difference segmentation.");
        Require(result.left[0].contours == contour && result.right[0].contours == contour &&
                result.left[0].camera_side == CameraSide::Left &&
                result.right[0].camera_side == CameraSide::Right,
                "Detection changed contour coordinates or camera provenance.");
        Require(cv::norm(bgr, originalBgr, cv::NORM_INF) == 0,
                "RunDetection modified the BGR input image.");
        std::cout << "Green contour checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
