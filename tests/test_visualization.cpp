#include "Visualization/Visualize.h"

#include <iostream>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
bool red(const cv::Mat& image, int x, int y)
{
    const auto pixel = image.at<cv::Vec3b>(y, x);
    return pixel[2] > 180 && pixel[0] < 100 && pixel[1] < 100;
}
}

int main(int argc, char* argv[])
{
    try {
        const cv::Mat leftImage(480, 640, CV_8UC3, cv::Scalar(12, 18, 24));
        const cv::Mat rightImage(960, 1280, CV_8UC3, cv::Scalar(24, 18, 12));
        const auto originalLeft = leftImage.clone();
        const auto originalRight = rightImage.clone();
        GuideLightDetectVisualize::FrameVisualize left{{leftImage, 10, 100}, {}};
        GuideLightDetectVisualize::FrameVisualize right{{rightImage, 11, 101}, {}};
        left.targets.push_back({{360, 280, 80, 80}, true, {400, 320},
                                {{380, 300}, {420, 300}, {420, 340}, {380, 340}}});
        right.targets.push_back({{720, 560, 160, 160}, true, {800, 640},
                                 {{760, 600}, {840, 600}, {840, 680}, {760, 680}}});
        const auto result = GuideLightDetectVisualize::Draw(left, right);
        require(result.size() == cv::Size(1280, 532), "Unexpected stereo canvas size");
        require(red(result, 400, 372), "Left full-frame center mapping is wrong");
        require(red(result, 1040, 372), "Right scaled center mapping is wrong");
        require(red(result, 380, 352), "Full-frame contour was not drawn");
        require(red(result, 98, 144), "Left ROI center offset is wrong");
        require(red(result, 1182, 144), "Right ROI center offset is wrong");
        require(cv::norm(leftImage, originalLeft, cv::NORM_INF) == 0, "Modified left input");
        require(cv::norm(rightImage, originalRight, cv::NORM_INF) == 0, "Modified right input");
        if (argc > 1) require(cv::imwrite(argv[1], result), "Could not save preview");

        // 即使失败项带有有限中心，也不允许画成功十字；NaN 同样安全。
        left.targets[0].success = false;
        auto failed = GuideLightDetectVisualize::Draw(left, right);
        require(!red(failed, 400, 372), "FAILED target drew a center");
        left.targets[0].center = GuideLightDetectVisualize::TargetVisualize{}.center;
        GuideLightDetectVisualize::Draw(left, right);
        left.targets[0].success = true;
        failed = GuideLightDetectVisualize::Draw(left, right);
        require(!red(failed, 400, 372), "NaN center was drawn");

        // 边界 ROI 裁剪、空结果和左右不同长宽比不应导致图像操作越界。
        left.targets[0].roi = {-10, -10, 30, 30};
        right.targets.clear();
        right.frame.image = cv::Mat(400, 100, CV_8UC3, cv::Scalar(0, 0, 0));
        GuideLightDetectVisualize::Draw(left, right);
        left.targets[0].roi = {-50, -50, 10, 10};
        GuideLightDetectVisualize::Draw(left, right);
        left.targets.clear();
        GuideLightDetectVisualize::Draw(left, right);

        YOLOInferenceVisualize::FrameVisualize yoloLeft{{leftImage, 10, 100}, {}};
        YOLOInferenceVisualize::FrameVisualize yoloRight{{rightImage, 11, 101}, {}};
        yoloLeft.targets.push_back({{360, 280, 80, 80}, 0.95});
        yoloLeft.targets.push_back({{-10, -10, 30, 30}, 0.8});
        const auto yolo = YOLOInferenceVisualize::Draw(yoloLeft, yoloRight);
        const auto box = yolo.at<cv::Vec3b>(332, 360);
        require(box[2] > 200 && box[0] < 100 && box[1] < 100, "YOLO ROI box was not drawn");
        require(cv::norm(leftImage, originalLeft, cv::NORM_INF) == 0, "YOLO modified input");
        GuideLightDetectVisualize::FrameVisualize guideLeft{{leftImage, 10, 100}, {}};
        GuideLightDetectVisualize::FrameVisualize guideRight{{rightImage, 11, 101}, {}};
        guideLeft.targets.push_back({{360, 280, 80, 80}, true, {400, 320},
                                     {{380, 300}, {420, 300}, {420, 340}, {380, 340}}});
        const auto combined = DetectionVisualize::Draw(yoloLeft, yoloRight, guideLeft, guideRight);
        require(combined.size() == cv::Size(1280, 1064), "Detection panels were not stacked vertically");
        require(red(combined, 360, 332), "Combined top panel lost YOLO ROI");
        require(red(combined, 400, 532 + 372), "Combined bottom panel lost guide-light center");
        require(cv::norm(leftImage, originalLeft, cv::NORM_INF) == 0, "Combined drawing modified input");

        // 三四五三角形验证L0取双目中点，且L0/L1均不包含高度分量。
        const auto metrics = StereoscopicMeasurementVisualize::Metrics({2, -4, 5}, {-2, 0, 2});
        require(std::abs(metrics.l0_m - 5.0) < 1e-12, "L0 must use the stereo midpoint in XZ");
        require(std::abs(metrics.l1_m - std::sqrt(29.0)) < 1e-12, "L1 must use right-camera XZ distance");
        require(metrics.delta_h_m == 4.0, "Height above right camera must be positive");
        require(StereoscopicMeasurementVisualize::Metrics({2, 4, 5}, {-2, 0, 2}).delta_h_m == -4.0,
                "Height below right camera must be negative");
        StereoscopicMeasurementVisualize::FrameVisualize scene;
        scene.left_camera_m = {-0.488, -0.0028, 0.0393};
        scene.left_frame_id = 10;
        scene.right_frame_id = 11;
        using ClassVisualize = StereoscopicMeasurementVisualize::TargetClassVisualize;
        scene.targets.push_back({{0.4, -1.2, 24.4}, ClassVisualize::BASE, 0.12, 0.18});
        scene.targets.push_back({{-0.2, -0.5, 15.2}, ClassVisualize::OUTPOST, 0.08, 0.11});
        const auto measurement = StereoscopicMeasurementVisualize::Draw(scene);
        require(measurement.size() == cv::Size(1280, 800), "Unexpected measurement panel size");
        cv::Mat greenPixels;
        cv::inRange(measurement, cv::Scalar(45, 160, 40), cv::Scalar(55, 180, 50), greenPixels);
        require(cv::countNonZero(greenPixels) > 0, "Measurement did not draw target markers");
        if (argc > 1) {
            require(cv::imwrite(std::string(argv[1]) + ".detection.png", combined), "Could not save detection preview");
            require(cv::imwrite(std::string(argv[1]) + ".measurement.png", measurement), "Could not save measurement preview");
        }
        scene.targets.clear();
        auto blank = StereoscopicMeasurementVisualize::Draw(scene);
        cv::inRange(blank, cv::Scalar(45, 160, 40), cv::Scalar(55, 180, 50), greenPixels);
        require(cv::countNonZero(greenPixels) == 0, "Empty measurement retained stale target markers");
        scene.targets.push_back({{std::numeric_limits<double>::quiet_NaN(), 0, 10}, ClassVisualize::BASE});
        blank = StereoscopicMeasurementVisualize::Draw(scene);
        cv::inRange(blank, cv::Scalar(45, 160, 40), cv::Scalar(55, 180, 50), greenPixels);
        require(cv::countNonZero(greenPixels) == 0, "Invalid measurement drew a target marker");
        scene.targets = {{{0, 0, 5}, ClassVisualize::OUTPOST}};
        StereoscopicMeasurementVisualize::Draw(scene); // 零高度、缺失误差显示N/A。
        yoloLeft.frame.image.release();
        bool rejected = false;
        try { YOLOInferenceVisualize::Draw(yoloLeft, yoloRight); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Empty image must be rejected");
        std::cout << "Visualization checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
