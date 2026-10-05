#include "Visualization/Visualize.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
bool green(const cv::Mat& image, int x, int y)
{
    const auto pixel = image.at<cv::Vec3b>(y, x);
    return pixel[1] > 180 && pixel[0] < 100 && pixel[2] < 120;
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
        require(green(result, 400, 372), "Left full-frame center mapping is wrong");
        require(green(result, 1040, 372), "Right scaled center mapping is wrong");
        require(green(result, 380, 352), "Full-frame contour was not drawn");
        require(green(result, 98, 144), "Left ROI center offset is wrong");
        require(green(result, 1182, 144), "Right ROI center offset is wrong");
        require(cv::norm(leftImage, originalLeft, cv::NORM_INF) == 0, "Modified left input");
        require(cv::norm(rightImage, originalRight, cv::NORM_INF) == 0, "Modified right input");
        if (argc > 1) require(cv::imwrite(argv[1], result), "Could not save preview");

        // 即使失败项带有有限中心，也不允许画成功十字；NaN 同样安全。
        left.targets[0].success = false;
        auto failed = GuideLightDetectVisualize::Draw(left, right);
        require(!green(failed, 400, 372), "FAILED target drew a center");
        left.targets[0].center = GuideLightDetectVisualize::TargetVisualize{}.center;
        GuideLightDetectVisualize::Draw(left, right);
        left.targets[0].success = true;
        failed = GuideLightDetectVisualize::Draw(left, right);
        require(!green(failed, 400, 372), "NaN center was drawn");

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
        require(box[0] > 200 && box[2] < 100, "YOLO ROI box was not drawn");
        require(cv::norm(leftImage, originalLeft, cv::NORM_INF) == 0, "YOLO modified input");
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
