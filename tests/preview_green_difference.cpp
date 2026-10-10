// 离线预览生产差分函数：源码在tests，程序在build，图片输出到调用方指定目录。
#include "Perception/GuideLightDetect.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <stdexcept>

class GuideLightDetectTestAccess {
public:
    static cv::Mat GreenDifference(GuideLightDetect& detector, const cv::Mat& image, CameraSide side)
    {
        return detector.GetGreenDifference(image, side);
    }
};

void Save(const std::filesystem::path& path, const cv::Mat& image)
{
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Cannot write " + path.string());
}

int main(int argc, char** argv)
{
    try {
        if (argc != 3) throw std::runtime_error("Usage: preview_green_difference INPUT_DIR OUTPUT_DIR");
        const std::filesystem::path inputDir(argv[1]), outputDir(argv[2]);
        std::filesystem::create_directories(outputDir);
        GuideLightDetect detector;
        for (const auto& name : {std::string("left1"), std::string("right1")}) {
            const cv::Mat original = cv::imread((inputDir / (name + ".bmp")).string(),
                cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
            if (original.empty()) throw std::runtime_error("Cannot read " + name);
            const auto side = name == "left1" ? CameraSide::Left : CameraSide::Right;
            const cv::Mat difference = GuideLightDetectTestAccess::GreenDifference(detector, original, side);
            // 显示固定为max(D,0)，不取绝对值、不做逐图归一化，避免把负差分显示为亮点。
            cv::Mat positive, display;
            cv::max(difference, 0.0, positive);
            positive.convertTo(display, CV_8U);
            Save(outputDir / (name + "_green_difference.png"), display);
            // TIFF单独保留完整浮点差分，包含负值及小数。
            Save(outputDir / (name + "_green_difference_float.tiff"), difference);
            const int width = 900;
            const int height = cvRound(original.rows * (double(width) / original.cols));
            cv::Mat before, after;
            cv::resize(original, before, {width, height}, 0, 0, cv::INTER_AREA);
            cv::resize(display, after, {width, height}, 0, 0, cv::INTER_AREA);
            cv::cvtColor(after, after, cv::COLOR_GRAY2BGR);
            cv::Mat preview(height + 54, width * 2, CV_8UC3, cv::Scalar(25, 25, 25));
            before.copyTo(preview(cv::Rect(0, 54, width, height)));
            after.copyTo(preview(cv::Rect(width, 54, width, height)));
            cv::putText(preview, name + " | Original", {18, 34}, cv::FONT_HERSHEY_SIMPLEX, .8, {240,240,240}, 1, cv::LINE_AA);
            cv::putText(preview, "Green difference | max(D, 0), no rescaling", {width+18, 34}, cv::FONT_HERSHEY_SIMPLEX, .7, {240,240,240}, 1, cv::LINE_AA);
            Save(outputDir / (name + "_comparison.png"), preview);
            double low, high; cv::Point peak;
            cv::minMaxLoc(difference, &low, &high, nullptr, &peak);
            // 最强绿色差分附近的64x64裁剪，仅用于观察；不作为检测结果。
            const int cropSize = 64;
            const cv::Rect crop(std::max(0, std::min(peak.x - cropSize / 2, original.cols - cropSize)),
                                std::max(0, std::min(peak.y - cropSize / 2, original.rows - cropSize)),
                                cropSize, cropSize);
            cv::Mat cropBefore, cropAfter;
            cv::resize(original(crop), cropBefore, {384, 384}, 0, 0, cv::INTER_NEAREST);
            cv::resize(display(crop), cropAfter, {384, 384}, 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(cropAfter, cropAfter, cv::COLOR_GRAY2BGR);
            cv::Mat closeup(438, 768, CV_8UC3, cv::Scalar(25, 25, 25));
            cropBefore.copyTo(closeup(cv::Rect(0, 54, 384, 384)));
            cropAfter.copyTo(closeup(cv::Rect(384, 54, 384, 384)));
            cv::putText(closeup, name + " | Original x6", {12, 34}, cv::FONT_HERSHEY_SIMPLEX, .65, {240,240,240}, 1, cv::LINE_AA);
            cv::putText(closeup, "Difference x6 | no rescaling", {396, 34}, cv::FONT_HERSHEY_SIMPLEX, .6, {240,240,240}, 1, cv::LINE_AA);
            Save(outputDir / (name + "_closeup.png"), closeup);
            std::cout << name << " " << original.cols << "x" << original.rows
                      << " D=[" << low << "," << high << "] peak=" << peak << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
