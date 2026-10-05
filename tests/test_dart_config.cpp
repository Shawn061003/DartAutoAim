#include "common/DartConfig.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

std::string replace(std::string text, const std::string& from, const std::string& to)
{
    const auto pos = text.find(from);
    require(pos != std::string::npos, "Missing fixture token: " + from);
    text.replace(pos, from.size(), to);
    return text;
}

void checkCalibration(const CameraParams& camera,
                      const std::filesystem::path& source)
{
    // CalibrationResults 可能在其他检出中未随仓库提供；存在时验证全部标定值。
    if (!std::filesystem::exists(source)) return;
    cv::FileStorage file(source.string(), cv::FileStorage::READ);
    cv::Mat k, d;
    file["camera_matrix"] >> k;
    file["distortion_coefficients"] >> d;
    require(camera.width == static_cast<int>(file["image_width"]) &&
            camera.height == static_cast<int>(file["image_height"]), "Calibration dimensions changed");
    require(cv::norm(camera.cameraMatrix, k, cv::NORM_INF) == 0 &&
            cv::norm(camera.distCoeffs, d, cv::NORM_INF) == 0, "Calibration precision changed");
}
} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 3, "Expected config path and fixture directory");
        const auto source = std::filesystem::absolute(argv[1]);
        const auto dir = std::filesystem::absolute(argv[2]);
        std::filesystem::create_directories(dir);
        const auto fixture = dir / "config.yaml";
        std::ifstream input(source);
        require(input.good(), "Cannot read configuration");
        const std::string original((std::istreambuf_iterator<char>(input)), {});
        auto write = [&](const std::string& text) {
            std::ofstream output(fixture);
            output << text;
            require(output.good(), "Cannot write fixture");
        };
        DartCongfig config(source.string());
        require(config.input.mode == InputMode::Camera, "Camera preset mode changed");
        require(config.frameQueue.capacity == 4 && config.frameQueue.maxTimestampDiffMs == 5,
                "Queue defaults changed");
        const auto root = source.parent_path().parent_path();
        checkCalibration(config.leftCamera, root / "CalibrationResults/MV-CS200-10GC_left_full_final/intrinsics.yaml");
        checkCalibration(config.rightCamera, root / "CalibrationResults/MV-CS016-10GC_final/intrinsics.yaml");

        // image 模式按 YAML 目录解析媒体与模型；配置数值原样传给各模块。
        auto image = replace(original, "mode: \"camera\"", "mode: \"image\"");
        image = replace(image, "left_path: \"\"", "left_path: \"left.png\"");
        image = replace(image, "right_path: \"\"", "right_path: \"right.png\"");
        image = replace(image, "capacity: 4", "capacity: 7");
        image = replace(image, "max_timestamp_diff_ms: 5", "max_timestamp_diff_ms: 0");
        image = replace(image, "input_size: 512", "input_size: 256");
        image = replace(image, "tile_overlap_px: 30", "tile_overlap_px: 7");
        image = replace(image, "left_padding_px: 25", "left_padding_px: 2");
        image = replace(image, "right_padding_px: 10", "right_padding_px: 3");
        image = replace(image, "confidence_threshold: 0.25", "confidence_threshold: 0.5");
        image = replace(image, "nms_iou_threshold: 0.45", "nms_iou_threshold: 0.75");
        image = replace(image, "warmup_count: 3", "warmup_count: 1");
        image = replace(image, "[60, 102, 81]", "[50, 100, 80]");
        write(image);
        config.load(fixture.string());
        require(config.input.mode == InputMode::Image &&
                config.input.leftPath == (dir / "left.png").string(), "Image mode/path incorrect");
        require(config.yolo.modelPath == (dir / "../weights/GuideLightDetect.onnx").lexically_normal().string(),
                "Model path does not use YAML directory");
        require(config.frameQueue.capacity == 7 && config.input.rightPath == (dir / "right.png").string(),
                "Input or queue options were ignored");
        require(config.frameQueue.maxTimestampDiffMs == 0 && config.yolo.inputSize == 256 &&
                config.yolo.tileOverlapPx == 7 && config.yolo.leftPaddingPx == 2 &&
                config.yolo.rightPaddingPx == 3 && config.yolo.confidenceThreshold == 0.5F &&
                config.yolo.nmsIouThreshold == 0.75F && config.yolo.warmupCount == 1 &&
                config.guideLight.hsvLower == cv::Scalar(50, 100, 80), "Configuration options were ignored");
        write(replace(image, "mode: \"image\"", "mode: \"video\""));
        require(DartCongfig(fixture.string()).input.mode == InputMode::Video,
                "Video mode incorrect");

        bool missingRejected = false;
        try { DartCongfig missing((dir / "missing.yaml").string()); }
        catch (const std::runtime_error& error) {
            missingRejected = true;
            require(std::string(error.what()).find("missing.yaml") != std::string::npos,
                    "Missing file diagnostic lost its path");
        }
        require(missingRejected, "Missing configuration was accepted");

        // 拒绝错误字段并保持之前成功加载的五板块配置。
        const std::vector<std::pair<std::string, std::string>> errors{
            {"mode: \"image\"", "mode: \"images\""},
            {"left_path: \"left.png\"", "left_path: \"\""},
            {"capacity: 7", "capacity: 0"},
            {"capacity: 7", "capacity: -1"},
            {"mode: \"image\"", "mode: \"unknown\""},
            {"max_timestamp_diff_ms: 0", "max_timestamp_diff_ms: -1"},
            {"input_size: 256", "input_size: 0"},
            {"tile_overlap_px: 7", "tile_overlap_px: 0.5"},
            {"left_padding_px: 2", "left_padding_px: -1"},
            {"confidence_threshold: 0.5", "confidence_threshold: .Nan"},
            {"nms_iou_threshold: 0.75", "nms_iou_threshold: 1.1"},
            {"warmup_count: 1", "warmup_count: 0"},
            {"hsv_lower: [50, 100, 80]", "hsv_lower: [180, 100, 80]"},
            {"hsv_upper: [86, 212, 255]", "hsv_upper: [40, 212, 255]"},
            {"hsv_upper: [86, 212, 255]", "hsv_upper: [86, 212]"},
            {"exposure_us: 0.", "exposure_us: -1."},
            {"image_height: 3648", "image_height: 2.5"},
            {"cameras:", "missing_cameras:"},
            {"device: \"CPU\"", "device: \"\""}
        };
        for (const auto& error : errors) {
            write(replace(image, error.first, error.second));
            bool rejected = false;
            try { config.load(fixture.string()); }
            catch (const std::runtime_error& exception) {
                rejected = true;
                require(std::string(exception.what()).find(fixture.string()) != std::string::npos,
                        "Missing configuration path in diagnostic");
            }
            require(rejected, "Invalid setting accepted: " + error.second);
            require(config.input.mode == InputMode::Image && config.yolo.inputSize == 256 &&
                    config.frameQueue.capacity == 7 && config.leftCamera.height == 3648 &&
                    config.guideLight.hsvLower == cv::Scalar(50, 100, 80), "Failed reload changed configuration");
        }
        write(image);
        std::cout << "Five-section config, calibration, path and " << errors.size()
                  << " invalid-setting checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

