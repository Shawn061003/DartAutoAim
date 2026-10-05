// 直接测试调参入口内部的参数分流和绘制链，避免测试另写一份相同逻辑。
#define DART_HSV_TUNER_TEST
#include "test_hsv_tuner.cpp"

#include <fstream>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

bool success(const GuideLightDetect::DetectResult& side)
{
    return side.size() == 1 && side[0].status == GuideLightDetect::DetectStatus::SUCCESS;
}

void checkIndependentBounds()
{
    StereoFrame frame{{cv::Mat::zeros(80, 100, CV_8UC3), 10, 100, CameraSide::Left},
                      {cv::Mat::zeros(120, 160, CV_8UC3), 11, 101, CameraSide::Right}};
    cv::circle(frame.left.image, {40, 40}, 10, {80, 200, 80}, cv::FILLED);
    cv::circle(frame.right.image, {90, 70}, 10, {80, 200, 80}, cv::FILLED);
    YOLOInference::StereoYOLOResult rois{
        {{{20, 20, 40, 40}, .9, 10, 100, CameraSide::Left}},
        {{{70, 50, 40, 40}, .8, 11, 101, CameraSide::Right}}};
    const Bounds green{60, 102, 81, 86, 212, 255};
    const Bounds red{0, 102, 81, 10, 212, 255};
    const auto baseline = detectPair(frame, rois, green, green);
    require(success(baseline.left) && success(baseline.right), "Baseline failed.");
    require(baseline.left[0].camera_side == CameraSide::Left
            && baseline.right[0].camera_side == CameraSide::Right, "Tuner lost camera provenance.");
    const auto changeLeft = detectPair(frame, rois, red, green);
    require(!success(changeLeft.left) && success(changeLeft.right), "Left bounds leaked into right.");
    require(changeLeft.right[0].contours == baseline.right[0].contours
            && changeLeft.right[0].CenterPoint == baseline.right[0].CenterPoint,
            "Right result changed with left-only tuning.");
    const auto changeRight = detectPair(frame, rois, green, red);
    require(success(changeRight.left) && !success(changeRight.right), "Right bounds leaked into left.");
    require(changeRight.left[0].contours == baseline.left[0].contours,
            "Left contour changed with right-only tuning.");
    const auto canvas = drawPair(frame, changeRight);
    require(canvas.size() == cv::Size(1280, 532), "Existing visualization was not used.");
    const auto empty = detectPair(frame, {}, green, red);
    require(empty.left.empty() && empty.right.empty(), "No-ROI results were fabricated.");
    require(!drawPair(frame, empty).empty(), "No-ROI view failed.");
    require(toBounds(toParams(green, red), CameraSide::Left) == green
            && toBounds(toParams(green, red), CameraSide::Right) == red, "HSV channel order changed.");
    bool rejected = false;
    try { toParams({86, 102, 81, 60, 212, 255}, green); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Crossed HSV bounds were accepted.");
}

void checkModeGuard(const std::filesystem::path& directory)
{
    std::filesystem::create_directories(directory);
    // 其余字段有效但文件都不存在；必须先报模式错误，不能读媒体或加载模型。
    const std::string camera =
        "      device_ip: \"\"\n      net_ip: \"\"\n"
        "      exposure_us: 0.\n      gain: 0.\n      extra_info_delay_s: 0.\n"
        "      image_width: 100\n      image_height: 80\n"
        "      camera_matrix: !!opencv-matrix\n"
        "         rows: 3\n         cols: 3\n         dt: d\n"
        "         data: [100., 0., 50., 0., 100., 40., 0., 0., 1.]\n"
        "      distortion_coefficients: !!opencv-matrix\n"
        "         rows: 5\n         cols: 1\n         dt: d\n"
        "         data: [0., 0., 0., 0., 0.]\n";
    for (const std::string mode : {"camera", "video"}) {
        const auto path = directory / (mode + ".yaml");
        std::ofstream file(path);
        file << "%YAML:1.0\n---\ninput:\n   mode: \"" << mode << "\"\n"
             << "   left_path: \"missing-left.bmp\"\n   right_path: \"missing-right.bmp\"\n"
             << "cameras:\n   left:\n" << camera << "   right:\n" << camera
             << "frame_queue:\n   capacity: 2\n   max_timestamp_diff_ms: 5\n"
             << "yolo:\n   model_path: \"missing.onnx\"\n   device: \"CPU\"\n"
             << "   input_size: 512\n   tile_overlap_px: 30\n"
             << "   left_padding_px: 25\n   right_padding_px: 10\n"
             << "   confidence_threshold: 0.25\n   nms_iou_threshold: 0.45\n   warmup_count: 1\n"
             << "guide_light:\n   left_hsv_lower: [61, 210, 140]\n   left_hsv_upper: [74, 230, 240]\n   right_hsv_lower: [60, 102, 81]\n   right_hsv_upper: [86, 212, 255]\n";
        file.close();
        require(bool(file), "Cannot create mode fixture.");
        bool rejected = false;
        try { runTuner({path.string(), true}); }
        catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()).find("requires input.mode=image") != std::string::npos;
        }
        require(rejected, "Non-image mode was not rejected before reading media/model.");
    }
}
} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 3, "Expected config path and fixture directory.");
        checkIndependentBounds();
        checkModeGuard(argv[2]);
        const auto config = loadImageConfig(argv[1]);
        const auto left = toBounds(config.guideLight, CameraSide::Left);
        const auto right = toBounds(config.guideLight, CameraSide::Right);
        const auto restored = toParams(left, right);
        require(restored.leftHsvLower == config.guideLight.leftHsvLower
                && restored.leftHsvUpper == config.guideLight.leftHsvUpper
                && restored.rightHsvLower == config.guideLight.rightHsvLower
                && restored.rightHsvUpper == config.guideLight.rightHsvUpper,
                "Per-camera YAML HSV initialization changed.");
        char name[] = "dart_hsv_tuner";
        char flag[] = "--config";
        char check[] = "--check";
        char* args[]{name, flag, argv[1], check};
        const auto options = parseOptions(4, args);
        require(options.checkOnly && options.configPath == argv[1], "CLI did not select YAML.");
        // 实际图片的读取、YOLO推理和左右可视化单独由--check验证，不依赖GUI。
        std::cout << "HSV tuner checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
