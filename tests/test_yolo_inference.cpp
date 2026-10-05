#include "Perception/YOLOinfer.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/pass/serialize.hpp>

namespace {
void Require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void WriteFixture(const std::filesystem::path& path,
                  const std::vector<std::array<float, 5>>& rows, bool mixed = false, int inputSize = 512)
{
    namespace op = ov::opset13;
    auto input = std::make_shared<op::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, inputSize, inputSize});
    auto mean = std::make_shared<op::ReduceMean>(input,
        op::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3}), false);
    auto axis = op::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto red = std::make_shared<op::Gather>(mean,
        op::Constant::create(ov::element::i64, ov::Shape{1}, {0}), axis);
    ov::Output<ov::Node> score = red;
    if (mixed) {
        auto blue = std::make_shared<op::Gather>(mean,
            op::Constant::create(ov::element::i64, ov::Shape{1}, {2}), axis);
        score = std::make_shared<op::Multiply>(std::make_shared<op::Multiply>(red, blue),
            op::Constant::create(ov::element::f32, ov::Shape{1, 1}, {4.0F}));
    }
    auto zero = std::make_shared<op::Multiply>(score,
        op::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0.0F}));
    ov::OutputVector channels;
    for (int channel = 0; channel < 5; ++channel) {
        std::vector<float> values;
        for (const auto& row : rows) {
            values.push_back(row[channel]);
        }
        auto constants = op::Constant::create(ov::element::f32, ov::Shape{1, rows.size()}, values);
        ov::Output<ov::Node> data;
        if (channel == 4) {
            data = std::make_shared<op::Multiply>(score, constants);
        } else {
            data = std::make_shared<op::Add>(zero, constants);
        }
        channels.push_back(std::make_shared<op::Unsqueeze>(data,
            op::Constant::create(ov::element::i64, ov::Shape{1}, {1})));
    }
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{std::make_shared<op::Concat>(channels, 1)}, ov::ParameterVector{input});
    ov::serialize(model, path.string());
}

// 用红色均值驱动分数，验证RGB预处理和batch顺序；混色模型用于验证整图回退。
void GenerateFixtures(const std::filesystem::path& path)
{
    std::filesystem::create_directories(path);
    WriteFixture(path / "raw.xml", {{256, 256, 20, 20, 1}});
    WriteFixture(path / "fallback.xml", {{256, 256, 20, 20, 1}}, true);
    WriteFixture(path / "edge.xml", {{5, 5, 20, 20, 1}});
    WriteFixture(path / "two_targets.xml", {
        {128, 128, 20, 20, .80F}, {384, 384, 20, 20, .95F}, {256, 256, 20, 20, .70F}});
    WriteFixture(path / "overlap_skip.xml", {
        {120, 120, 40, 40, .95F}, {122, 122, 40, 40, .94F},
        {150, 120, 40, 40, .90F}, {310, 110, 20, 20, .85F}, {410, 410, 20, 20, .80F}});
    WriteFixture(path / "overlap_only.xml", {
        {120, 120, 40, 40, .95F}, {150, 120, 40, 40, .90F}});
    WriteFixture(path / "touching.xml", {
        {110, 110, 20, 20, .95F}, {130, 110, 20, 20, .90F}});
    WriteFixture(path / "tiny_overlap.xml", {
        {110, 110, 20, 20, .95F}, {129.875F, 110, 20, 20, .90F}});
    // 最高分框与其余两个框分别重叠，而后两者互不重叠：仍优先保留最高分。
    WriteFixture(path / "keep_best.xml", {
        {140, 120, 80, 40, .95F}, {105, 120, 30, 40, .90F}, {175, 120, 30, 40, .85F}});
}


void Expect(const YOLOInference::YOLOResults& results,
            const cv::Rect& roi, std::uint64_t frame, std::int64_t timestamp, std::size_t index = 0)
{
    Require(results.size() > index && results.size() <= 2, "Unexpected detection count.");
    const auto& result = results[index];
    Require(result.roi == roi, "Unexpected ROI: " + std::to_string(result.roi.x)
        + "," + std::to_string(result.roi.y) + "," + std::to_string(result.roi.width)
        + "," + std::to_string(result.roi.height));
    Require(result.frame_id == frame && result.timestamp_ms == timestamp,
            "Frame metadata was not preserved.");
}
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 1;
    }
    try {
        const std::filesystem::path models(argv[1]);
        GenerateFixtures(models);
        YOLOInference detector((models / "raw.xml").string(), "CPU");
        CameraFrame right{cv::Mat(512, 512, CV_8UC3, cv::Scalar(0, 0, 255)), 7, 1234};
        CameraFrame left{cv::Mat(), 9, 1235};

        // 994切成四张512，单独点亮每个象限，验证四个batch位置和482px偏移。
        for (int quadrant = 0; quadrant < 4; ++quadrant) {
            left.image = cv::Mat::zeros(994, 994, CV_8UC3);
            const int x = quadrant % 2 * 497;
            const int y = quadrant / 2 * 497;
            left.image(cv::Rect(x, y, 497, 497)).setTo(cv::Scalar(0, 0, 255));
            const auto result = detector.RunYOLOInfer(left, right);
            Expect(result.left, cv::Rect(221 + quadrant % 2 * 482,
                                         221 + quadrant / 2 * 482, 70, 70), 9, 1235);
            Expect(result.right, cv::Rect(236, 236, 40, 40), 7, 1234);
        }

        // 非方形输入含padding；像素均值也能检查114填充值、归一化和RGB顺序。
        right.image = cv::Mat(384, 512, CV_8UC3, cv::Scalar(0, 0, 255));
        auto result = detector.RunYOLOInfer(left, right);
        Expect(result.right, cv::Rect(236, 172, 40, 40), 7, 1234);
        Require(std::abs(result.right.front().conf - (0.75 + 0.25 * 114.0 / 255.0)) < 0.01,
                "Letterbox pixels or normalization are incorrect.");

        // 非连续Mat也可作为输入；每次请求必须覆盖上一次的tensor。
        cv::Mat backing(514, 514, CV_8UC3, cv::Scalar(0, 0, 255));
        right.image = backing(cv::Rect(1, 1, 512, 512));
        Expect(detector.RunYOLOInfer(left, right).right, cv::Rect(236, 236, 40, 40), 7, 1234);
        right.image.setTo(cv::Scalar(255, 0, 0));
        Require(detector.RunYOLOInfer(left, right).right.empty(), "BGR/RGB mapping is incorrect.");

        // 四块中红蓝各自集中，分数低于阈值；只有整图混合后能检出。
        YOLOInference fallback((models / "fallback.xml").string(), "CPU");
        left.image = cv::Mat(994, 994, CV_8UC3, cv::Scalar(255, 0, 0));
        left.image(cv::Rect(0, 0, 497, 994)).setTo(cv::Scalar(0, 0, 255));
        right.image = cv::Mat::zeros(512, 512, CV_8UC3);
        result = fallback.RunYOLOInfer(left, right);
        Expect(result.left, cv::Rect(452, 452, 90, 90), 9, 1235);
        Require(result.right.empty(), "Blank right image should have no detection.");

        // 负坐标框外扩后仍须裁至图像边界。
        YOLOInference edge((models / "edge.xml").string(), "CPU");
        left.image = cv::Mat(994, 994, CV_8UC3, cv::Scalar(0, 0, 255));
        right.image = cv::Mat(512, 512, CV_8UC3, cv::Scalar(0, 0, 255));
        result = edge.RunYOLOInfer(left, right);
        Expect(result.left, cv::Rect(0, 0, 40, 40), 9, 1235);
        Expect(result.right, cv::Rect(0, 0, 25, 25), 7, 1234);

        // 三个分离目标只保留最高分的两个，输出顺序不依赖模型行顺序。
        left.image.setTo(cv::Scalar());
        YOLOInference twoTargets((models / "two_targets.xml").string(), "CPU");
        result = twoTargets.RunYOLOInfer(left, right);
        Require(result.left.empty() && result.right.size() == 2, "Expected exactly two right targets.");
        Expect(result.right, cv::Rect(364, 364, 40, 40), 7, 1234, 0);
        Expect(result.right, cv::Rect(108, 108, 40, 40), 7, 1234, 1);
        Require(result.right[0].conf > result.right[1].conf, "Results are not sorted by confidence.");

        // 完全重复框由NMS抑制；轻微重叠框虽通过NMS，也不能成为第二目标。
        YOLOInference overlapSkip((models / "overlap_skip.xml").string(), "CPU");
        result = overlapSkip.RunYOLOInfer(left, right);
        Require(result.right.size() == 2, "Expected a lower-score disjoint second target.");
        Expect(result.right, cv::Rect(90, 90, 60, 60), 7, 1234, 0);
        Expect(result.right, cv::Rect(290, 90, 40, 40), 7, 1234, 1);

        YOLOInference overlapOnly((models / "overlap_only.xml").string(), "CPU");
        result = overlapOnly.RunYOLOInfer(left, right);
        Require(result.right.size() == 1, "Overlapping candidates must not fill the second slot.");

        // 边缘相接的检测框IoU=0，即使外扩后的ROI重叠，仍保留两个。
        YOLOInference touching((models / "touching.xml").string(), "CPU");
        result = touching.RunYOLOInfer(left, right);
        Require(result.right.size() == 2, "Touching boxes should both be retained.");
        Expect(result.right, cv::Rect(90, 90, 40, 40), 7, 1234, 0);
        Expect(result.right, cv::Rect(110, 90, 40, 40), 7, 1234, 1);

        YOLOInference tinyOverlap((models / "tiny_overlap.xml").string(), "CPU");
        Require(tinyOverlap.RunYOLOInfer(left, right).right.size() == 1,
                "Any positive overlap must exclude the second candidate.");
        YOLOInference keepBest((models / "keep_best.xml").string(), "CPU");
        result = keepBest.RunYOLOInfer(left, right);
        Require(result.right.size() == 1, "A disjoint lower-score pair must not replace the highest score.");
        Expect(result.right, cv::Rect(90, 90, 100, 60), 7, 1234);

        // 同一batch中不同切块的相同局部框，须在原图坐标下判断是否重叠。
        left.image.setTo(cv::Scalar(0, 0, 255));
        result = detector.RunYOLOInfer(left, right);
        Require(result.left.size() == 2 && result.right.size() == 1, "Per-camera counts are incorrect.");
        Expect(result.left, cv::Rect(221, 221, 70, 70), 9, 1235, 0);
        Expect(result.left, cv::Rect(703, 221, 70, 70), 9, 1235, 1);


        // 没有检测时不得返回上一次的ROI；空图须报错。
        left.image.setTo(cv::Scalar());
        right.image.setTo(cv::Scalar());
        result = detector.RunYOLOInfer(left, right);
        Require(result.left.empty() && result.right.empty(), "Empty detections reused a previous ROI.");
        right.image.release();
        bool rejected = false;
        try { detector.RunYOLOInfer(left, right); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Empty frame was not rejected.");

        // 独立构造 256 输入模型，验证边长、切块重叠和左右 padding 实际生效。
        WriteFixture(models / "small.xml", {{128, 128, 20, 20, 1}}, false, 256);
        YOLOParams params;
        params.modelPath = (models / "small.xml").string();
        params.inputSize = 256;
        params.tileOverlapPx = 0;
        params.leftPaddingPx = 3;
        params.rightPaddingPx = 7;
        params.warmupCount = 1;
        CameraFrame configuredLeft{cv::Mat(512, 512, CV_8UC3, cv::Scalar(0, 0, 255)), 5, 10};
        CameraFrame configuredRight{cv::Mat(256, 256, CV_8UC3, cv::Scalar(0, 0, 255)), 6, 10};
        YOLOInference configured(params);
        auto configuredResult = configured.RunYOLOInfer(configuredLeft, configuredRight);
        Expect(configuredResult.left, cv::Rect(115, 115, 26, 26), 5, 10);
        Expect(configuredResult.right, cv::Rect(111, 111, 34, 34), 6, 10);

        params.confidenceThreshold = 1.0F;
        YOLOInference filtered(params);
        Require(filtered.RunYOLOInfer(configuredLeft, configuredRight).right.empty(),
                "Configured confidence threshold was ignored.");

        params.confidenceThreshold = 0.25F;
        params.inputSize = 512;
        bool mismatchRejected = false;
        try { YOLOInference bad(params); }
        catch (const std::runtime_error&) { mismatchRejected = true; }
        Require(mismatchRejected, "Model/input_size mismatch was accepted.");

        // 调整 NMS 阈值后，第一框抑制中间框，使后续分离框保留下来。
        WriteFixture(models / "nms_chain.xml", {
            {120, 120, 40, 40, .95F}, {150, 120, 40, 40, .90F}, {180, 120, 40, 40, .85F}});
        params.modelPath = (models / "nms_chain.xml").string();
        params.nmsIouThreshold = 0.1F;
        params.rightPaddingPx = 0;
        configuredRight.image = cv::Mat(512, 512, CV_8UC3, cv::Scalar(0, 0, 255));
        YOLOInference nmsConfigured(params);
        configuredResult = nmsConfigured.RunYOLOInfer(configuredLeft, configuredRight);
        Expect(configuredResult.right, cv::Rect(160, 100, 40, 40), 6, 10, 1);
        std::cout << "YOLO regression checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

