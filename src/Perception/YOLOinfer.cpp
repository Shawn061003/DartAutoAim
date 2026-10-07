#include "Perception/YOLOinfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/properties.hpp>

namespace {

// 当前取图策略保持左图四切、右图整图。
constexpr bool kLeftUseTiles = true;
constexpr std::array<std::size_t, 2> kBatchSizes{1, 4};

void CheckFrame(const cv::Mat& image)
{
    // 入口只接受非空的8位三通道图像；BGR顺序由调用方保证。
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC3) {
        throw std::invalid_argument("YOLO input must be a non-empty CV_8UC3 BGR image.");
    }
}

std::vector<cv::Rect> MakeRegions(const cv::Size& size, bool useTiles, int overlap)
{
    // 整图模式只生成一个覆盖原图的区域。
    if (!useTiles) {
        return {cv::Rect(0, 0, size.width, size.height)};
    }
    if (size.width <= overlap || size.height <= overlap) {
        throw std::invalid_argument("Four-tile input width/height must exceed yolo.tile_overlap_px.");
    }

    // 按配置宽度分配中线两侧重叠，奇数像素余量落在右/下侧。
    const int leftEnd = size.width / 2 + overlap / 2;
    const int rightStart = leftEnd - overlap;
    const int topEnd = size.height / 2 + overlap / 2;
    const int bottomStart = topEnd - overlap;
    return {
        cv::Rect(0, 0, leftEnd, topEnd),
        cv::Rect(rightStart, 0, size.width - rightStart, topEnd),
        cv::Rect(0, bottomStart, leftEnd, size.height - bottomStart),
        cv::Rect(rightStart, bottomStart, size.width - rightStart, size.height - bottomStart)
    };
}

} // namespace

struct YOLOInference::ModelRuntime {
    ov::Core core;
    std::array<ov::CompiledModel, 2> models;
    std::array<ov::InferRequest, 2> requests;
};

YOLOInference::YOLOInference(const YOLOParams& params) : params_(params)
{
    // 构造时校验参数、加载并预热，后续帧复用同一份资源。
    params_.validate();
    LoadYOLOModule(params_.modelPath, params_.device);
}

YOLOInference::YOLOInference(const std::string& modelPath, const std::string& device)
    : YOLOInference([&] {
        YOLOParams params;
        params.modelPath = modelPath;
        params.device = device;
        return params;
    }())
{
}

// ModelRuntime随对象析构，自动释放模型和推理请求。
YOLOInference::~YOLOInference() = default;

void YOLOInference::LoadYOLOModule(const std::string& modelPath, const std::string& device)
{
    // 1. 读取模型，确认输入尺寸、精度及输入输出数量。
    auto runtime = std::make_unique<ModelRuntime>();
    const auto original = runtime->core.read_model(modelPath);
    if (original->inputs().size() != 1 || original->outputs().size() != 1) {
        throw std::runtime_error("YOLO requires one image input and one detection output.");
    }
    const auto shape = original->input().get_partial_shape();
    if (shape.rank().is_dynamic() || shape.rank().get_length() != 4
        || !shape[1].compatible(3) || !shape[2].compatible(params_.inputSize)
        || !shape[3].compatible(params_.inputSize)) {
        throw std::runtime_error("YOLO model input must support NCHW [B,3,input_size,input_size]; check yolo.input_size.");
    }
    const auto type = original->input().get_element_type();
    if (type != ov::element::f32 && type != ov::element::f16) {
        throw std::runtime_error("YOLO model input must be FP32 or FP16.");
    }

    // 2. 分别准备batch=1和batch=4，运行时不再reshape或重新编译。
    for (std::size_t slot = 0; slot < kBatchSizes.size(); ++slot) {
        const std::size_t batch = kBatchSizes[slot];
        auto model = original->clone();
        model->reshape(ov::PartialShape{static_cast<std::int64_t>(batch), 3,
                                       params_.inputSize, params_.inputSize});

        // 统一对外使用FP32张量，模型内部所需的精度转换交给OpenVINO。
        ov::preprocess::PrePostProcessor processor(model);
        processor.input().tensor().set_element_type(ov::element::f32);
        processor.input().preprocess().convert_element_type(type);
        processor.output().tensor().set_element_type(ov::element::f32);
        model = processor.build();

        // 编译时优先低延迟；显式选择GPU时请求FP16计算。
        ov::AnyMap properties;
        properties[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        if (device == "GPU" || device.rfind("GPU.", 0) == 0) {
            properties[ov::hint::inference_precision.name()] = ov::element::f16;
        }
        runtime->models[slot] = runtime->core.compile_model(model, device, properties);
        runtime->requests[slot] = runtime->models[slot].create_infer_request();

        // 用零输入预热，初始化时确认单类别RAW输出[B,5,N]。
        auto input = runtime->requests[slot].get_input_tensor();
        std::fill_n(input.data<float>(), input.get_size(), 0.0F);
        for (int i = 0; i < params_.warmupCount; ++i) {
            runtime->requests[slot].infer();
        }
        const auto outputShape = runtime->requests[slot].get_output_tensor().get_shape();
        if (outputShape.size() != 3 || outputShape[0] != batch
            || outputShape[1] != 5 || outputShape[2] == 0) {
            throw std::runtime_error("YOLO output must be single-class RAW [B,5,N].");
        }
    }
    // 3. 两套资源均准备完成后，交由当前对象长期持有。
    model_runtime_ = std::move(runtime);
}

std::vector<YOLOInference::YOLOPreProcessResult> YOLOInference::PreProcess(
    const cv::Mat& image, const std::vector<cv::Rect>& regions)
{
    std::vector<YOLOPreProcessResult> results;
    results.reserve(regions.size());
    for (const auto& region : regions) {
        // 1. 按长边缩放到配置的输入边长，短边尺寸取整后居中放置。
        const double scale = std::min(static_cast<double>(params_.inputSize) / region.width,
                                      static_cast<double>(params_.inputSize) / region.height);
        const int width = std::clamp(static_cast<int>(std::round(region.width * scale)),
                                     1, params_.inputSize);
        const int height = std::clamp(static_cast<int>(std::round(region.height * scale)),
                                      1, params_.inputSize);
        const int left = (params_.inputSize - width) / 2;
        const int top = (params_.inputSize - height) / 2;
        // 2. 裁出当前区域并缩放，空白区域按YOLO约定填充114。
        cv::Mat resized;
        cv::resize(image(region), resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::Mat letterbox;
        cv::copyMakeBorder(resized, letterbox, top, params_.inputSize - height - top,
                           left, params_.inputSize - width - left, cv::BORDER_CONSTANT,
                           cv::Scalar(114, 114, 114));

        // 3. 保存取图原点、实际缩放比例和padding，供后处理还原坐标。
        YOLOPreProcessResult result;
        result.source_region = region;
        result.scale_x = static_cast<double>(width) / region.width;
        result.scale_y = static_cast<double>(height) / region.height;
        result.padding = cv::Point(left, top);
        // 4. BGR转RGB并归一化至[0,1]，排列为[1,3,inputSize,inputSize]。
        result.input = cv::dnn::blobFromImage(letterbox, 1.0 / 255.0, cv::Size(),
                                             cv::Scalar(), true, false, CV_32F);
        results.push_back(std::move(result));
    }
    return results;
}

YOLOInference::YOLOBatchResult YOLOInference::YOLOInfer(
    const std::vector<YOLOPreProcessResult>& inputs)
{
    // 1. 选择已预热的请求，按输入顺序将各图连续拷入batch张量。
    const std::size_t slot = inputs.size() == 1 ? 0 : 1;
    auto& request = model_runtime_->requests[slot];
    auto tensor = request.get_input_tensor();
    const std::size_t imageElements = 3 * static_cast<std::size_t>(params_.inputSize) * params_.inputSize;
    float* destination = tensor.data<float>();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const cv::Mat& input = inputs[i].input;
        std::memcpy(destination + i * imageElements, input.ptr<float>(),
                    imageElements * sizeof(float));
    }
    // 2. 同步推理；返回后输出张量才可读取。
    request.infer();

    // 3. 输出固定为[B,5,N]，五个通道依次为cx、cy、w、h、score。
    const auto output = request.get_output_tensor();
    const float* data = output.data<const float>();
    const std::size_t count = output.get_shape()[2];
    const std::size_t stride = 5 * count;
    YOLOBatchResult result;
    result.detections.resize(inputs.size());

    // 4. 逐图解码并筛选候选，无检测的图片保留空集合。
    for (std::size_t batch = 0; batch < inputs.size(); ++batch) {
        const float* values = data + batch * stride;
        for (std::size_t i = 0; i < count; ++i) {
            // 按通道读取候选，将中心坐标转换为左上角；单类别无需class_id。
            const float confidence = values[4 * count + i];
            const float width = values[2 * count + i];
            const float height = values[3 * count + i];
            const float x = values[i] - width * 0.5F;
            const float y = values[count + i] - height * 0.5F;
            // 过滤低分、非有限数值和无效尺寸，留下可用的检测框。
            if (!std::isfinite(confidence) || confidence <= params_.confidenceThreshold
                || confidence > 1.0F || !std::isfinite(x) || !std::isfinite(y)
                || !std::isfinite(width) || !std::isfinite(height)
                || width <= 0.0F || height <= 0.0F) {
                continue;
            }
            result.detections[batch].push_back({cv::Rect2f(x, y, width, height), confidence});
        }
    }
    return result;
}

YOLOInference::YOLOResults YOLOInference::PostProcess(
    const YOLOBatchResult& batchResult,
    const std::vector<YOLOPreProcessResult>& inputs,
    const CameraFrame& frame, int roiPadding)
{
    // 1. 将各图候选统一还原到原图坐标，再汇总处理。
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& info = inputs[i];
        for (const auto& detection : batchResult.detections[i]) {
            const auto& box = detection.box;
            // 先去掉padding并限制到实际取图区域，排除完全位于填充带中的框。
            const double x1 = std::clamp((box.x - info.padding.x) / info.scale_x,
                                         0.0, static_cast<double>(info.source_region.width));
            const double y1 = std::clamp((box.y - info.padding.y) / info.scale_y,
                                         0.0, static_cast<double>(info.source_region.height));
            const double x2 = std::clamp(
                (static_cast<double>(box.x) + box.width - info.padding.x) / info.scale_x,
                0.0, static_cast<double>(info.source_region.width));
            const double y2 = std::clamp(
                (static_cast<double>(box.y) + box.height - info.padding.y) / info.scale_y,
                0.0, static_cast<double>(info.source_region.height));
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
            // 加上取图原点，将切块内的局部坐标转换为原图坐标。
            boxes.emplace_back(x1 + info.source_region.x, y1 + info.source_region.y,
                               x2 - x1, y2 - y1);
            scores.push_back(static_cast<float>(detection.conf));
        }
    }

    // 2. 跨切块执行NMS，去除重叠区域重复检出的目标。
    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, scores, params_.confidenceThreshold, params_.nmsIouThreshold, kept);
    // 3. 优先保留最高分框，再找与它无交集的最高分框；最多取两个。
    std::stable_sort(kept.begin(), kept.end(),
        [&scores](int a, int b) { return scores[a] > scores[b]; });
    YOLOResults results;
    results.reserve(2);
    for (const int index : kept) {
        // 框面积均为正，交集面积为0等价于IoU=0；判断发生在ROI外扩前。
        if (!results.empty() && (boxes[kept.front()] & boxes[index]).area() > 0.0) {
            continue;
        }

        // 4. 分别外扩、向外取整并裁边，只输出区域和来源信息。
        const auto& box = boxes[index];
        const int x1 = static_cast<int>(std::floor(std::max(0.0, box.x - roiPadding)));
        const int y1 = static_cast<int>(std::floor(std::max(0.0, box.y - roiPadding)));
        const int x2 = static_cast<int>(std::ceil(
            std::min(static_cast<double>(frame.image.cols), box.x + box.width + roiPadding)));
        const int y2 = static_cast<int>(std::ceil(
            std::min(static_cast<double>(frame.image.rows), box.y + box.height + roiPadding)));
        results.push_back({cv::Rect(x1, y1, x2 - x1, y2 - y1), scores[index],
                           frame.frame_id, frame.timestamp_ms, frame.camera_side});
        if (results.size() == 2) {
            break;
        }
    }
    return results;
}

YOLOInference::StereoYOLOResult YOLOInference::RunYOLOInfer(
    const CameraFrame& leftFrame, const CameraFrame& rightFrame)
{
    // 1. 检查两路输入；时间配对由上游采集模块完成。
    if (leftFrame.camera_side != CameraSide::Left || rightFrame.camera_side != CameraSide::Right)
        throw std::invalid_argument("YOLO requires Left/Right camera_side on the corresponding input frames");
    CheckFrame(leftFrame.image);
    CheckFrame(rightFrame.image);

    // 2. 左相机按固定模式完成取图、预处理、推理和ROI生成。
    const auto leftInputs = PreProcess(leftFrame.image,
                                       MakeRegions(leftFrame.image.size(), kLeftUseTiles, params_.tileOverlapPx));
    const auto leftBatch = YOLOInfer(leftInputs);
    StereoYOLOResult result;
    result.left = PostProcess(leftBatch, leftInputs, leftFrame, params_.leftPaddingPx);
    // 3. 四切后处理无有效ROI时回退整图，包括候选全部位于填充区域的情况。
    if (kLeftUseTiles && result.left.empty()) {
        const auto fullInput = PreProcess(leftFrame.image, MakeRegions(leftFrame.image.size(), false, params_.tileOverlapPx));
        result.left = PostProcess(YOLOInfer(fullInput), fullInput, leftFrame, params_.leftPaddingPx);
    }

    // 4. 右相机始终整图推理，与左相机独立选择最多两个目标。
    const auto rightInput = PreProcess(rightFrame.image, MakeRegions(rightFrame.image.size(), false, params_.tileOverlapPx));
    result.right = PostProcess(YOLOInfer(rightInput), rightInput, rightFrame, params_.rightPaddingPx);
    return result;
}
