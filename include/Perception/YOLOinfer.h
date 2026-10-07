#ifndef YOLO_INFER_H
#define YOLO_INFER_H

// 坐标约定：box位于inputSize × inputSize模型输入坐标系；source_region和roi位于原图坐标系。
// 矩形使用左上角x/y和宽/高；padding为模型输入中的左、上填充像素数。
// 后处理先减padding、除以实际缩放比例，再加source_region左上角，得到原图坐标。
// roi为外扩、向外取整并裁至原图边界后的区域，x/y为其左上角。
// 下游根据roi裁剪图像，局部坐标加roi左上角还原至原图。

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "Acquisition/CameraInput.h"
#include "common/DartConfig.h"

class YOLOInference {
public:

    struct YOLOInferResult {
        cv::Rect roi;                   // 最终ROI区域，不保存裁剪图像
        double conf = 0.0;              // 选中目标的置信度
        std::uint64_t frame_id = 0;      // 来源帧号
        std::int64_t timestamp_ms = 0;   // 来源帧采集时间戳（毫秒）
        CameraSide camera_side = CameraSide::Unknown; // 从输入帧继承，单独取出ROI也保留来源
    };

    using YOLOResults = std::vector<YOLOInferResult>;
    using LeftYOLOResult = YOLOResults;
    using RightYOLOResult = YOLOResults;

    struct StereoYOLOResult {
        LeftYOLOResult left;     // 每路0~2个结果，按置信度降序排列
        RightYOLOResult right;
    };

    /// @brief 使用 YAML 解析后的完整参数加载模型、编译并预热。
    explicit YOLOInference(const YOLOParams& params);

    /// @brief 离线工具快捷入口：使用默认检测参数，显式指定模型与设备。
    YOLOInference(const std::string& modelPath, const std::string& device);

    /// @brief 释放模型资源；定义放在.cpp中。
    ~YOLOInference();

    // 推理资源由当前对象独立持有，不允许复制。
    YOLOInference(const YOLOInference&) = delete;
    YOLOInference& operator=(const YOLOInference&) = delete;

    /// @brief 同步处理一对已配对图像，返回左右各自最多两个目标的ROI及帧信息。
    /// @pre 两帧camera_side须分别为Left/Right；Unknown或左右颠倒时抛异常。
    /// @note 右相机整图；左相机整图/四切，不做自适应，四切后处理无有效ROI时回退整图。
    ///       四切总重叠和左右ROI外扩量由YOLOParams指定。
    ///       不启动线程；调用间保留模型，不重复加载或warmup，同一对象不并发调用。
    StereoYOLOResult RunYOLOInfer(const CameraFrame& leftFrame, const CameraFrame& rightFrame);

private:

    struct YOLODetection {
        cv::Rect2f box;          // 候选检测框
        double conf = 0.0;      // 目标置信度
    };

    struct YOLOPreProcessResult {
        cv::Mat input;          // 单张inputSize × inputSize输入，布局、通道顺序和类型与模型一致
        cv::Rect source_region;
        double scale_x = 1.0;   // 实际缩放后内容宽度 / 原区域宽度
        double scale_y = 1.0;   // 实际缩放后内容高度 / 原区域高度
        cv::Point padding;
    };

    struct YOLOBatchResult {
        // 外层对应输入图片，内层为该图片的有效候选；无候选时保留空集合。
        // 四切时0~3依次为左上、右上、左下、右下；整图时只有一组。
        std::vector<std::vector<YOLODetection>> detections;
    };

    // 模型、编译结果和推理请求在.cpp中定义，空闲时保留，随对象析构释放。
    const YOLOParams params_; // 初始化后保持不变，预处理与模型形状共用同一配置
    struct ModelRuntime;
    std::unique_ptr<ModelRuntime> model_runtime_;

    /// @brief 初始化阶段加载、编译模型，完成单张/四张输入所需的warmup。
    void LoadYOLOModule(const std::string& modelPath, const std::string& device);

    /// @brief 按regions取图，等比例缩放、padding至inputSize × inputSize并转换为模型输入格式。
    /// @return 与regions顺序一致的输入及还原信息；regions须为原图内有效区域。
    std::vector<YOLOPreProcessResult> PreProcess(const cv::Mat& image, const std::vector<cv::Rect>& regions);

    /// @brief 对同一相机同一帧的一张或四张输入执行推理，解码并筛选有效候选。
    /// @return 每张输入的候选集合，组数和顺序与inputs一致。
    YOLOBatchResult YOLOInfer(const std::vector<YOLOPreProcessResult>& inputs);

    /// @brief 去重后先选最高分框，再选与其IoU=0的最高分框，各自外扩生成ROI。
    /// @return 按置信度降序返回0~2个ROI及来源信息，图像裁剪由下游完成。
    /// @note IoU按原图坐标下未外扩的检测框计算；不足两个时保留实际数量。
    YOLOResults PostProcess(const YOLOBatchResult& batchResult,
                                             const std::vector<YOLOPreProcessResult>& inputs,
                                             const CameraFrame& frame, int roiPadding);

};

#endif // YOLO_INFER_H
