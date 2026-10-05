#pragma once

#include "Acquisition/AcquireFrame.h"

#include <memory>
#include <string>

// 文件输入后端：以后视频后端只需实现此接口，并在模式工厂中注册。
class StereoImageSource {
public:
    virtual ~StereoImageSource() = default;
    // 成功返回两张 CV_8UC3 BGR 全图；结束返回 false；读取错误抛异常。
    // 可复用缓冲区，图像须保持有效直到下一次 readNext 调用。
    virtual bool readNext(cv::Mat& left, cv::Mat& right) = 0;
};

struct VideoInputConfig {
    enum class Mode { Images, Video };
    Mode mode = Mode::Images;
    std::string leftPath;
    std::string rightPath;
    std::size_t queueCapacity = 4; // 左右每侧可缓存的帧数
    bool repeat = true;           // 图片重复采集；false 时只输出一对

    // 相对媒体路径以 YAML 所在目录为基准；失败异常包含配置路径。
    static VideoInputConfig load(const std::string& configPath);
};

class VideoInput {
public:
    explicit VideoInput(const std::string& configPath = "config/video.yaml");
    explicit VideoInput(const VideoInputConfig& config);
    // 注入视频等新后端，队列和下游接口无需修改。
    VideoInput(std::unique_ptr<StereoImageSource> source, std::size_t queueCapacity);

    // 由单个采集线程调用，每次读取并入队一对；不自动启动线程或限帧。
    // 帧号从 0 递增；左右共用 steady_clock 毫秒时间戳（采集时间，非视频 PTS）。
    // 输入结束后持续返回 false；读取错误抛异常，不发布半对数据。
    bool captureNext();
    // 仅提供公共队列；匹配、取帧由 AcquireFrame 中的 StereoFrameQueue 负责。
    // 下游接收此引用后无需依赖 VideoInput；引用有效期不超过当前输入对象。
    StereoFrameQueue& frameQueue() noexcept;
    std::size_t queuedPairs() const;

private:
    std::unique_ptr<StereoImageSource> source_;
    StereoFrameQueue queue_;
    std::uint64_t nextFrameId_ = 0;
    bool exhausted_ = false;
};
