#include "Acquisition/VideoInput.h"

#include <chrono>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace {
class ImagePairSource final : public StereoImageSource {
public:
    ImagePairSource(const InputParams& input, bool repeat) : repeat_(repeat)
    {
        left_ = loadImage(input.leftPath, "input.left_path");
        right_ = loadImage(input.rightPath, "input.right_path");
    }

    bool readNext(cv::Mat& left, cv::Mat& right) override
    {
        if (emitted_ && !repeat_) return false;
        left = left_;
        right = right_;
        emitted_ = true;
        return true;
    }

private:
    static cv::Mat loadImage(const std::string& path, const char* field)
    {
        // 保持原始尺寸和像素坐标，不应用 EXIF 旋转；统一解码成 8 位 BGR。
        auto image = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (image.empty()) throw std::runtime_error(std::string(field) + ": cannot read image [" + path + "]");
        return image;
    }
    cv::Mat left_, right_;
    bool repeat_;
    bool emitted_ = false;
};

std::unique_ptr<StereoImageSource> makeSource(const InputParams& input, bool repeat)
{
    switch (input.mode) {
    case InputMode::Image:
        return std::make_unique<ImagePairSource>(input, repeat);
    // 文件输入后端在创建数据源时检查模式支持情况。
    case InputMode::Camera:
        throw std::invalid_argument("input.mode: camera requires the camera acquisition backend");
    case InputMode::Video:
        throw std::runtime_error("mode=video is reserved but not implemented yet; use input.mode=image");
    }
    throw std::runtime_error("Unsupported input mode");
}
} // namespace

VideoInput::VideoInput(const InputParams& input, std::size_t queueCapacity, bool repeat)
    : VideoInput(makeSource(input, repeat), queueCapacity) {}

VideoInput::VideoInput(std::unique_ptr<StereoImageSource> source, std::size_t queueCapacity)
    : source_(std::move(source)), queue_(queueCapacity)
{
    if (!source_) throw std::invalid_argument("VideoInput requires a source");
}

bool VideoInput::captureNext()
{
    if (exhausted_) return false;
    // 入队前即使用完整 CameraFrame；队列直接保存该类型，出队不重建元数据。
    CameraFrame left, right;
    if (!source_->readNext(left.image, right.image)) {
        exhausted_ = true;
        return false;
    }
    left.frame_id = right.frame_id = nextFrameId_;
    left.timestamp_ms = right.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    queue_.push(left, right);
    ++nextFrameId_;
    return true;
}

StereoFrameQueue& VideoInput::frameQueue() noexcept { return queue_; }

std::size_t VideoInput::queuedPairs() const { return queue_.size(); }
