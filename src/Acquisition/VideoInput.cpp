#include "Acquisition/VideoInput.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace {
std::string readString(const cv::FileNode& node, const std::string& field)
{
    if (!node.isString() || static_cast<std::string>(node).empty())
        throw std::runtime_error(field + ": expected a non-empty string");
    return static_cast<std::string>(node);
}

class ImagePairSource final : public StereoImageSource {
public:
    explicit ImagePairSource(const VideoInputConfig& config) : repeat_(config.repeat)
    {
        left_ = loadImage(config.leftPath, "left_camera.path");
        right_ = loadImage(config.rightPath, "right_camera.path");
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

std::unique_ptr<StereoImageSource> makeSource(const VideoInputConfig& config)
{
    switch (config.mode) {
    case VideoInputConfig::Mode::Images:
        return std::make_unique<ImagePairSource>(config);
    case VideoInputConfig::Mode::Video:
        throw std::runtime_error("mode=video is reserved but not implemented yet; use mode=images");
    }
    throw std::runtime_error("Unsupported input mode");
}
} // namespace

VideoInputConfig VideoInputConfig::load(const std::string& configPath)
{
    try {
        cv::FileStorage file(configPath, cv::FileStorage::READ);
        if (!file.isOpened()) throw std::runtime_error("cannot open configuration file");
        VideoInputConfig config;
        const auto mode = readString(file["mode"], "mode");
        if (mode == "images") config.mode = Mode::Images;
        else if (mode == "video") config.mode = Mode::Video;
        else throw std::runtime_error("mode: expected images or video");

        const auto base = std::filesystem::absolute(configPath).parent_path();
        const auto readPath = [&](const char* camera) {
            const auto node = file[camera];
            if (!node.isMap()) throw std::runtime_error(std::string(camera) + ": expected a mapping");
            const std::filesystem::path path(readString(node["path"], std::string(camera) + ".path"));
            return (path.is_absolute() ? path : base / path).lexically_normal().string();
        };
        config.leftPath = readPath("left_camera");
        config.rightPath = readPath("right_camera");

        const auto capacity = file["queue_capacity"];
        if (!capacity.isInt() || static_cast<int>(capacity) <= 0)
            throw std::runtime_error("queue_capacity: expected a positive integer");
        config.queueCapacity = static_cast<std::size_t>(static_cast<int>(capacity));
        const auto repeat = file["repeat"];
        if (!repeat.isInt() || (static_cast<int>(repeat) != 0 && static_cast<int>(repeat) != 1))
            throw std::runtime_error("repeat: expected 0 or 1");
        config.repeat = static_cast<int>(repeat) != 0;
        return config;
    } catch (const std::exception& error) {
        throw std::runtime_error("VideoInput [" + configPath + "]: " + error.what());
    }
}

VideoInput::VideoInput(const std::string& configPath) : VideoInput(VideoInputConfig::load(configPath)) {}

VideoInput::VideoInput(const VideoInputConfig& config) : VideoInput(makeSource(config), config.queueCapacity) {}

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
