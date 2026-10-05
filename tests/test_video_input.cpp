#include "Acquisition/VideoInput.h"
#include "Acquisition/AcquireFrame.h"
#include "common/DartConfig.h"

#include <atomic>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <opencv2/imgcodecs.hpp>

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void expectError(const std::function<void()>& action, const std::string& fragment)
{
    try { action(); }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(fragment) != std::string::npos,
                "Wrong diagnostic: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected an error containing: " + fragment);
}

// 模拟未来解码器反复覆盖同一块 Mat 缓冲区，第三次读取后结束。
class BufferedSource final : public StereoImageSource {
public:
    bool readNext(cv::Mat& left, cv::Mat& right) override
    {
        if (count_ == 3) return false;
        buffer_.setTo(cv::Scalar(++count_, 2, 3));
        left = buffer_;
        right = buffer_;
        return true;
    }
private:
    int count_ = 0;
    cv::Mat buffer_ = cv::Mat(2, 3, CV_8UC3);
};

void testSourceAndOwnership()
{
    std::optional<StereoFrame> retained;
    {
        VideoInput input(std::make_unique<BufferedSource>(), 3);
        for (int i = 0; i < 3; ++i) require(input.captureNext(), "Source ended early");
        require(!input.captureNext() && !input.captureNext(), "EOF is not sticky");
        require(input.queuedPairs() == 3, "EOF discarded queued frames");
        std::int64_t previousTime = 0;
        for (int i = 0; i < 3; ++i) {
            retained = input.frameQueue().GetFrame();
            require(retained.has_value(), "Missing decoded pair");
            require(retained->left.frame_id == static_cast<std::uint64_t>(i) &&
                    retained->right.frame_id == retained->left.frame_id, "Frame IDs out of order");
            require(retained->left.timestamp_ms == retained->right.timestamp_ms &&
                    retained->left.timestamp_ms >= previousTime, "Clock mismatch");
            previousTime = retained->left.timestamp_ms;
            require(retained->left.image.at<cv::Vec3b>(0, 0)[0] == i + 1,
                    "Decoder reuse overwrote queued data");
            retained->left.image.setTo(cv::Scalar(99, 99, 99));
            require(retained->right.image.at<cv::Vec3b>(0, 0)[0] == i + 1,
                    "Left and right buffers alias");
        }
    }
    require(retained->right.image.at<cv::Vec3b>(0, 0)[0] == 3,
            "Frame lost validity after source destruction");
}

void testImages(const std::filesystem::path& dir)
{
    // 不同尺寸、灰度与 BGRA 文件均转成原尺寸 BGR，左右内容应不同。
    require(cv::imwrite((dir / "left.png").string(), cv::Mat(3, 4, CV_8UC1, cv::Scalar(17))),
            "Cannot write left fixture");
    require(cv::imwrite((dir / "right.png").string(), cv::Mat(5, 6, CV_8UC4, cv::Scalar(10, 20, 30, 40))),
            "Cannot write right fixture");
    // 采集测试直接构造输入参数；YAML 解析与路径解析由配置模块测试覆盖。
    const InputParams config{InputMode::Image, (dir / "left.png").string(),
                             (dir / "right.png").string()};
    VideoInput input(config, 2, true);
    require(!input.frameQueue().GetFrame(), "Empty queue returned a valid result");
    for (int i = 0; i < 3; ++i) require(input.captureNext(), "Repeat mode ended");
    require(input.queuedPairs() == 2, "Queue exceeded capacity");
    auto pair = input.frameQueue().GetFrame();
    require(pair && pair->left.frame_id == 1 && pair->right.frame_id == 1,
            "Overflow did not discard oldest pair");
    require(pair->left.image.type() == CV_8UC3 && pair->right.image.type() == CV_8UC3,
            "Image format is not BGR8");
    require(pair->left.image.size() == cv::Size(4, 3) && pair->right.image.size() == cv::Size(6, 5),
            "Image dimensions changed");
    require(pair->left.image.at<cv::Vec3b>(0, 0) == cv::Vec3b(17, 17, 17) &&
            pair->right.image.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30), "Wrong side or channel order");
    pair->left.image.setTo(cv::Scalar(0, 0, 0));
    pair = input.frameQueue().GetFrame();
    require(pair && pair->left.frame_id == 2 && pair->left.image.at<cv::Vec3b>(0, 0)[0] == 17,
            "Repeated frames share writable storage");
    require(input.captureNext(), "Repeat mode ended after consumption");
    pair = input.frameQueue().GetFrame();
    require(pair && pair->left.frame_id == 3 && pair->left.image.at<cv::Vec3b>(0, 0)[0] == 17,
            "Consumer modified source image");
    pair = input.frameQueue().GetFrame();
    require(!pair, "Empty result retained a previously returned pair");

    VideoInput once(config, 1); // 默认单次采集
    require(once.captureNext() && !once.captureNext() && once.queuedPairs() == 1,
            "Single-pair mode failed");
    require(once.frameQueue().GetFrame().has_value() && !once.frameQueue().GetFrame(),
            "Single-pair drain failed");

    expectError([&] { VideoInput bad(config, 0); }, "capacity");
    auto missingImage = config;
    missingImage.rightPath = (dir / "missing.png").string();
    expectError([&] { VideoInput bad(missingImage, 2); }, "input.right_path");
    auto emptyPath = config;
    emptyPath.leftPath.clear();
    expectError([&] { VideoInput bad(emptyPath, 2); }, "input.left_path");

    // 模式支持检查集中在数据源工厂，直接构造采集模块也会得到明确错误。
    InputParams camera;
    camera.mode = InputMode::Camera;
    expectError([&] { VideoInput bad(camera, 2); }, "camera acquisition backend");
    auto video = config;
    video.mode = InputMode::Video;
    expectError([&] { VideoInput bad(video, 2); }, "not implemented");
}

void testQueue()
{
    expectError([] { StereoFrameQueue bad(0); }, "capacity");
    expectError([] { VideoInput bad(nullptr, 1); }, "source");
    StereoFrameQueue queue(1);
    CameraFrame frame{cv::Mat(2, 2, CV_8UC3, cv::Scalar(4, 5, 6)), 7, 123};
    queue.push(frame, frame);
    CameraFrame empty;
    expectError([&] { queue.push(frame, empty); }, "CV_8UC3");
    require(queue.size() == 1, "Invalid pair changed the queue");
    frame.image.setTo(cv::Scalar(0, 0, 0));
    CameraFrame left, right;
    expectError([&] { queue.tryPop(left, left); }, "distinct");
    require(queue.tryPop(left, right) && left.frame_id == 7 && right.timestamp_ms == 123 &&
            left.image.at<cv::Vec3b>(0, 0) == cv::Vec3b(4, 5, 6), "Queue lost pixels or metadata");

    // 两个线程生产/消费：允许溢出跳号，但绝不能错配或倒序。
    StereoFrameQueue concurrent(4);
    std::atomic<bool> done{false};
    std::exception_ptr producerError;
    std::thread producer([&] {
        try {
            for (std::uint64_t i = 1; i <= 1000; ++i) {
                frame.frame_id = i;
                concurrent.push(frame, frame);
            }
        } catch (...) { producerError = std::current_exception(); }
        done.store(true);
    });
    std::uint64_t last = 0;
    bool valid = true;
    while (!done.load() || concurrent.size() > 0) {
        if (concurrent.tryPop(left, right)) {
            valid = valid && left.frame_id == right.frame_id && left.frame_id > last;
            last = left.frame_id;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();
    if (producerError) std::rethrow_exception(producerError);
    require(valid && last == 1000, "Concurrent queue mismatch or missing last pair");
}

CameraFrame makeFrame(std::int64_t timestamp, std::uint64_t id)
{
    return {cv::Mat(2, 3, CV_8UC3, cv::Scalar(7, 8, 9)), id, timestamp};
}

void testFrameMatching()
{
    StereoFrameQueue queue(4);
    require(!queue.CheckFrame() && !queue.GetFrame(), "Empty matching must return an invalid result");
    queue.pushLeft(makeFrame(100, 1));
    queue.pushLeft(makeFrame(200, 2));
    queue.pushRight(makeFrame(203, 30));
    require(!queue.CheckFrame(5) && queue.leftSize() == 1 && queue.rightSize() == 1,
            "Mismatch must discard only the older left frame");
    require(queue.CheckFrame(5) && queue.leftSize() == 1 && queue.rightSize() == 1,
            "Successful CheckFrame must not consume frames");
    auto pair = queue.GetFrame(5);
    require(pair && pair->left.frame_id == 2 && pair->right.frame_id == 30 &&
            pair->left.timestamp_ms == 200 && pair->right.timestamp_ms == 203,
            "Returned CameraFrames lost their metadata or the retained right frame");
    require(pair->left.image.at<cv::Vec3b>(0, 0) == cv::Vec3b(7, 8, 9) &&
            pair->right.image.at<cv::Vec3b>(0, 0) == cv::Vec3b(7, 8, 9), "Matching changed pixels");
    require(queue.leftSize() == 0 && queue.rightSize() == 0, "Matched frames were not consumed");

    // 镜像场景：一次检查只删除右侧的一张，不提前删除第二张旧帧。
    queue.pushLeft(makeFrame(300, 3));
    queue.pushRight(makeFrame(100, 10));
    queue.pushRight(makeFrame(200, 20));
    queue.pushRight(makeFrame(305, 40));
    require(!queue.CheckFrame(5) && queue.leftSize() == 1 && queue.rightSize() == 2,
            "CheckFrame removed more than one frame or removed the newer left frame");
    pair = queue.GetFrame(5);
    require(!pair && queue.leftSize() == 1 && queue.rightSize() == 1,
            "Mismatch must discard one older right frame and return an invalid result");
    pair = queue.GetFrame(5);
    require(pair && pair->left.frame_id == 3 && pair->right.frame_id == 40 &&
            pair->left.timestamp_ms == 300 && pair->right.timestamp_ms == 305,
            "Inclusive threshold boundary or original metadata is incorrect");

    // 多张过旧帧需要调用方多次尝试；每次只能删除一张并返回无效结果。
    queue.pushLeft(makeFrame(400, 4));
    queue.pushLeft(makeFrame(410, 5));
    queue.pushLeft(makeFrame(499, 6));
    queue.pushRight(makeFrame(500, 50));
    for (std::size_t remaining : {2U, 1U}) {
        require(!queue.GetFrame(1) && queue.leftSize() == remaining && queue.rightSize() == 1,
                "GetFrame must perform exactly one attempt per call");
    }
    pair = queue.GetFrame(1);
    require(pair && pair->left.frame_id == 6 && pair->right.frame_id == 50 &&
            pair->left.timestamp_ms == 499 && pair->right.timestamp_ms == 500,
            "Caller retries failed to return the matching CameraFrames");

    // 没有匹配结果时返回 nullopt，保留较新帧直至另一侧后续到达。
    queue.pushLeft(makeFrame(600, 7));
    queue.pushRight(makeFrame(700, 60));
    pair = queue.GetFrame(5);
    require(!pair && queue.leftSize() == 0 && queue.rightSize() == 1,
            "Failed matching returned stale frames or discarded the waiting newer frame");
    require(!queue.GetFrame() && !queue.CheckFrame(5) && queue.rightSize() == 1,
            "Empty side consumed the waiting frame or returned a valid result");
    queue.pushLeft(makeFrame(700, 8));
    pair = queue.GetFrame(0);
    require(pair && pair->left.frame_id == 8 && pair->right.frame_id == 60,
            "Later arrival could not match the retained frame");

    queue.push(makeFrame(800, 9), makeFrame(801, 70));
    expectError([&] { queue.CheckFrame(-1); }, "non-negative");
    expectError([&] { queue.GetFrame(-1); }, "non-negative");
    require(queue.leftSize() == 1 && queue.rightSize() == 1, "Invalid call mutated queues");
    require(!queue.GetFrame(0) && queue.leftSize() == 0 && queue.rightSize() == 1,
            "Zero threshold allowed unequal timestamps");

    // 单侧溢出只影响该侧，另一侧仍能与新到达帧匹配。
    StereoFrameQueue bounded(1);
    bounded.pushLeft(makeFrame(1000, 10));
    bounded.pushRight(makeFrame(900, 80));
    bounded.pushRight(makeFrame(1000, 90));
    require(bounded.leftSize() == 1 && bounded.rightSize() == 1, "Overflow changed queue bounds");
    pair = bounded.GetFrame(0);
    require(pair && pair->left.frame_id == 10 && pair->right.frame_id == 90,
            "Single-side overflow discarded the other camera's frame");

    // 对 int64 极值也不能因有符号减法溢出而误判为接近。
    StereoFrameQueue extreme(1);
    extreme.push(makeFrame(std::numeric_limits<std::int64_t>::min(), 1),
                 makeFrame(std::numeric_limits<std::int64_t>::max(), 2));
    require(!extreme.CheckFrame(std::numeric_limits<std::int64_t>::max()) &&
            extreme.leftSize() == 0 && extreme.rightSize() == 1, "Timestamp difference overflowed");
    extreme.pushLeft(makeFrame(std::numeric_limits<std::int64_t>::max(), 3));
    require(extreme.GetFrame(0).has_value(), "Extreme equal timestamps failed");
}
} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "Expected fixture output directory");
        const auto dir = std::filesystem::absolute(argv[1]);
        std::filesystem::create_directories(dir);
        testImages(dir);
        testSourceAndOwnership();
        testQueue();
        testFrameMatching();
        std::cout << "Video input tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
