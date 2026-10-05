#include "Acquisition/AcquireFrame.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {
void validateFrame(const CameraFrame& frame, CameraSide side)
{
    if (frame.image.empty() || frame.image.dims != 2 || frame.image.type() != CV_8UC3)
        throw std::invalid_argument("Frame queue requires non-empty CV_8UC3 BGR images");
    if (frame.camera_side != CameraSide::Unknown && frame.camera_side != side)
        throw std::invalid_argument("Frame camera_side conflicts with the queue input side");
}

void validateThreshold(std::int64_t maxTimestampDiffMs)
{
    if (maxTimestampDiffMs < 0)
        throw std::invalid_argument("maxTimestampDiffMs must be non-negative");
}
} // namespace

StereoFrameQueue::StereoFrameQueue(std::size_t capacity) : capacity_(capacity)
{
    if (capacity == 0) throw std::invalid_argument("Frame queue capacity must be positive");
}

void StereoFrameQueue::push(const CameraFrame& left, const CameraFrame& right)
{
    validateFrame(left, CameraSide::Left);
    validateFrame(right, CameraSide::Right);
    CameraFrame leftCopy = left;
    CameraFrame rightCopy = right;
    leftCopy.camera_side = CameraSide::Left;
    rightCopy.camera_side = CameraSide::Right;
    leftCopy.image = left.image.clone();
    rightCopy.image = right.image.clone();
    std::lock_guard<std::mutex> lock(mutex_);
    leftFrames_.push_back(std::move(leftCopy));
    try {
        rightFrames_.push_back(std::move(rightCopy));
    } catch (...) {
        // 第二侧入队失败时回滚，避免只发布半对数据。
        leftFrames_.pop_back();
        throw;
    }
    if (leftFrames_.size() > capacity_) leftFrames_.pop_front();
    if (rightFrames_.size() > capacity_) rightFrames_.pop_front();
}

void StereoFrameQueue::pushSingle(std::deque<CameraFrame>& frames, const CameraFrame& frame, CameraSide side)
{
    validateFrame(frame, side);
    CameraFrame copy = frame;
    copy.camera_side = side;
    copy.image = frame.image.clone();
    std::lock_guard<std::mutex> lock(mutex_);
    frames.push_back(std::move(copy));
    if (frames.size() > capacity_) frames.pop_front();
}

void StereoFrameQueue::pushLeft(const CameraFrame& frame) { pushSingle(leftFrames_, frame, CameraSide::Left); }

void StereoFrameQueue::pushRight(const CameraFrame& frame) { pushSingle(rightFrames_, frame, CameraSide::Right); }

bool StereoFrameQueue::checkFrameLocked(std::int64_t maxTimestampDiffMs)
{
    if (leftFrames_.empty() || rightFrames_.empty()) return false;
    const auto leftTime = leftFrames_.front().timestamp_ms;
    const auto rightTime = rightFrames_.front().timestamp_ms;
    // 先确定大小，再用无符号减法，避免 int64 极值相减或 abs(INT64_MIN) 溢出。
    const auto difference = leftTime >= rightTime
        ? static_cast<std::uint64_t>(leftTime) - static_cast<std::uint64_t>(rightTime)
        : static_cast<std::uint64_t>(rightTime) - static_cast<std::uint64_t>(leftTime);
    if (difference <= static_cast<std::uint64_t>(maxTimestampDiffMs)) return true;

    if (leftTime < rightTime) leftFrames_.pop_front();
    else rightFrames_.pop_front();
    return false;
}

bool StereoFrameQueue::CheckFrame(std::int64_t maxTimestampDiffMs)
{
    validateThreshold(maxTimestampDiffMs);
    std::lock_guard<std::mutex> lock(mutex_);
    return checkFrameLocked(maxTimestampDiffMs);
}

std::optional<StereoFrame> StereoFrameQueue::GetFrame(std::int64_t maxTimestampDiffMs)
{
    validateThreshold(maxTimestampDiffMs);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!checkFrameLocked(maxTimestampDiffMs)) return std::nullopt;
    StereoFrame result{std::move(leftFrames_.front()), std::move(rightFrames_.front())};
    leftFrames_.pop_front();
    rightFrames_.pop_front();
    return result;
}

bool StereoFrameQueue::tryPop(CameraFrame& left, CameraFrame& right)
{
    if (&left == &right) throw std::invalid_argument("Left and right outputs must be distinct");
    auto result = GetFrame();
    if (!result) return false;
    left = std::move(result->left);
    right = std::move(result->right);
    return true;
}

std::size_t StereoFrameQueue::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::min(leftFrames_.size(), rightFrames_.size());
}

std::size_t StereoFrameQueue::leftSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return leftFrames_.size();
}

std::size_t StereoFrameQueue::rightSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return rightFrames_.size();
}
