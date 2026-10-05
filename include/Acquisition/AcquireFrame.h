#pragma once

#include "Acquisition/CameraInput.h"

#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

// 一次成功匹配的结果；两侧都保留各自的图像、帧号和时间戳。
struct StereoFrame {
    CameraFrame left;
    CameraFrame right;
};

// 相机、视频、图片共用的缓存与取帧接口，下游只需依赖本文件。
// 左右相机独立缓存，每侧按 timestamp_ms 非递减顺序入队，并使用相同时间基准。
// 所有队列操作线程安全；单侧容量为 capacity，溢出时只丢弃该侧最旧帧。
class StereoFrameQueue {
public:
    explicit StereoFrameQueue(std::size_t capacity);

    // 图片/视频一次提交左右帧；相机异步采集可分别调用 pushLeft/pushRight。
    // 入队拷贝图像数据，隔离相机 SDK 或视频解码器复用的缓冲区。
    // 按入队侧补全Unknown；显式侧别与入口冲突时抛异常，不修改输入或队列。
    void push(const CameraFrame& left, const CameraFrame& right);
    void pushLeft(const CameraFrame& frame);
    void pushRight(const CameraFrame& frame);

    // 下游主入口：每次仅检查一次队首，时间差 <= maxTimestampDiffMs 时匹配。
    // 不匹配时只删除较旧的一帧并返回 std::nullopt，重试循环由调用方负责。
    // 成功时两帧出队，保留原始图像、帧号和时间戳，不要求左右帧号相等。
    // 任一侧缺帧也返回 std::nullopt（无效结果、不阻塞），保留另一侧帧。
    // 有效结果的 left/right 均为 CameraFrame，出队后图像保持有效，下游按只读使用。
    // 阈值单位为毫秒，必须 >= 0；默认 5 ms 可按采集同步精度调整。
    std::optional<StereoFrame> GetFrame(std::int64_t maxTimestampDiffMs = 5);

    // 只检查当前两侧队首：匹配返回 true，不出队；不匹配仅删除较旧的一帧。
    // 任一侧为空时返回 false，不删除另一侧。调用后队列可能被其他线程更新，
    // 消费帧应使用 GetFrame，其检查和出队在同一锁内完成。
    bool CheckFrame(std::int64_t maxTimestampDiffMs = 5);

    // 兼容已有调用：使用默认阈值调用 GetFrame，无结果时返回 false 且输出不变。
    bool tryPop(CameraFrame& left, CameraFrame& right);
    // 两侧帧数的较小值，表示待检查的候选对数，不保证已通过时间戳匹配。
    std::size_t size() const;
    std::size_t leftSize() const;
    std::size_t rightSize() const;

private:
    // 调用方已持锁且已验证阈值，避免 GetFrame 中重复加锁。
    bool checkFrameLocked(std::int64_t maxTimestampDiffMs);
    void pushSingle(std::deque<CameraFrame>& frames, const CameraFrame& frame, CameraSide side);

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<CameraFrame> leftFrames_;
    std::deque<CameraFrame> rightFrames_;
};
