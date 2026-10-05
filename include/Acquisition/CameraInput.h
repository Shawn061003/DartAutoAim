#pragma once

// 图像坐标约定：以全图左上角为原点，x向右、y向下。

#include <cstdint>
#include <opencv2/core.hpp>

// 所有采集、感知和解算模块共用；Unknown表示尚未标明来源，不能直接进入检测。
enum class CameraSide { Unknown, Left, Right };

// 采集模块提供的单帧数据，供感知与解算模块共享。
struct CameraFrame {
    cv::Mat image;                  // CV_8UC3 BGR全图，使用期间保持有效且不被修改
    std::uint64_t frame_id = 0;      // 相机帧号
    std::int64_t timestamp_ms = 0;   // 采集时间戳（毫秒），左右使用同一时间基准
    CameraSide camera_side = CameraSide::Unknown; // 来源相机，随帧传递
};
