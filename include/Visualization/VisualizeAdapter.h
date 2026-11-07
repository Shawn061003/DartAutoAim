#pragma once

#include "Acquisition/AcquireFrame.h"
#include "Computation/LocationTransform.h"
#include "Perception/GuideLightDetect.h"
#include "Perception/YOLOinfer.h"
#include "Visualization/Visualize.h"
#include <vector>

// 连接业务结果与独立的可视化类型；只负责转换和调用，不在此层绘图或处理瞄准状态。
namespace visualization {

// 组合左右YOLO和引导灯结果，返回原显示接口的键值，由主流程处理按键。
int showDetection(const StereoFrame& frame,
                  const YOLOInference::StereoYOLOResult& rois,
                  const GuideLightDetect::StereoDetectResult& detections,
                  int delayMs = 1);

// 镖口坐标由毫米转为米，角度由弧度转为度；空位置输入仍保留当前角度。
StereoscopicMeasurementVisualize::FrameVisualize makeLocationVisualize(
    const std::vector<LocationTransform::LocationTransformResult>& locations, double currentAngleRad);

// delayMs原样传给显示接口：0等待按键，-1只提交位置画面。
int showLocations(const std::vector<LocationTransform::LocationTransformResult>& locations,
                  double currentAngleRad, int delayMs = 1);

} // namespace visualization
