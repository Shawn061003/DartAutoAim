#include "Visualization/VisualizeAdapter.h"

namespace visualization {
namespace {
// 检测模块的结构体仅在适配层转换，底层Visualize.h/Visualize.cpp保持独立。
YOLOInferenceVisualize::FrameVisualize makeYOLOVisualize(
    const CameraFrame& frame, const YOLOInference::YOLOResults& results)
{
    YOLOInferenceVisualize::FrameVisualize data{
        {frame.image, frame.frame_id, frame.timestamp_ms}, {}};
    data.targets.reserve(results.size());
    for (const auto& result : results)
        data.targets.push_back({result.roi, result.conf});
    return data;
}

GuideLightDetectVisualize::FrameVisualize makeGuideLightVisualize(
    const CameraFrame& frame, const GuideLightDetect::DetectResult& results)
{
    GuideLightDetectVisualize::FrameVisualize data{
        {frame.image, frame.frame_id, frame.timestamp_ms}, {}};
    data.targets.reserve(results.size());
    for (const auto& result : results)
        data.targets.push_back({result.roi,
            result.status == GuideLightDetect::DetectStatus::SUCCESS,
            result.CenterPoint, result.contours});
    return data;
}

} // namespace

StereoscopicMeasurementVisualize::FrameVisualize makeLocationVisualize(
    const std::vector<LocationTransform::LocationTransformResult>& locations)
{
    StereoscopicMeasurementVisualize::FrameVisualize data;
    data.targets.reserve(locations.size());
    using ClassVisualize = StereoscopicMeasurementVisualize::TargetClassVisualize;
    for (const auto& result : locations) {
        auto category = ClassVisualize::UNKNOWN;
        if (result.TargetName == TargetClasses::BASE) category = ClassVisualize::BASE;
        else if (result.TargetName == TargetClasses::OUTPOST) category = ClassVisualize::OUTPOST;
        data.targets.push_back({result.RealLocation * 0.001, category});
    }
    return data;
}

int showDetection(const StereoFrame& frame,
                  const YOLOInference::StereoYOLOResult& rois,
                  const GuideLightDetect::StereoDetectResult& detections,
                  int delayMs)
{
    return DetectionVisualize::Show(makeYOLOVisualize(frame.left, rois.left),
                                   makeYOLOVisualize(frame.right, rois.right),
                                   makeGuideLightVisualize(frame.left, detections.left),
                                   makeGuideLightVisualize(frame.right, detections.right), delayMs);
}

int showLocations(const std::vector<LocationTransform::LocationTransformResult>& locations,
                  int delayMs)
{
    return StereoscopicMeasurementVisualize::Show(makeLocationVisualize(locations), delayMs);
}

} // namespace visualization
