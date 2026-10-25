#include "Computation/YawSolver.h"

#include <cmath>

std::optional<double> YawSolver::RunSolve(
    const std::vector<LocationTransform::LocationTransformResult>& locations,
    bool aim_base) const
{
    const auto targetName = aim_base ? TargetClasses::BASE : TargetClasses::OUTPOST;
    for (const auto& location : locations) {
        if (location.TargetName != targetName) continue;

        const auto& point = location.RealLocation;
        // 无效坐标或水平投影为零时，无法确定目标的水平方向。
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z) || (point.x == 0.0 && point.y == 0.0)) {
            return std::nullopt;
        }

        // 相对+Y轴计算：X向右，因此atan2(X, Y)满足右正左负。
        return std::atan2(point.x, point.y);
    }

    // 所选类别缺失时，不切换目标，也不沿用上一帧结果。
    return std::nullopt;
}

double YawSolver::RunSearch(const GuideLightDetect::DetectResult& detections) const
{
    constexpr double targetUPx = 2552.0;
    // 初始15.8度，目标14.7～16.9度，最大偏差1.1度。
    constexpr double searchStepRad = 1.1 * 3.14159265358979323846 / 180.0;
    for (const auto& detection : detections) {
        if (detection.status != GuideLightDetect::DetectStatus::SUCCESS ||
            !std::isfinite(detection.CenterPoint.x)) continue;

        // 严格按搜索方向约定：u小于阈值向右，否则向左。
        return detection.CenterPoint.x < targetUPx ? searchStepRad : -searchStepRad;
    }

    return 0.0;
}
