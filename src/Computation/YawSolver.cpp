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
