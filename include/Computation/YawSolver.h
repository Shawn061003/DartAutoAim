#pragma once

#include "Perception/GuideLightDetect.h"
#include "Computation/LocationTransform.h"
#include <optional>
#include <vector>

// 镖口坐标：轨道终点为原点，X向右、Y向前、Z向上。
// yaw为目标连线在XY平面的投影相对当前+Y轴的有符号角度，单位rad，右正左负。
class YawSolver {
public:
    /// @brief 按瞄准模式选择目标，解算水平方向的yaw角度差。
    /// @param locations 当前帧的镖口坐标转换结果，不修改输入。
    /// @param aim_base true选择BASE，false选择OUTPOST；按TargetName查找，不依赖下标。
    /// @pre 同一类别至多一个目标，与LocationTransform的上游输出约定一致。
    /// @return 成功返回yaw弧度值；空输入、所选目标缺失、所选坐标含非有限值，
    ///         或所选目标X和Y同时为零时返回std::nullopt。
    /// @note yaw = atan2(X, Y)，范围[-pi, pi]；正前方为0，右前方为正，左前方为负。
    ///       Z不参与角度计算；不会因所选目标缺失而切换到另一类别或沿用上一帧结果。
    ///       调用方仅在返回值有效时交给通信模块下发，并统一通信协议的正负方向。
    std::optional<double> RunSolve(
        const std::vector<LocationTransform::LocationTransformResult>& locations,
        bool aim_base) const;

    /// @brief 右相机未检出目标时，根据左图目标位置返回固定搜索步长。
    /// @param detections 左相机当前帧检测结果，按输入顺序使用首个SUCCESS且u有限的目标。
    /// @return u<2552时返回+1.1度，否则返回-1.1度；结果单位rad，右正左负。
    ///         没有有效目标时返回0，调用方跳过本帧搜索命令。
    /// @note 阈值2552像素近似对应当前标定下镖口X=0、Y=24.5米的目标投影。
    ///       步长依据初始朝向15.8度、目标方位14.7～16.9度时最大1.1度偏差选取。
    double RunSearch(const GuideLightDetect::DetectResult& detections) const;
};
