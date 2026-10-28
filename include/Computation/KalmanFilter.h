#pragma once

#include "Computation/LocationTransform.h"
#include "common/DartConfig.h"
#include <cstddef>
#include <limits>
#include <opencv2/core.hpp>

// 三维静态位置模型：F=I、H=I，无控制输入。
// 输入、状态和输出均位于当前镖口坐标系，位置单位mm，协方差单位mm²。
class DartKalmanFilter {
public:
    struct KalmanFilterResult
    {
        cv::Point3d XkHat{
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()}; // 后验状态 x̂(k|k)。
        cv::Matx33d Pk = cv::Matx33d::zeros(); // 后验协方差 P(k|k)。
        TargetClasses TargetName = TargetClasses::UNKNOWN;
        std::size_t age = 0; // 成功融合的观测数，初始化首帧为1；缺失帧不增加。
        double yawSigmaDeg = std::numeric_limits<double>::infinity();
        double nis = std::numeric_limits<double>::quiet_NaN(); // r(k)^T*S(k)^(-1)*r(k)。
        double yawResidualMeanDeg = std::numeric_limits<double>::quiet_NaN();
        bool stable = false; // 年龄、yaw精度和残差条件持续满足后才为true。
    };

    /// @brief 引用DartConfig中的KF配置，不保存配置副本。
    /// @pre params须完成数值校验，且生命周期长于滤波器；使用期间不得修改或重新加载。
    DartKalmanFilter(const KalmanFilterParams& params);

    /// @brief 对完成坐标转换后的单一目标递推估计，不修改输入。
    /// @param result 当前测量Zk；位置须有限，类别须为BASE或OUTPOST。
    /// @return 后验位置、协方差、年龄及稳定性诊断；yaw精度阈值指标准差，不是硬误差界。
    /// @note 首次有效观测用Zk初始化XkHat、用P0初始化Pk；类别切换时重新初始化。
    ///       仅维护当前一个目标；镖架运动或长时间丢失后，调用方须先Reset。
    KalmanFilterResult RunKalmanFilter(const LocationTransform::LocationTransformResult& result);

    /// @brief 清除初始化状态和目标类别，下一次有效观测重新初始化。
    void Reset() noexcept;

    /// @brief 缺失当前目标时撤销连续稳定计数，保留短暂丢失期间的KF状态和年龄。
    void InvalidateStability() noexcept;

private:
    const KalmanFilterParams& params_; // Q、R、P0来自统一配置，只保存只读引用。
    bool initialized_ = false;
    cv::Vec3d XkHat_{0.0, 0.0, 0.0};       // 上一次后验状态 x̂(k|k)。
    cv::Matx33d Pk_ = cv::Matx33d::zeros(); // 上一次后验协方差 P(k|k)。
    TargetClasses target_ = TargetClasses::UNKNOWN;
    std::size_t age_ = 0;
    int stableCount_ = 0;
    cv::Vec3d rkMean_{0.0, 0.0, 0.0}; // 创新残差的指数均值，单位mm。
    // 更新函数内局部量：Zk为测量，XkMinus/PkMinus为先验，Kk为卡尔曼增益。
};
