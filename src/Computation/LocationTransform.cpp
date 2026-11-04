#include "Computation/LocationTransform.h"

std::vector<LocationTransform::LocationTransformResult> LocationTransform::RunTransform(
    const std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult>& results, bool aim_base) const
{
    std::vector<LocationTransformResult> transformed;
    transformed.reserve(results.size());
    const cv::Matx33d rotation = rightCameraToDartR_;
    const cv::Vec3d translation(rightCameraToDartTmm_.at<double>(0, 0),
                              rightCameraToDartTmm_.at<double>(1, 0),
                              rightCameraToDartTmm_.at<double>(2, 0));
    // 表三yaw仅用于离线拟合数据调零；镖架运行时会转动，不附加固定yaw旋转。
    // 输入输出均为mm，原cm模型的常数项已乘10。
    for (const auto& result : results) {
        const cv::Vec3d point(result.RealLocation.x, result.RealLocation.y, result.RealLocation.z);
        // 1. 先按配置的R/T，将右相机光心坐标转换到当前镖口坐标。
        const cv::Vec3d dart = rotation * point + translation;
        // 2. 直接在当前镖口坐标方向补偿，不使用标定采样时的固定yaw。
        const double x = dart[0];
        const double y = dart[1];
        double correctedX, correctedY;
        if(aim_base) {
            // 3. 基地使用表二远段模型；结果是补偿坐标，不是增量。
            correctedX = 1.0498717487281446 * x - 0.001027267589373153 * y + 38.23319457140655;
            correctedY = 0.174066113198185 * x + 1.1482037428326253 * y - 2682.213580347702;
        }
        else {
            // 前哨站使用表一近段模型。
            correctedX = 0.9971164791510051 * x - 0.003349964609264049 * y + 64.23270187708702;
            correctedY = 0.037962522888115774 * x + 1.0784697868419688 * y - 1320.1506748712245;
        }
        // 4. 输出当前镖口方向下的补偿结果；Z保留R/T转换结果。
        transformed.push_back({{correctedX, correctedY, dart[2]}, result.TargetName});
    }
    return transformed;
}
