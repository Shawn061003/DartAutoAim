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
    // 模型来源：outputs/compensation_muzzle_20261104/muzzle_xy_models.json。
    // 真值已按表三角度逆旋转到镖口系重新拟合，输入为未调零的镖口测量XY。
    // 输入输出均为mm；运行时不再附加表三角度或current_angle旋转。
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
            correctedX = 1.0482269747531723 * x - 0.0019828467671149565 * y + 63.866907696152566;
            correctedY = 0.17311053402080517 * x + 1.1498485168078993 * y - 2681.7256544822112;
        }
        else {
            // 前哨站使用表一近段模型。
            correctedX = 0.9967931135150897 * x - 0.0041306312564936779 * y + 76.847208117170993;
            correctedY = 0.037181856240888093 * x + 1.0787931524777847 * y - 1319.4764685108794;
        }
        // 4. 输出当前镖口方向下的补偿结果；Z保留R/T转换结果。
        transformed.push_back({{correctedX, correctedY, dart[2]}, result.TargetName});
    }
    return transformed;
}
