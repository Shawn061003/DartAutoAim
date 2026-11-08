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
    // 补偿模型接收未补偿的镖口测量XY，输入输出均为mm。
    for (const auto& result : results) {
        const cv::Vec3d point(result.RealLocation.x, result.RealLocation.y, result.RealLocation.z);
        // 1. 先按配置的R/T，将右相机光心坐标转换到当前镖口坐标。
        const cv::Vec3d dart = rotation * point + translation;
        // 2. 直接在当前镖口坐标方向补偿，不使用标定采样时的固定yaw。
        const double x = dart[0];
        const double y = dart[1];
        double correctedX, correctedY;
        if(aim_base) {
            // 3. 基地远段模型使用15.8°姿态下关闭旧补偿的数据拟合。
            // 地面真值先减去镖口原点偏移(145, -60) mm，再旋转到镖口系。
            correctedX = 1.0733 * x + 570.0;
            correctedY = y + 1127.8 + 0.1232 * x;
        }
        else {
            // 前哨站使用表一近段模型，来源为outputs/compensation_muzzle_20261104/muzzle_xy_models.json。
            correctedX = 0.9967931135150897 * x - 0.0041306312564936779 * y + 76.847208117170993;
            correctedY = 0.037181856240888093 * x + 1.0787931524777847 * y - 1319.4764685108794;
        }
        // 4. 输出当前镖口方向下的补偿结果；Z保留R/T转换结果。
        transformed.push_back({{correctedX, correctedY, dart[2]}, result.TargetName});
    }
    return transformed;
}
