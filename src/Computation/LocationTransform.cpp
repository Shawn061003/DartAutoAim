#include "Computation/LocationTransform.h"

std::vector<LocationTransform::LocationTransformResult> LocationTransform::RunTransform(
    const std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult>& results) const
{
    std::vector<LocationTransformResult> transformed;
    transformed.reserve(results.size());
    const cv::Matx33d rotation = rightCameraToDartR_;
    const cv::Vec3d translation(rightCameraToDartTmm_.at<double>(0, 0),
                              rightCameraToDartTmm_.at<double>(1, 0),
                              rightCameraToDartTmm_.at<double>(2, 0));
    for (const auto& result : results) {
        const cv::Vec3d point(result.RealLocation.x, result.RealLocation.y, result.RealLocation.z);
        transformed.push_back({cv::Point3d(rotation * point + translation), result.TargetName, result.stable});
    }
    return transformed;
}
