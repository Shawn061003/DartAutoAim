#include "Computation/StereoscopicMeasurement.h"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>

std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult>
StereoscopicMeasurement::RunMeasurement(const GuideLightDetect::StereoDetectResult& result) const
{
    // 1. 根据左右检测中心生成通过几何检查的候选。
    const auto candidates = GetCandidate(result);
    // 2. 按XZ平面投影距离选择目标，并按BASE优先规则完成分类。
    return SelectFromCandidate(candidates);
}

std::vector<StereoscopicMeasurement::Candidate> StereoscopicMeasurement::GetCandidate(
    const GuideLightDetect::StereoDetectResult& result) const
{
    // 1. 仅按检测状态收集中心，保留原始索引；SUCCESS中心的有效性由检测模块保证。
    std::vector<std::size_t> leftIndices, rightIndices;
    std::vector<cv::Point2d> leftPixels, rightPixels;
    const auto collectCenters = [](const GuideLightDetect::DetectResult& detections,
                                   std::vector<std::size_t>& indices,
                                   std::vector<cv::Point2d>& pixels) {
        for (std::size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].status != GuideLightDetect::DetectStatus::SUCCESS) continue;
            indices.push_back(i);
            pixels.emplace_back(detections[i].CenterPoint);
        }
    };
    collectCenters(result.left, leftIndices, leftPixels);
    collectCenters(result.right, rightIndices, rightPixels);
    if (leftPixels.empty() || rightPixels.empty()) {
        return {};
    }

    // 2. 去畸变得到归一化坐标，用[I|0]、[R|T]在左相机坐标系中三角化。
    std::vector<cv::Point2d> leftNormalized, rightNormalized;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 1e-12);
    cv::undistortPoints(leftPixels, leftNormalized, leftCameraMatrix_, leftDistCoeffs_,
                        cv::noArray(), cv::noArray(), criteria);
    cv::undistortPoints(rightPixels, rightNormalized, rightCameraMatrix_, rightDistCoeffs_,
                        cv::noArray(), cv::noArray(), criteria);
    const cv::Matx33d leftK = leftCameraMatrix_;
    const cv::Matx33d rightK = rightCameraMatrix_;
    const cv::Matx33d rotation = leftToRightR_;
    const cv::Vec3d translation(leftToRightTmm_.at<double>(0, 0),
                                leftToRightTmm_.at<double>(1, 0),
                                leftToRightTmm_.at<double>(2, 0));
    const cv::Matx34d leftProjection(1, 0, 0, 0,
                                     0, 1, 0, 0,
                                     0, 0, 1, 0);
    cv::Matx34d rightProjection;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) rightProjection(row, col) = rotation(row, col);
        rightProjection(row, 3) = translation[row];
    }

    // 3. F作用于去畸变后的像素坐标，满足p_right^T * F * p_left = 0。
    const cv::Matx33d translationCross(0, -translation[2], translation[1],
                                      translation[2], 0, -translation[0],
                                      -translation[1], translation[0], 0);
    const cv::Matx33d fundamental = rightK.inv().t() * translationCross * rotation * leftK.inv();
    const auto toPixel = [](const cv::Matx33d& cameraMatrix, const cv::Vec3d& point) {
        const cv::Vec3d pixel = cameraMatrix * point;
        return cv::Point2d(pixel[0] / pixel[2], pixel[1] / pixel[2]);
    };

    // 初始几何筛选阈值，后续按实拍数据调整；不能直接用它们保证距离精度。
    constexpr double maxEpipolarErrorPx = 50.0;
    constexpr double maxReprojectionErrorPx = 25.0;
    constexpr double minRaySin = 1e-6; // 仅排除近乎平行的射线，不限定工作距离。
    constexpr double minHomogeneousW = 1e-12;
    std::vector<Candidate> candidates;
    candidates.reserve(leftPixels.size() * rightPixels.size());

    // 4. 枚举左右组合；这里只生成候选，一对一匹配留给SelectFromCandidate。
    for (std::size_t left = 0; left < leftNormalized.size(); ++left) {
        for (std::size_t right = 0; right < rightNormalized.size(); ++right) {
            const cv::Vec3d leftRay(leftNormalized[left].x, leftNormalized[left].y, 1.0);
            const cv::Vec3d rightRay(rightNormalized[right].x, rightNormalized[right].y, 1.0);
            const cv::Point2d leftUndistorted = toPixel(leftK, leftRay);
            const cv::Point2d rightUndistorted = toPixel(rightK, rightRay);
            const cv::Vec3d leftPoint(leftUndistorted.x, leftUndistorted.y, 1.0);
            const cv::Vec3d rightPoint(rightUndistorted.x, rightUndistorted.y, 1.0);
            const cv::Vec3d rightLine = fundamental * leftPoint;
            const cv::Vec3d leftLine = fundamental.t() * rightPoint;
            const double rightLineNorm = std::hypot(rightLine[0], rightLine[1]);
            const double leftLineNorm = std::hypot(leftLine[0], leftLine[1]);
            if (!(rightLineNorm > 0.0) || !(leftLineNorm > 0.0)) {
                continue;
            }
            const double epipolarResidual = std::abs(rightPoint.dot(rightLine));
            const double epipolarError = std::max(epipolarResidual / rightLineNorm,
                                                  epipolarResidual / leftLineNorm);
            if (!std::isfinite(epipolarError) || epipolarError > maxEpipolarErrorPx) {
                continue;
            }

            // 将两条射线转到同一坐标系，排除三角化退化的情况。
            const cv::Vec3d leftDirection = leftRay / cv::norm(leftRay);
            const cv::Vec3d rightDirection = rotation.t() * (rightRay / cv::norm(rightRay));
            const double raySin = cv::norm(leftDirection.cross(rightDirection));
            if (!std::isfinite(raySin) || raySin <= minRaySin) {
                continue;
            }

            cv::Mat homogeneous;
            cv::triangulatePoints(leftProjection, rightProjection,
                                  std::vector<cv::Point2d>{leftNormalized[left]},
                                  std::vector<cv::Point2d>{rightNormalized[right]}, homogeneous);
            const double w = homogeneous.at<double>(3, 0);
            if (!std::isfinite(w) || std::abs(w) <= minHomogeneousW) {
                continue;
            }
            const cv::Vec3d positionLeft(homogeneous.at<double>(0, 0) / w,
                                         homogeneous.at<double>(1, 0) / w,
                                         homogeneous.at<double>(2, 0) / w);
            const cv::Vec3d positionRight = rotation * positionLeft + translation;
            if (!std::isfinite(cv::norm(positionLeft)) || !std::isfinite(cv::norm(positionRight))
                || positionLeft[2] <= 0.0 || positionRight[2] <= 0.0) {
                continue;
            }

            // 5. 用去畸变像素检查重投影误差；输出位置转为右相机坐标，单位毫米。
            const double leftError = cv::norm(toPixel(leftK, positionLeft) - leftUndistorted);
            const double rightError = cv::norm(toPixel(rightK, positionRight) - rightUndistorted);
            if (!std::isfinite(leftError) || !std::isfinite(rightError)) {
                continue;
            }
            const double reprojectionError = std::max(leftError, rightError);
            if (reprojectionError > maxReprojectionErrorPx) {
                continue;
            }

            Candidate candidate;
            candidate.leftIndex = leftIndices[left];
            candidate.rightIndex = rightIndices[right];
            candidate.measurement.leftPixel = result.left[candidate.leftIndex].CenterPoint;
            candidate.measurement.rightPixel = result.right[candidate.rightIndex].CenterPoint;
            candidate.measurement.RealLocation = cv::Point3d(positionRight);
            candidate.measurement.TargetName = TargetClasses::UNKNOWN;
            candidate.measurement.leftReprojectionErrorPx = leftError;
            candidate.measurement.rightReprojectionErrorPx = rightError;
            candidate.epipolarErrorPx = epipolarError;
            candidate.reprojectionErrorPx = reprojectionError;
            candidates.push_back(candidate);
        }
    }
    return candidates;
}

std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult>
StereoscopicMeasurement::SelectFromCandidate(const std::vector<Candidate>& candidates) const
{
    // 1. 使用右相机XZ平面投影距离sqrt(X*X+Z*Z)，单位毫米；区间包含两端。
    constexpr double baseMinDistance = 20500.0;
    constexpr double baseMaxDistance = 28500.0;
    constexpr double baseReferenceDistance = 24400.0;
    constexpr double outpostMinDistance = 12000.0;
    constexpr double outpostMaxDistance = 18000.0;
    constexpr double outpostReferenceDistance = 15200.0;

    // 2. 优先确定BASE，选取区间内最接近参考位置的候选；同误差保留先出现者。
    const Candidate* base = nullptr;
    double baseError = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        const auto& position = candidate.measurement.RealLocation;
        const double distance = std::hypot(position.x, position.z);
        if (!(distance >= baseMinDistance && distance <= baseMaxDistance)) continue;
        const double error = std::abs(distance - baseReferenceDistance);
        if (error < baseError) {
            base = &candidate;
            baseError = error;
        }
    }

    // 3. 排除复用BASE左检测或右检测的候选，再选最接近参考位置的OUTPOST。
    const Candidate* outpost = nullptr;
    double outpostError = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        if (base && (candidate.leftIndex == base->leftIndex
                     || candidate.rightIndex == base->rightIndex)) continue;
        const auto& position = candidate.measurement.RealLocation;
        const double distance = std::hypot(position.x, position.z);
        if (!(distance >= outpostMinDistance && distance <= outpostMaxDistance)) continue;
        const double error = std::abs(distance - outpostReferenceDistance);
        if (error < outpostError) {
            outpost = &candidate;
            outpostError = error;
        }
    }

    // 4. 只返回存在的目标，依次添加BASE、OUTPOST；复制后打标签，不修改输入候选。
    std::vector<StereoscopicMeasurementResult> selected;
    selected.reserve(2);
    if (base) {
        selected.push_back(base->measurement);
        selected.back().TargetName = TargetClasses::BASE;
        selected.back().reprojectionErrorPx = base->reprojectionErrorPx;
    }
    if (outpost) {
        selected.push_back(outpost->measurement);
        selected.back().TargetName = TargetClasses::OUTPOST;
        selected.back().reprojectionErrorPx = outpost->reprojectionErrorPx;
    }
    return selected;
}
