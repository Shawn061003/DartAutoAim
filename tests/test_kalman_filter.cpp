#include "Computation/KalmanFilter.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

KalmanFilterParams parameters()
{
    KalmanFilterParams params;
    params.Q = cv::Matx33d::eye();
    params.R = cv::Matx33d::eye() * 100.0;
    params.P0 = params.R;
    return params;
}
} // namespace

int main()
{
    try {
        auto params = parameters();
        params.stableFrames = 1;
        LocationTransform::LocationTransformResult z{{0, 25000, 0}, TargetClasses::BASE};
        DartKalmanFilter ageFilter(params);
        for (std::size_t age = 1; age <= 10; ++age) {
            const auto result = ageFilter.RunKalmanFilter(z);
            require(result.age == age, "Age does not count valid observations from one");
            require(result.stable == (age == 10), "Age >= 10 boundary is incorrect");
            require(result.yawSigmaDeg <= 0.05, "Expected yaw uncertainty below threshold");
        }

        params = parameters();
        DartKalmanFilter persistenceFilter(params);
        DartKalmanFilter::KalmanFilterResult result;
        for (int age = 1; age <= 14; ++age) {
            result = persistenceFilter.RunKalmanFilter(z);
            require(result.stable == (age == 14), "Five consecutive qualifying updates were not required");
        }
        persistenceFilter.InvalidateStability();
        for (int count = 1; count <= 5; ++count) {
            result = persistenceFilter.RunKalmanFilter(z);
            require(result.age == static_cast<std::size_t>(14 + count), "Short loss reset age");
            require(result.stable == (count == 5), "Loss retained previous stable streak");
        }
        persistenceFilter.Reset();
        result = persistenceFilter.RunKalmanFilter(z);
        require(result.age == 1 && !result.stable, "Reset retained age or stability");
        z.TargetName = TargetClasses::OUTPOST;
        result = persistenceFilter.RunKalmanFilter(z);
        require(result.age == 1 && !result.stable, "Target switch retained age or stability");

        // 稳定观测也不能绕过精度门限。
        auto uncertain = parameters();
        uncertain.R = cv::Matx33d::eye() * 1e10;
        uncertain.P0 = uncertain.R;
        DartKalmanFilter uncertaintyFilter(uncertain);
        for (int i = 0; i < 30; ++i) result = uncertaintyFilter.RunKalmanFilter(z);
        require(result.yawSigmaDeg > 0.05 && !result.stable, "Large uncertainty passed gate");

        // 独立核对J*P*J^T及度/弧度转换，包括非零交叉协方差。
        auto coupled = parameters();
        coupled.P0 = {100, 30, 0, 30, 200, 0, 0, 0, 100};
        DartKalmanFilter propagationFilter(coupled);
        z.RealLocation = {3000, 24000, 500};
        result = propagationFilter.RunKalmanFilter(z);
        const double d = 3000.0*3000.0 + 24000.0*24000.0;
        const double expectedVariance = (24000.0*24000.0*100 + 3000.0*3000.0*200
            - 2.0*3000.0*24000.0*30) / (d*d);
        require(std::abs(result.yawSigmaDeg - std::sqrt(expectedVariance)*180.0/CV_PI) < 1e-12,
                "Yaw covariance propagation or angle units are incorrect");

        auto spikeParams = parameters();
        DartKalmanFilter spikeFilter(spikeParams);
        z.RealLocation = {0, 25000, 0};
        for (int i = 0; i < 14; ++i) result = spikeFilter.RunKalmanFilter(z);
        require(result.stable, "Spike fixture did not stabilize");
        z.RealLocation.x = 1000;
        result = spikeFilter.RunKalmanFilter(z);
        require(result.nis > spikeParams.maxNis && !result.stable, "Innovation outlier passed gate");

        // 即使单次NIS较小且P较小，持续偏移的残差均值也应阻止发送。
        auto biasParams = parameters();
        biasParams.Q = cv::Matx33d::zeros();
        biasParams.R = cv::Matx33d::eye() * 1e6;
        biasParams.P0 = cv::Matx33d::eye() * 1e-6;
        DartKalmanFilter biasFilter(biasParams);
        z.RealLocation.x = 0;
        for (int i = 0; i < 14; ++i) result = biasFilter.RunKalmanFilter(z);
        require(result.stable, "Bias fixture did not stabilize");
        for (int i = 0; i < 10; ++i) {
            z.RealLocation.x += 100;
            result = biasFilter.RunKalmanFilter(z);
        }
        require(result.nis < biasParams.maxNis && result.yawSigmaDeg < 0.05 &&
                std::abs(result.yawResidualMeanDeg) > 0.05 && !result.stable,
                "Persistent directional residual passed gate");

        z.RealLocation = {0, 0, 100};
        propagationFilter.Reset();
        for (int i = 0; i < 20; ++i) result = propagationFilter.RunKalmanFilter(z);
        require(!std::isfinite(result.yawSigmaDeg) && !result.stable, "Undefined yaw passed gate");
        std::cout << "KF age, precision, innovation, persistence, resets and yaw propagation: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
