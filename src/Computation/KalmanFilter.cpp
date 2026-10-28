#include "Computation/KalmanFilter.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
bool isFinite(const cv::Vec3d& X)
{
    return std::isfinite(X[0]) && std::isfinite(X[1]) && std::isfinite(X[2]);
}

bool isFinite(const cv::Matx33d& P)
{
    for (const double value : P.val) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

DartKalmanFilter::KalmanFilterResult evaluateResult(
    const cv::Vec3d& XkHat, const cv::Matx33d& Pk, TargetClasses target,
    std::size_t age, double nis, const cv::Vec3d& rkMean,
    const KalmanFilterParams& params, int& stableCount)
{
    DartKalmanFilter::KalmanFilterResult result;
    result.XkHat = cv::Point3d(XkHat);
    result.Pk = Pk;
    result.TargetName = target;
    result.age = age;
    result.nis = nis;
    const double horizontalSquared = XkHat[0] * XkHat[0] + XkHat[1] * XkHat[1];
    if (std::isfinite(horizontalSquared) && horizontalSquared > 0.0) {
        // theta=atan2(X,Y)，一阶误差传播：Var(theta)=J*Pk*J^T。
        const cv::Vec3d Jk(XkHat[1] / horizontalSquared, -XkHat[0] / horizontalSquared, 0.0);
        const double variance = Jk.dot(Pk * Jk);
        if (std::isfinite(variance) && variance >= 0.0)
            result.yawSigmaDeg = std::sqrt(variance) * 180.0 / CV_PI;
        result.yawResidualMeanDeg = Jk.dot(rkMean) * 180.0 / CV_PI;
    }
    // 初始化没有创新残差，不参与稳定计数；只有之后成功更新的观测才能确认稳定。
    const bool ready = age >= static_cast<std::size_t>(params.minAge) &&
        std::isfinite(result.yawSigmaDeg) && result.yawSigmaDeg <= params.maxYawErrorDeg &&
        std::isfinite(nis) && nis >= 0.0 && nis <= params.maxNis &&
        std::isfinite(result.yawResidualMeanDeg) &&
        std::abs(result.yawResidualMeanDeg) <= params.maxYawErrorDeg;
    if (!ready) stableCount = 0;
    else if (stableCount < params.stableFrames) ++stableCount;
    result.stable = ready && stableCount >= params.stableFrames;
    return result;
}
} // namespace

DartKalmanFilter::DartKalmanFilter(const KalmanFilterParams& params)
    : params_(params)
{
}

void DartKalmanFilter::Reset() noexcept
{
    initialized_ = false;
    XkHat_ = cv::Vec3d(0.0, 0.0, 0.0);
    Pk_ = cv::Matx33d::zeros();
    target_ = TargetClasses::UNKNOWN;
    age_ = 0;
    stableCount_ = 0;
    rkMean_ = cv::Vec3d(0.0, 0.0, 0.0);
}

void DartKalmanFilter::InvalidateStability() noexcept
{
    stableCount_ = 0;
}

DartKalmanFilter::KalmanFilterResult DartKalmanFilter::RunKalmanFilter(
    const LocationTransform::LocationTransformResult& result)
{
    const cv::Vec3d Zk(result.RealLocation.x, result.RealLocation.y, result.RealLocation.z);
    if (!isFinite(Zk)) {
        throw std::invalid_argument("Kalman measurement requires a finite position");
    }

    // 首帧或类别切换：直接用当前测量初始化，不把不同物理目标的历史混合。
    if (!initialized_ || result.TargetName != target_) {
        Reset();
        XkHat_ = Zk;
        Pk_ = params_.P0;
        target_ = result.TargetName;
        initialized_ = true;
        age_ = 1;
        return evaluateResult(XkHat_, Pk_, target_, age_,
            std::numeric_limits<double>::quiet_NaN(), rkMean_, params_, stableCount_);
    }

    // ----------------------------------------------------------------
    // step1  状态预测
    // ----------------------------------------------------------------
    // F=I，无控制输入，先验位置沿用上一帧后验位置。
    const cv::Vec3d XkMinus = XkHat_;

    // ----------------------------------------------------------------
    // step2  协方差预测
    // ----------------------------------------------------------------
    // 过程噪声Q按每次有效更新累加，再加入R得到创新协方差Sk。
    const cv::Matx33d PkMinus = Pk_ + params_.Q;
    const cv::Matx33d Sk = PkMinus + params_.R;
    if (!isFinite(PkMinus) || !isFinite(Sk)) {
        throw std::runtime_error("Kalman covariance prediction is nonfinite");
    }

    // ----------------------------------------------------------------
    // step3  计算卡尔曼增益
    // ----------------------------------------------------------------
    // Kk=PkMinus*Sk^(-1)。解Sk*Kk^T=PkMinus^T，避免显式求逆。
    cv::Matx33d KkTranspose;
    if (!cv::solve(Sk, PkMinus.t(), KkTranspose, cv::DECOMP_CHOLESKY)) {
        throw std::runtime_error("Kalman innovation covariance is not positive definite");
    }
    const cv::Matx33d Kk = KkTranspose.t();
    const cv::Vec3d rk = Zk - XkMinus;
    cv::Vec3d SkInverseRk;
    if (!cv::solve(Sk, rk, SkInverseRk, cv::DECOMP_CHOLESKY))
        throw std::runtime_error("Kalman innovation residual solve failed");
    const double nis = rk.dot(SkInverseRk);
    if (!isFinite(rk) || !std::isfinite(nis))
        throw std::runtime_error("Kalman innovation is nonfinite");

    // ----------------------------------------------------------------
    // step4  状态更新
    // ----------------------------------------------------------------
    // 用增益加权创新残差，修正先验位置。
    const cv::Vec3d XkHat = XkMinus + Kk * rk;

    // ----------------------------------------------------------------
    // step5  协方差更新
    // ----------------------------------------------------------------
    // 使用Joseph形式，并消除浮点运算产生的不对称误差。
    const cv::Matx33d IkMinusKk = cv::Matx33d::eye() - Kk;
    const cv::Matx33d PkJoseph = IkMinusKk * PkMinus * IkMinusKk.t()
        + Kk * params_.R * Kk.t();
    const cv::Matx33d Pk = PkJoseph * 0.5 + PkJoseph.t() * 0.5;
    if (!isFinite(XkHat) || !isFinite(Pk)) {
        throw std::runtime_error("Kalman update produced a nonfinite state or covariance");
    }

    // ----------------------------------------------------------------
    // step6  更新年龄与稳定性诊断
    // ----------------------------------------------------------------
    // 首次创新直接初始化均值，后续指数更新，避免从零启动时稀释持续偏移。
    const cv::Vec3d rkMean = age_ == 1 ? rk
        : params_.residualMeanAlpha * rk + (1.0 - params_.residualMeanAlpha) * rkMean_;
    const std::size_t age = age_ < std::numeric_limits<std::size_t>::max() ? age_ + 1 : age_;
    int stableCount = stableCount_;
    const auto filtered = evaluateResult(XkHat, Pk, target_, age, nis, rkMean, params_, stableCount);

    // 所有计算成功后才提交，异常测量或数值失败不会覆盖上一帧的有效状态。
    XkHat_ = XkHat;
    Pk_ = Pk;
    age_ = age;
    stableCount_ = stableCount;
    rkMean_ = rkMean;
    return filtered;
}
