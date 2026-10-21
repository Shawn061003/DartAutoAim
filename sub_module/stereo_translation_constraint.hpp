#pragma once
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stereo_calibration {
// X_right = R * X_left + T. T is expressed in the RIGHT optical frame: right/down/forward, mm.
struct TranslationBounds {
    cv::Vec3d center, halfWidth{60.,10.,80.};
    explicit TranslationBounds(double initialX) : center(initialX<0 ? -510. : 510.,0.,0.) {
        if (!std::isfinite(initialX) || std::abs(initialX)<1e-9)
            throw std::runtime_error("Initial Tx has no reliable sign for the translation bounds");
    }
    cv::Vec3d lower() const { return center-halfWidth; }
    cv::Vec3d upper() const { return center+halfWidth; }
    cv::Mat decode(const cv::Mat& parameters, cv::Vec3d* derivative=nullptr) const {
        cv::Mat translation(3,1,CV_64F);
        for (int axis=0;axis<3;++axis) {
            const double fraction=std::tanh(parameters.at<double>(3+axis));
            translation.at<double>(axis)=center[axis]+halfWidth[axis]*fraction;
            if (derivative) (*derivative)[axis]=halfWidth[axis]*(1.-fraction*fraction);
        }
        return translation;
    }
    bool contains(const cv::Mat& translation) const {
        if (translation.total()!=3 || translation.type()!=CV_64FC1 || !cv::checkRange(translation)) return false;
        const auto lo=lower(),hi=upper();
        for (int axis=0;axis<3;++axis)
            if (translation.at<double>(axis)<lo[axis] || translation.at<double>(axis)>hi[axis]) return false;
        return true;
    }
    bool nearLimit(const cv::Mat& translation) const {
        const auto lo=lower(),hi=upper();
        for (int axis=0;axis<3;++axis)
            if (std::min(translation.at<double>(axis)-lo[axis],hi[axis]-translation.at<double>(axis))<.1) return true;
        return false;
    }
};

struct FixedStereoIntrinsics { cv::Mat leftK,leftD,rightK,rightD; };
struct TranslationConstraintReport {
    cv::Mat initialR,initialT;
    double initialRms=0;
    cv::Vec3d lowerMm,upperMm;
    int iterations=0;
    bool converged=false,atLimit=false;
};

// Fixed K/D bundle adjustment: one shared rig pose and an independent board pose per image pair.
// T = center + halfWidth*tanh(q) enforces the bounds on every optimizer evaluation.
class BoundedStereoProblem : public cv::LMSolver::Callback {
    std::vector<cv::Point3d> board_;
    std::vector<std::vector<cv::Point2f>> left_,right_;
    FixedStereoIntrinsics cameras_;
public:
    TranslationBounds bounds;
    BoundedStereoProblem(const std::vector<cv::Point3f>& board,
        const std::vector<std::vector<cv::Point2f>>& left,
        const std::vector<std::vector<cv::Point2f>>& right,
        FixedStereoIntrinsics cameras,double initialX)
        :left_(left),right_(right),cameras_(std::move(cameras)),bounds(initialX) {
        for (const auto& point:board) board_.emplace_back(point.x,point.y,point.z);
        if (board_.empty() || left_.empty() || left_.size()!=right_.size())
            throw std::runtime_error("Invalid observations for bounded stereo calibration");
        for (std::size_t i=0;i<left_.size();++i)
            if (left_[i].size()!=board_.size() || right_[i].size()!=board_.size())
                throw std::runtime_error("Incomplete observations for bounded stereo calibration");
    }
    cv::Mat initialize(const cv::Mat& rotation,const cv::Mat& translation) const {
        cv::Mat parameters=cv::Mat::zeros(6+6*static_cast<int>(left_.size()),1,CV_64F);
        cv::Mat rotationVector; cv::Rodrigues(rotation,rotationVector);
        rotationVector.copyTo(parameters.rowRange(0,3));
        for (int axis=0;axis<3;++axis) {
            // Start inside the interval so an out-of-bounds initial estimate retains a useful gradient.
            const double fraction=std::clamp((translation.at<double>(axis)-bounds.center[axis])/bounds.halfWidth[axis],-.9,.9);
            parameters.at<double>(3+axis)=std::atanh(fraction);
        }
        for (std::size_t i=0;i<left_.size();++i) {
            cv::Mat rv,tv;
            if (!cv::solvePnP(board_,left_[i],cameras_.leftK,cameras_.leftD,rv,tv) || tv.at<double>(2)<=0)
                throw std::runtime_error("Cannot initialize a front-facing board pose");
            const int base=6+6*static_cast<int>(i);
            rv.copyTo(parameters.rowRange(base,base+3)); tv.copyTo(parameters.rowRange(base+3,base+6));
        }
        return parameters;
    }
    bool compute(cv::InputArray input,cv::OutputArray error,cv::OutputArray jacobian) const override {
        const cv::Mat p=input.getMat();
        const bool derivatives=jacobian.needed();
        const int points=static_cast<int>(board_.size()),rows=4*points*static_cast<int>(left_.size());
        error.create(rows,1,CV_64F); cv::Mat residual=error.getMat();
        cv::Mat J;
        if (derivatives) { jacobian.create(rows,p.rows,CV_64F); J=jacobian.getMat(); J.setTo(0); }
        cv::Vec3d dTranslation;
        const cv::Mat rigRotation=p.rowRange(0,3),rigTranslation=bounds.decode(p,&dTranslation);
        for (std::size_t view=0;view<left_.size();++view) {
            const int base=6+6*static_cast<int>(view),row=4*points*static_cast<int>(view);
            const cv::Mat boardRotation=p.rowRange(base,base+3),boardTranslation=p.rowRange(base+3,base+6);
            cv::Mat rightRotation,rightTranslation,rrBoard,rrRig,ttBoard,ttRig,rtBoard,rtRig,trBoard,trRig;
            if (derivatives) {
                cv::composeRT(boardRotation,boardTranslation,rigRotation,rigTranslation,rightRotation,rightTranslation,
                    rrBoard,rtBoard,rrRig,rtRig,trBoard,ttBoard,trRig,ttRig);
            } else cv::composeRT(boardRotation,boardTranslation,rigRotation,rigTranslation,rightRotation,rightTranslation);
            std::vector<cv::Point2d> projectedLeft,projectedRight;
            cv::Mat jl,jr;
            if (derivatives) {
                cv::projectPoints(board_,boardRotation,boardTranslation,cameras_.leftK,cameras_.leftD,projectedLeft,jl);
                cv::projectPoints(board_,rightRotation,rightTranslation,cameras_.rightK,cameras_.rightD,projectedRight,jr);
            } else {
                cv::projectPoints(board_,boardRotation,boardTranslation,cameras_.leftK,cameras_.leftD,projectedLeft);
                cv::projectPoints(board_,rightRotation,rightTranslation,cameras_.rightK,cameras_.rightD,projectedRight);
            }
            for (int j=0;j<points;++j) {
                residual.at<double>(row+2*j)=projectedLeft[j].x-left_[view][j].x;
                residual.at<double>(row+2*j+1)=projectedLeft[j].y-left_[view][j].y;
                residual.at<double>(row+2*points+2*j)=projectedRight[j].x-right_[view][j].x;
                residual.at<double>(row+2*points+2*j+1)=projectedRight[j].y-right_[view][j].y;
            }
            if (derivatives) {
                jl.colRange(0,6).copyTo(J(cv::Rect(base,row,6,2*points)));
                cv::Mat rigJacobian(2*points,6,CV_64F),boardJacobian(2*points,6,CV_64F);
                cv::Mat leftHalf=jr.colRange(0,3),rightHalf=jr.colRange(3,6);
                cv::Mat value=leftHalf*rrRig+rightHalf*trRig; value.copyTo(rigJacobian.colRange(0,3));
                value=leftHalf*rtRig+rightHalf*ttRig; value.copyTo(rigJacobian.colRange(3,6));
                for (int axis=0;axis<3;++axis) rigJacobian.col(3+axis)*=dTranslation[axis];
                value=leftHalf*rrBoard+rightHalf*trBoard; value.copyTo(boardJacobian.colRange(0,3));
                value=leftHalf*rtBoard+rightHalf*ttBoard; value.copyTo(boardJacobian.colRange(3,6));
                rigJacobian.copyTo(J(cv::Rect(0,row+2*points,6,2*points)));
                boardJacobian.copyTo(J(cv::Rect(base,row+2*points,6,2*points)));
            }
        }
        return cv::checkRange(residual) && (!derivatives || cv::checkRange(J));
    }
    bool inFront(const cv::Mat& p) const {
        cv::Mat rigRotation; cv::Rodrigues(p.rowRange(0,3),rigRotation);
        const cv::Mat rigTranslation=bounds.decode(p);
        for (std::size_t view=0;view<left_.size();++view) {
            const int base=6+6*static_cast<int>(view);
            cv::Mat boardRotation; cv::Rodrigues(p.rowRange(base,base+3),boardRotation);
            for (const auto& point:board_) {
                const cv::Mat left=boardRotation*(cv::Mat_<double>(3,1)<<point.x,point.y,point.z)+p.rowRange(base+3,base+6);
                const cv::Mat right=rigRotation*left+rigTranslation;
                if (left.at<double>(2)<=0 || right.at<double>(2)<=0) return false;
            }
        }
        return true;
    }
};

inline TranslationConstraintReport RefineStereoTranslation(const std::vector<cv::Point3f>& board,
    const std::vector<std::vector<cv::Point2f>>& left,const std::vector<std::vector<cv::Point2f>>& right,
    const FixedStereoIntrinsics& cameras,cv::Mat& R,cv::Mat& T,cv::Mat& E,cv::Mat& F,cv::Mat& errors,double& rms) {
    TranslationConstraintReport report;
    report.initialR=R.clone(); report.initialT=T.clone(); report.initialRms=rms;
    auto problem=cv::makePtr<BoundedStereoProblem>(board,left,right,cameras,T.at<double>(0));
    report.lowerMm=problem->bounds.lower(); report.upperMm=problem->bounds.upper();
    cv::Mat parameters=problem->initialize(R,T);
    constexpr int maxIterations=200;
    report.iterations=cv::LMSolver::create(problem,maxIterations,1e-9)->run(parameters);
    report.converged=report.iterations>=0 && report.iterations<maxIterations;
    if (!cv::checkRange(parameters) || !problem->inFront(parameters))
        throw std::runtime_error("Bounded stereo calibration produced invalid/behind-camera board poses");
    cv::Rodrigues(parameters.rowRange(0,3),R); T=problem->bounds.decode(parameters);
    if (!problem->bounds.contains(T)) throw std::runtime_error("Bounded stereo translation escaped its limits");
    report.atLimit=problem->bounds.nearLimit(T);
    cv::Mat residual; if (!problem->compute(parameters,residual,cv::noArray()))
        throw std::runtime_error("Nonfinite bounded stereo residuals");
    const int count=static_cast<int>(board.size());
    errors.create(static_cast<int>(left.size()),2,CV_64F);
    for (int i=0;i<errors.rows;++i) for (int side=0;side<2;++side)
        errors.at<double>(i,side)=cv::norm(residual.rowRange(4*count*i+2*count*side,4*count*i+2*count*(side+1)))/std::sqrt(count);
    rms=cv::norm(residual)/std::sqrt(2.*count*left.size());
    const double x=T.at<double>(0),y=T.at<double>(1),z=T.at<double>(2);
    const cv::Mat skew=(cv::Mat_<double>(3,3)<<0.,-z,y,z,0.,-x,-y,x,0.);
    E=skew*R; F=cameras.rightK.inv().t()*E*cameras.leftK.inv();
    if (!std::isfinite(rms) || !cv::checkRange(F)) throw std::runtime_error("Invalid bounded stereo result");
    return report;
}

inline void WriteTranslationConstraint(cv::FileStorage& yaml,const TranslationConstraintReport& report) {
    yaml<<"translation_constraint"<<"{"<<"enabled"<<1
        <<"coordinate_frame"<<"right_camera_optical_x_right_y_down_z_forward"
        <<"x_sign_policy"<<"preserve_unconstrained_initial_Tx_sign"
        <<"lower_mm"<<report.lowerMm<<"upper_mm"<<report.upperMm
        <<"optimizer"<<"joint_R_T_and_board_poses_LM_with_tanh_translation_bounds"
        <<"iterations"<<report.iterations<<"converged"<<static_cast<int>(report.converged)
        <<"near_bound_0_1_mm"<<static_cast<int>(report.atLimit)<<"}"
        <<"unconstrained_R"<<report.initialR<<"unconstrained_T_mm"<<report.initialT
        <<"unconstrained_rms_reprojection_error_px"<<report.initialRms;
}
} // namespace stereo_calibration
