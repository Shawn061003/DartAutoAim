#pragma once
#include "Acquisition/CameraInput.h"
#include "stereo_translation_constraint.hpp"
#include "../aprilgrid.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <limits>
#include <string>
#include <vector>
namespace fs = std::filesystem;
namespace stereo_calibration {
constexpr int kColumns=aprilgrid::kColumns,kRows=aprilgrid::kRows;
inline const cv::Size kFrameSize(1440,1080);
struct Camera { cv::Mat K, D; cv::Size referenceSize; };
struct MonocularFit {
    Camera camera;
    double rms=std::numeric_limits<double>::quiet_NaN();
    cv::Mat perViewErrors;
    std::vector<cv::Mat> rvecs,tvecs;
};
MonocularFit FitMonocular(const std::vector<std::vector<cv::Point2f>>&,cv::Size);
void WriteMonocularReport(const fs::path&,const MonocularFit&,
    const std::vector<std::vector<cv::Point2f>>&,const std::vector<fs::path>&);
struct Pair {
    std::size_t group;
    int number;
    fs::path leftPath, rightPath;
    std::vector<cv::Point2f> left, right; // 左全幅参考坐标 / 右原图坐标。
    bool rightReversed = false;
    double epipolarRms = 0;
};
void Require(bool,const std::string&);
Camera ReadCamera(const fs::path&);
std::string Number(double,int digits=3);
} // namespace stereo_calibration
namespace stereo_live {
using namespace stereo_calibration;
using Clock=std::chrono::steady_clock;
struct Options {
    fs::path config = fs::path(DART_PROJECT_ROOT)/"config/dart.yaml", output;
    fs::path leftIntrinsics, rightIntrinsics;
    std::string captureMode="sdk_live_native_full_frames";
    double holdSeconds = 0.8;
    int minPairs = 10, maxPairs = 30;
    std::int64_t pairMs = -1;
    bool automatic = true, help = false;
};
struct LivePair {
    bool timestampsKnown=true;
    Pair pair;
    std::uint64_t leftId=0,rightId=0;
    std::int64_t leftTime=0,rightTime=0;
};
struct Epipolar {
    double rms=0,p95=0,maximum=0;
    std::vector<double> maxSide;
};
struct Solution {
    std::vector<LivePair> pairs;
    // 与本轮图对、外参和误差一同保存，避免显示/导出时混用旧配置内参。
    MonocularFit leftFit,rightFit;
    bool intrinsicsRefitted=false;
    cv::Mat R,T,E,F,errors;
    std::vector<Epipolar> perPair;
    double rms=std::numeric_limits<double>::quiet_NaN(),epiRms=0,epiP95=0,epiMax=0;
    bool passed=false;
    TranslationConstraintReport translationConstraint;
};
Camera Intrinsics(const CameraParams&,const fs::path&);
std::uint64_t TimestampDifference(std::int64_t,std::int64_t);
bool AlignRight(const std::vector<cv::Point2f>&,std::vector<cv::Point2f>&);
Epipolar MeasureEpipolar(const std::vector<cv::Point2f>&,const std::vector<cv::Point2f>&,const Camera&,const Camera&,const cv::Mat&);
Solution Solve(std::vector<LivePair>,cv::Size,cv::Size,int);
Solution SolveFixedIntrinsics(std::vector<LivePair>,const Camera&,const Camera&,int);
Solution UnconstrainedSolution(const Solution&,int);
std::size_t Worst(const Solution&);
void Save(const Options&,const Solution&);
} // namespace stereo_live
