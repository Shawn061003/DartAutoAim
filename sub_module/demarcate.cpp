// Live single-camera calibration: Hikrobot SDK + OpenCV 4.5.4 interactive calibration.
// Upstream support code and license: opencv_interactive_calibration/README.md.
// ./build/demarcate --side=left [--config=config/dart.yaml] [--model=auto|full|narrow|k1]
#include "Acquisition/CameraInput.h"
#include "Acquisition/Quit.h"
#include "common/DartConfig.h"
#include "opencv_interactive_calibration/frameProcessor.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <filesystem>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

bool calib::showOverlayMessage(const std::string& message)
{
    std::cout << "[calibration] " << message << '\n';
    return true;
}

namespace livecalib {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr const char* windowName = "Hikrobot - OpenCV interactive calibration";

struct Options {
    fs::path config = fs::path(DART_PROJECT_ROOT) / "config/dart.yaml";
    fs::path output;
    std::string side, model = "auto";
    cv::Size board{11, 8};
    double squareMm = 30, holdSeconds = 0.8;
    int minViews = 10, maxViews = 30;
    bool automatic = true;
};

Options parseOptions(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help | | Show help without opening a camera}"
        "{config | | dart.yaml path (input.mode is ignored; always live SDK capture)}"
        "{side | | Required: left or right; only this camera is opened}"
        "{output | | New output directory (must not exist)}"
        "{model | auto | auto (official flags tuning), full, narrow or k1}"
        "{cols | 11 | Chessboard inner corner columns}"
        "{rows | 8 | Chessboard inner corner rows}"
        "{square-mm | 30 | Physical square side in millimetres}"
        "{hold | 0.8 | Stable duration in seconds before automatic capture}"
        "{min-views | 10 | Minimum views for quality evaluation}"
        "{max-views | 30 | Maximum retained views; official filter removes excess}"
        "{auto | true | Start with automatic capture enabled}");
    if (parser.has("help")) { parser.printMessage(); return {}; }
    Options o;
    if (parser.has("config")) o.config = parser.get<std::string>("config");
    o.side = parser.get<std::string>("side");
    o.model = parser.get<std::string>("model");
    o.board = {parser.get<int>("cols"), parser.get<int>("rows")};
    o.squareMm = parser.get<double>("square-mm");
    o.holdSeconds = parser.get<double>("hold");
    o.minViews = parser.get<int>("min-views"); o.maxViews = parser.get<int>("max-views");
    o.automatic = parser.get<bool>("auto");
    if (!parser.check()) { parser.printErrors(); throw std::invalid_argument("Invalid command line"); }
    // CommandLineParser accepts unrecognized keys, so reject them explicitly.
    const std::vector<std::string> keys{"config", "side", "output", "model", "cols", "rows",
        "square-mm", "hold", "min-views", "max-views", "auto"};
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) throw std::invalid_argument("Use --name=value options");
        key = key.substr(2, key.find('=') == std::string::npos ? std::string::npos : key.find('=') - 2);
        if (std::find(keys.begin(), keys.end(), key) == keys.end())
            throw std::invalid_argument("Unknown option: " + key);
    }
    if (o.side != "left" && o.side != "right") throw std::invalid_argument("--side=left or --side=right is required");
    if (o.model != "auto" && o.model != "full" && o.model != "narrow" && o.model != "k1")
        throw std::invalid_argument("--model must be auto, full, narrow or k1");
    if (o.board.width < 3 || o.board.height < 3 || o.board.width > 100 || o.board.height > 100 ||
        !std::isfinite(o.squareMm) || o.squareMm <= 0 || o.squareMm > 10000 ||
        !std::isfinite(o.holdSeconds) || o.holdSeconds < 0.1 || o.holdSeconds > 60 ||
        o.minViews < 3 || o.maxViews <= o.minViews || o.maxViews > 100)
        throw std::invalid_argument("Invalid board, hold duration, or view count limits");
    if (parser.has("output")) {
        o.output = parser.get<std::string>("output");
        if (o.output.empty()) throw std::invalid_argument("--output cannot be empty");
    } else {
        const auto time = std::time(nullptr); std::tm local{}; localtime_r(&time, &local);
        std::ostringstream name; name << "live_" << o.side << '_' << std::put_time(&local, "%Y%m%d_%H%M%S");
        o.output = fs::path(DART_PROJECT_ROOT) / "CalibrationResults" / name.str();
    }
    o.output = fs::absolute(o.output);
    if (fs::exists(o.output)) throw std::invalid_argument("Output already exists; choose a new directory: " + o.output.string());
    return o;
}

// SDK acquisition keeps running while detection/calibration is busy. Only one pending frame is retained.
class LatestCamera {
    CameraDevice device_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    std::mutex mutex_;
    std::optional<CameraFrame> latest_;
    std::exception_ptr error_;
public:
    LatestCamera(const CameraParams& params, CameraSide side) : device_(params, side) {
        device_.StartGrabbing();
        thread_ = std::thread([this] {
            try {
                while (!stop_.load()) {
                    auto frame = device_.GrabImage(100);
                    if (frame) { std::lock_guard<std::mutex> lock(mutex_); latest_ = std::move(frame); }
                }
            } catch (...) { std::lock_guard<std::mutex> lock(mutex_); error_ = std::current_exception(); }
        });
    }
    ~LatestCamera() { stop_.store(true); if (thread_.joinable()) thread_.join(); device_.StopGrabbing(); }
    std::optional<CameraFrame> take() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
        auto result = std::move(latest_); latest_.reset(); return result;
    }
};

// Keep the upstream chessboard detection, object-point generation and angle check.
// Only frame-count-based intake is replaced by wall-clock stability and pose diversity.
class BoardProcessor : public calib::CalibProcessor {
public:
    BoardProcessor(cv::Ptr<calib::calibrationData> data, calib::captureParameters& params)
        : calib::CalibProcessor(data, params) {}
    bool observe(cv::Mat& annotated) {
        mCurrentImagePoints.clear(); mTemplateLocations.clear();
        return detectAndParseChessboard(annotated);
    }
    const std::vector<cv::Point2f>& points() const { return mCurrentImagePoints; }
    bool acceptableAngle() {
        saveFrameData();
        const bool rejected = checkLastFrame();
        if (!rejected) { mCalibData->imagePoints.pop_back(); mCalibData->objectPoints.pop_back(); }
        return !rejected;
    }
};

// A job owns its frame and calibration snapshot. No processor state is shared with the UI/solver.
struct DetectionJob {
    CameraFrame frame;
    calib::captureParameters capture;
    cv::Mat cameraMatrix, distCoeffs;
    Clock::time_point receivedAt;
    std::uint64_t epoch = 0;
};

struct DetectionResult {
    CameraFrame frame;
    cv::Mat annotated;
    std::vector<cv::Point2f> points;
    Clock::time_point receivedAt;
    std::uint64_t epoch = 0;
    bool found = false, angleOK = false;
    double milliseconds = 0;
};

DetectionResult detectBoard(DetectionJob job) {
    const auto started = Clock::now();
    auto data = cv::makePtr<calib::calibrationData>();
    data->imageSize = job.frame.image.size();
    data->cameraMatrix = std::move(job.cameraMatrix); data->distCoeffs = std::move(job.distCoeffs);
    BoardProcessor processor(data,job.capture);
    DetectionResult result;
    result.frame = std::move(job.frame); result.receivedAt = job.receivedAt; result.epoch = job.epoch;
    result.annotated = result.frame.image.clone();
    result.found = processor.observe(result.annotated);
    if (result.found) { result.points = processor.points(); result.angleOK = processor.acceptableAngle(); }
    result.milliseconds = std::chrono::duration<double,std::milli>(Clock::now()-started).count();
    return result;
}

// One in-flight job, no detection queue. The UI submits the newest frame when this becomes free.
class BackgroundDetector {
    std::future<DetectionResult> pending_;
    std::function<DetectionResult(DetectionJob)> detect_;
public:
    explicit BackgroundDetector(std::function<DetectionResult(DetectionJob)> detect = detectBoard)
        : detect_(std::move(detect)) {}
    bool busy() const { return pending_.valid(); }
    bool submit(DetectionJob job) {
        if (busy()) return false;
        pending_ = std::async(std::launch::async,detect_,std::move(job));
        return true;
    }
    std::optional<DetectionResult> takeReady() {
        if (!busy() || pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return {};
        return pending_.get(); // Propagate worker errors on the UI thread; destruction joins outstanding work.
    }
};

struct Sample {
    std::vector<cv::Point2f> points;
    std::string image;
    std::uint64_t frameId = 0;
};

struct FrameWriteError : std::runtime_error { using std::runtime_error::runtime_error; };

double cornerMotion(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    if (a.size() != b.size() || a.empty()) return std::numeric_limits<double>::infinity();
    // A 180-degree corner ordering change must not count as a new pose.
    double forward = 0, reverse = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        forward += cv::norm(a[i] - b[i]) * cv::norm(a[i] - b[i]);
        reverse += cv::norm(a[i] - b[a.size() - 1 - i]) * cv::norm(a[i] - b[a.size() - 1 - i]);
    }
    return std::sqrt(std::min(forward, reverse) / a.size());
}

bool diverse(const std::vector<cv::Point2f>& points, const std::vector<Sample>& samples, cv::Size size) {
    const double minimumMotion = 0.008 * std::hypot(size.width, size.height);
    for (const auto& sample : samples) if (cornerMotion(points, sample.points) < minimumMotion) return false;
    return true;
}

struct Solution {
    cv::Ptr<calib::calibrationData> data;
    std::vector<Sample> samples;
    int flags = 0;
    bool enough = false, rmsOK = false, confidenceOK = false, coverageOK = false;
};

int modelFlags(const std::string& model) {
    int flags = 0;
    if (model == "narrow" || model == "k1")
        flags = cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3;
    if (model == "narrow") flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    return flags;
}

std::vector<cv::Point3f> objectTemplate(const Options& o) {
    std::vector<cv::Point3f> points;
    for (int y = 0; y < o.board.height; ++y) for (int x = 0; x < o.board.width; ++x)
        points.emplace_back(x * o.squareMm, y * o.squareMm, 0);
    return points;
}

Solution solve(const Options& o, cv::Size size, std::vector<Sample> samples, int flags) {
    Solution result; result.samples = std::move(samples); result.flags = flags;
    result.data = cv::makePtr<calib::calibrationData>(); result.data->imageSize = size;
    auto& data = *result.data;
    for (const auto& sample : result.samples) {
        data.imagePoints.push_back(sample.points); data.objectPoints.push_back(objectTemplate(o));
    }
    if (result.samples.size() < 3) return result;
    data.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    data.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    calib::calibController quality(result.data, flags, o.model == "auto", o.minViews);
    calib::calibDataController filter(result.data, o.maxViews, 0.1);
    for (int pass = 0; pass < 10; ++pass) {
        data.totalAvgErr = cv::calibrateCamera(data.objectPoints, data.imagePoints, size,
            data.cameraMatrix, data.distCoeffs, data.rvecs, data.tvecs,
            data.stdDeviations, cv::noArray(), data.perViewErrors, flags,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 60, 1e-8));
        if (!std::isfinite(data.totalAvgErr) || !cv::checkRange(data.cameraMatrix) ||
            !cv::checkRange(data.distCoeffs) || data.cameraMatrix.at<double>(0,0) <= 0 ||
            data.cameraMatrix.at<double>(1,1) <= 0)
            throw std::runtime_error("Calibration produced invalid parameters; collect more diverse views");
        // Official excess-view selection; keep filenames aligned, then refit retained views.
        const auto before = data.imagePoints;
        filter.filterFrames();
        if (data.imagePoints.size() != before.size()) {
            std::vector<Sample> retained;
            for (std::size_t i = 0; i < before.size(); ++i)
                if (std::find(data.imagePoints.begin(), data.imagePoints.end(), before[i]) != data.imagePoints.end())
                    retained.push_back(result.samples[i]);
            result.samples = std::move(retained);
            continue;
        }
        quality.updateState();
        const int nextFlags = quality.getNewFlags();
        if (nextFlags != flags) { flags = nextFlags; continue; }
        result.flags = flags;
        result.enough = quality.getFramesNumberState(); result.rmsOK = quality.getRMSState();
        result.confidenceOK = quality.getConfidenceIntrervalsState(); result.coverageOK = quality.getPointsCoverageState();
        return result;
    }
    throw std::runtime_error("Calibration/flag selection did not converge");
}

void saveReport(const Options& o, const Solution& solution) {
    if (!solution.data || solution.data->cameraMatrix.empty()) throw std::runtime_error("Need at least 3 solved views before saving");
    const auto& d = *solution.data;
    // Write a temporary report first; replace only this session's own report after a successful close.
    const auto temporary = o.output / "intrinsics.pending.yaml";
    cv::FileStorage file(temporary.string(), cv::FileStorage::WRITE);
    if (!file.isOpened()) throw std::runtime_error("Cannot write calibration report");
    file << "image_width" << d.imageSize.width << "image_height" << d.imageSize.height
         << "board_columns" << o.board.width << "board_rows" << o.board.height << "square_size_mm" << o.squareMm
         << "camera_matrix" << d.cameraMatrix << "distortion_coefficients" << d.distCoeffs.reshape(1,5)
         << "distortion_order" << "k1,k2,p1,p2,k3" << "coordinate_system" << "native_full_image_zero_offset"
         << "camera_side" << o.side << "config_source" << fs::absolute(o.config).string()
         << "source_directory" << o.output.string() << "calibration_model" << o.model << "calibration_flags" << solution.flags
         << "calibration_backend" << "OpenCV 4.5.4 interactive calibration (SDK adaptation)"
         << "used_views" << static_cast<int>(solution.samples.size())
         << "rms_reprojection_error_px" << d.totalAvgErr
         << "std_deviations_intrinsics" << d.stdDeviations << "per_view_rms_px" << d.perViewErrors
         << "enough_views" << static_cast<int>(solution.enough) << "rms_check_passed" << static_cast<int>(solution.rmsOK)
         << "confidence_check_passed" << static_cast<int>(solution.confidenceOK)
         << "coverage_check_passed" << static_cast<int>(solution.coverageOK) << "distance_validated" << 0;
    file << "board_poses" << "[";
    for (std::size_t i = 0; i < solution.samples.size(); ++i) {
        const auto& sample = solution.samples[i];
        file << "{" << "image" << sample.image << "frame_id" << std::to_string(sample.frameId)
             << "offset_x" << 0 << "offset_y" << 0 << "source_width" << d.imageSize.width << "source_height" << d.imageSize.height
             << "image_points_reference" << sample.points << "rvec" << d.rvecs[i] << "tvec_mm" << d.tvecs[i] << "}";
    }
    file << "]"; file.release();
    fs::rename(temporary, o.output / "intrinsics.yaml");
    std::cout << "Saved " << o.output / "intrinsics.yaml" << "; views=" << solution.samples.size()
              << " RMS=" << d.totalAvgErr << " px; real-world distance is not validated\n";
}

void text(cv::Mat& canvas, const std::string& value, int x, int y, cv::Scalar color = {230,230,230}) {
    cv::putText(canvas, value, {x,y}, cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv::LINE_AA);
}

cv::Mat dashboard(const cv::Mat& annotated, const Options& o, const std::vector<Sample>& samples,
                  const Solution& solved, bool busy, bool automatic, bool undistorted,
                  const std::string& status, const DetectionResult* detection = nullptr,
                  bool detecting = false, std::uint64_t liveFrameId = 0) {
    cv::Mat canvas(900,1280,CV_8UC3,cv::Scalar(25,25,25));
    if (!annotated.empty()) {
        const double scale = std::min(960.0 / annotated.cols, 640.0 / annotated.rows);
        cv::Mat preview; cv::resize(annotated, preview, {}, scale, scale, cv::INTER_AREA);
        if (undistorted && solved.data && !solved.data->cameraMatrix.empty()) {
            cv::Mat k = solved.data->cameraMatrix.clone();
            k.row(0) *= static_cast<double>(preview.cols) / annotated.cols;
            k.row(1) *= static_cast<double>(preview.rows) / annotated.rows;
            cv::Mat corrected; cv::undistort(preview, corrected, k, solved.data->distCoeffs); preview = corrected;
        }
        preview.copyTo(canvas(cv::Rect(0,0,preview.cols,preview.rows)));
    }
    if (!annotated.empty()) text(canvas,"LIVE #" + std::to_string(liveFrameId),10,22,{80,255,80});
    // Historical corners belong only on their own image, never on the latest live frame.
    if (detection && !detection->annotated.empty()) {
        cv::rectangle(canvas,{645,405,315,230},{20,20,20},cv::FILLED);
        text(canvas,std::string(detection->found ? "Board #" : "No board #") +
             std::to_string(detection->frame.frame_id) + " (RAW)",650,423);
        const double scale = std::min(300.0/detection->annotated.cols,200.0/detection->annotated.rows);
        cv::Mat inset; cv::resize(detection->annotated,inset,{},scale,scale,cv::INTER_AREA);
        inset.copyTo(canvas(cv::Rect(650,430,inset.cols,inset.rows)));
    }
    std::string detectionStatus = detecting ? "Detector: WORKING" : "Detector: READY";
    if (detection) detectionStatus += cv::format(" | last %.0f ms | frame age %.1f s",detection->milliseconds,
        std::chrono::duration<double>(Clock::now()-detection->receivedAt).count());
    text(canvas,detectionStatus,10,645);
    const int x = 975; int y = 25;
    auto line = [&](const std::string& s) { text(canvas,s,x,y); y += 28; };
    line("Camera: " + o.side); line("Model: " + o.model);
    line("Views: " + std::to_string(samples.size()) + " / " + std::to_string(o.maxViews));
    line(automatic ? "Capture: AUTO" : "Capture: MANUAL");
    line(busy ? "Solver: WORKING" : "Solver: READY");
    line(undistorted ? "View: UNDISTORTED" : "View: RAW");
    if (solved.data && !solved.data->cameraMatrix.empty()) {
        const auto& d = *solved.data;
        line(cv::format("RMS: %.4f px",d.totalAvgErr));
        double maximum = 0; cv::minMaxLoc(d.perViewErrors,nullptr,&maximum); line(cv::format("Worst view: %.4f px",maximum));
        line(cv::format("fx: %.2f",d.cameraMatrix.at<double>(0,0)));
        line(cv::format("fy: %.2f",d.cameraMatrix.at<double>(1,1)));
        line(cv::format("cx: %.2f",d.cameraMatrix.at<double>(0,2)));
        line(cv::format("cy: %.2f",d.cameraMatrix.at<double>(1,2)));
        line(cv::format("95%% dfx: %.2f",1.96*d.stdDeviations.at<double>(0)));
        line(cv::format("95%% dfy: %.2f",1.96*d.stdDeviations.at<double>(1)));
        line(std::string("Count: ") + (solved.enough ? "OK" : "NEED MORE"));
        line(std::string("RMS: ") + (solved.rmsOK ? "OK" : "HIGH"));
        line(std::string("Confidence: ") + (solved.confidenceOK ? "OK" : "WEAK"));
        line(std::string("Coverage: ") + (solved.coverageOK ? "OK" : "NEED MORE"));
        line("Applied flags: " + std::to_string(solved.flags));
    }
    // Native sensor corner occupancy; preview scaling never enters calibration.
    cv::Mat coverage(200,400,CV_8UC3,cv::Scalar(40,40,40));
    const cv::Size size = solved.data ? solved.data->imageSize : annotated.size();
    if (!size.empty()) for (const auto& sample : samples) for (const auto& p : sample.points) {
        const int gx = std::clamp(static_cast<int>(p.x*10/size.width),0,9);
        const int gy = std::clamp(static_cast<int>(p.y*10/size.height),0,9);
        cv::rectangle(coverage,{gx*40,gy*20,40,20},{55,140,55},cv::FILLED);
    }
    for (int i=0;i<=10;++i) { cv::line(coverage,{i*40,0},{i*40,199},{85,85,85}); cv::line(coverage,{0,i*20},{399,i*20},{85,85,85}); }
    coverage.copyTo(canvas(cv::Rect(10,675,400,200)));
    text(canvas,"Corner coverage (native sensor)",10,660);
    text(canvas,status.substr(0,95),435,690,{80,220,255});
    text(canvas,"SPACE: capture | A: auto/manual | R: undo | D: clear",435,735);
    text(canvas,"S: save | U: undistort preview | ESC/Q: quit",435,770);
    text(canvas,"Pause the board; vary position, scale and tilt.",435,815);
    text(canvas,"RMS/quality OK does not validate stereo ranging.",435,855);
    return canvas;
}

int run(const Options& o) {
    const DartCongfig config(o.config.string());
    const CameraParams& params = o.side == "left" ? config.leftCamera : config.rightCamera;
    const CameraSide side = o.side == "left" ? CameraSide::Left : CameraSide::Right;
    CameraStopSignals signals;
    LatestCamera camera(params, side);
    // Do not overwrite an existing session even if it appeared after option validation.
    if (!fs::create_directories(o.output)) throw std::runtime_error("Cannot create a new output directory");
    fs::create_directory(o.output / "frames");
    std::cout << "Live calibration: " << o.side << " camera=" << params.deviceIp << " net=" << params.netIp
              << " size=" << params.width << 'x' << params.height << " output=" << o.output << '\n'
              << "Reading camera settings only; input.mode and old K/D do not constrain this calibration.\n";
    cv::namedWindow(windowName,cv::WINDOW_NORMAL); cv::resizeWindow(windowName,1280,900);
    auto data = cv::makePtr<calib::calibrationData>(); data->imageSize = {params.width,params.height};
    calib::captureParameters capture; capture.board = calib::Chessboard; capture.boardSize = o.board;
    capture.squareSize = o.squareMm; capture.templDst = 295; capture.captureDelay = 1; capture.fps = 1;
    BackgroundDetector detector;
    std::optional<DetectionResult> detection;
    std::optional<CameraFrame> latest;
    std::vector<Sample> samples;
    Solution solved; solved.data = data; solved.flags = modelFlags(o.model);
    std::future<Solution> pending;
    std::uint64_t revision = 0, pendingRevision = 0, serial = 0, detectionEpoch = 0;
    std::uint64_t previewSequence = 0, submittedSequence = 0;
    std::uint64_t solvedRevision = std::numeric_limits<std::uint64_t>::max();
    bool automatic = o.automatic, undistorted = false, captureRequested = false;
    bool saved = false;
    std::string status = "Waiting for a complete chessboard";
    cv::Mat annotated;
    std::vector<cv::Point2f> anchor;
    auto stableSince = Clock::now(), lastFrameAt = Clock::now(), captureAfter = Clock::now();
    auto launch = [&](cv::Mat raw = {}, std::string image = {}) {
        const auto snapshot = samples; const auto size = data->imageSize;
        const int flags = o.model == "auto" ? solved.flags : modelFlags(o.model);
        pendingRevision = revision;
        pending = std::async(std::launch::async,[o,size,snapshot,flags,raw,image] {
            if (!raw.empty()) {
                try {
                    if (!cv::imwrite((o.output/image).string(),raw,{cv::IMWRITE_PNG_COMPRESSION,1}))
                        throw FrameWriteError("Cannot save original camera frame");
                } catch (const cv::Exception& e) { throw FrameWriteError(e.what()); }
            }
            return solve(o,size,snapshot,flags);
        });
    };
    try {
        while (!signals.stopRequested()) {
            if (pending.valid() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                try {
                    Solution completed = pending.get();
                    if (pendingRevision == revision) {
                        solved = std::move(completed); samples = solved.samples;
                        solvedRevision = revision;
                        data->cameraMatrix = solved.data->cameraMatrix; data->distCoeffs = solved.data->distCoeffs;
                        status = data->cameraMatrix.empty() ? "Need at least 3 distinct views" : "Updated; move board to a new pose";
                        if (!data->cameraMatrix.empty()) {
                            calib::calibDataController report(solved.data,o.maxViews,0.1); report.printParametersToConsole(std::cout);
                        }
                    }
                } catch (const FrameWriteError&) { throw; }
                catch (const std::exception& e) { status = std::string("Solve failed: ")+e.what(); std::cerr << status << '\n'; }
                anchor.clear(); ++detectionEpoch;
            }
            const bool busy = pending.valid();
            if (auto frame = camera.take()) {
                lastFrameAt = Clock::now(); latest = std::move(frame); ++previewSequence;
                annotated = latest->image; // Read-only native frame; the worker draws on its own clone.
            } else if (Clock::now()-lastFrameAt>std::chrono::seconds(5)) {
                throw std::runtime_error("No camera frame for 5 seconds; check SDK connection/exposure");
            }
            if (auto result = detector.takeReady()) {
                if (result->epoch == detectionEpoch) {
                    const bool manual = captureRequested && result->receivedAt >= captureAfter;
                    if (!result->found || busy) anchor.clear();
                    else {
                        const auto& points = result->points;
                        const double threshold = 0.0006 * std::hypot(params.width,params.height);
                        if (anchor.empty() || cornerMotion(points,anchor)>threshold) {
                            anchor = points; stableSince = result->receivedAt;
                        }
                        const double held = std::chrono::duration<double>(result->receivedAt-stableSince).count();
                        const bool novel = diverse(points,samples,data->imageSize);
                        status = novel ? cv::format("Hold steady: %.1f / %.1f s",held,o.holdSeconds)
                                       : "Already sampled this pose; move/tilt/resize the board";
                        if ((manual || (automatic && held >= o.holdSeconds)) && novel) {
                            if (result->angleOK) {
                                const std::string image = cv::format("frames/frame_%06llu.png",static_cast<unsigned long long>(++serial));
                                samples.push_back({points,image,result->frame.frame_id});
                                ++revision; ++detectionEpoch; saved = false;
                                launch(result->frame.image,image); status = "Saving original frame and updating calibration";
                            } else status = "Rejected by official board-angle check; reduce tilt";
                            anchor.clear();
                        }
                    }
                    if (manual || busy) captureRequested = false;
                    if (!result->found && !busy) status = "Complete chessboard not found";
                }
                detection = std::move(result);
            }
            if (latest && !detector.busy() && submittedSequence != previewSequence) {
                DetectionJob job; job.frame = *latest; job.capture = capture;
                job.cameraMatrix = data->cameraMatrix.clone(); job.distCoeffs = data->distCoeffs.clone();
                job.receivedAt = lastFrameAt; job.epoch = detectionEpoch;
                detector.submit(std::move(job)); submittedSequence = previewSequence;
            }
            cv::imshow(windowName,dashboard(annotated,o,samples,solved,pending.valid(),automatic,undistorted,status,
                detection ? &*detection : nullptr,detector.busy(),latest ? latest->frame_id : 0));
            const int key = cv::waitKey(1) & 0xff;
            if (key==27 || key=='q' || cv::getWindowProperty(windowName,cv::WND_PROP_VISIBLE)==0) break;
            if (key=='a') { automatic = !automatic; anchor.clear(); captureRequested = false; ++detectionEpoch; }
            if (key=='u') undistorted = !undistorted;
            if (key==' ') { captureRequested = !pending.valid(); captureAfter = Clock::now(); }
            if (key=='s') {
                if (pending.valid() || solvedRevision!=revision || data->cameraMatrix.empty()) status = "Wait for a valid updated solution before saving";
                else { saveReport(o,solved); saved = true; status = "Saved intrinsics.yaml and source frames"; }
            }
            if ((key=='r' || key=='d') && !pending.valid()) {
                if (key=='d') samples.clear(); else if (!samples.empty()) samples.pop_back();
                ++revision; ++detectionEpoch; saved = false; anchor.clear(); captureRequested = false;
                // Rebuild from retained observations; never show stale confidence after undo/clear.
                solved = {}; solved.data = cv::makePtr<calib::calibrationData>(); solved.data->imageSize = data->imageSize;
                solved.flags = modelFlags(o.model); data->cameraMatrix.release(); data->distCoeffs.release();
                launch(); status = "Rebuilding retained sample set";
            }
        }
        if (pending.valid()) { solved = pending.get(); samples = solved.samples; solvedRevision = revision; saved = false; }
        if (!saved && solvedRevision==revision && solved.data && !solved.data->cameraMatrix.empty()) saveReport(o,solved);
        cv::destroyWindow(windowName);
    } catch (...) { cv::destroyWindow(windowName); throw; }
    return 0;
}
} // namespace livecalib

#ifndef DART_CALIBRATION_NO_MAIN
int main(int argc,char** argv) {
    try { const auto options = livecalib::parseOptions(argc,argv); return options.side.empty()?0:livecalib::run(options); }
    catch (const std::exception& e) { std::cerr << "Calibration failed: " << e.what() << '\n'; return 1; }
}
#endif
