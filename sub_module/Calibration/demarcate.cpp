// Single-camera BMP collection + offline manual ROI calibration.
// Upstream support code and license: opencv_interactive_calibration/README.md.
// ./build/demarcate --collect / --calibration --side=left
#include "../calibration_ui.hpp"
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
    enum class Mode { Collect, Calibration };
    Mode mode = Mode::Calibration;
    fs::path config = fs::path(DART_PROJECT_ROOT) / "config/dart.yaml";
    fs::path input, output;
    std::string side, model = "auto";
    cv::Size board{11, 8};
    double squareMm = 30, interval = 1.;
    int minViews = 10, maxViews = 30;
    bool automatic = true;
};

Options parseOptions(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h | | Show help without opening a camera}"
        "{collect | | Collect native BMP frames from the selected SDK camera}"
        "{calibration | | Offline manual ROI annotation and monocular calibration}"
        "{config | | dart.yaml path (collection camera settings only)}"
        "{side | | Required: left or right}"
        "{input | | BMP folder; default: sub_module/Calibration/Input/<side>}"
        "{output | | New calibration result directory}"
        "{interval | 1.0 | Seconds between automatic saved BMPs}"
        "{model | auto | auto, full, narrow or k1}"
        "{cols | 11 | Chessboard inner corner columns}"
        "{rows | 8 | Chessboard inner corner rows}"
        "{square-mm | 30 | Physical square side in millimetres}"
        "{min-views | 10 | Minimum views for quality evaluation}"
        "{max-views | 30 | Maximum selected views}"
        "{auto | true | Start collection with automatic frame saving enabled}");
    if (parser.has("help")) {
        parser.printMessage();
        std::cout << "./build/demarcate --collect --side=left [--interval=1]\n"
                     "./build/demarcate --calibration --side=left [--input=/path/to/BMPs]\n"
                     "Collection: A=auto/pause, SPACE=save next frame, Q/ESC=quit.\n"
                     "Offline: LMB=ROI, wheel=zoom, RMB click=zoom in, Shift+RMB=zoom out, RMB drag=pan, SPACE=accept+next,\n"
                     "N=skip, B/P=previous/next, R=clear ROI, D=remove, U=undo, S=save, +/-=zoom, [/]=list scroll, 0=reset zoom.\n";
        return {};
    }
    const std::vector<std::string> keys{"collect", "calibration", "config", "side", "input", "output", "interval",
        "model", "cols", "rows", "square-mm", "min-views", "max-views", "auto"};
    for (int i=1; i<argc; ++i) {
        const std::string arg=argv[i]; const auto equal=arg.find('=');
        if (arg.rfind("--",0)!=0) throw std::invalid_argument("Use --collect / --calibration and --name=value options");
        const auto key=arg.substr(2,equal==std::string::npos?equal:equal-2);
        if (std::find(keys.begin(),keys.end(),key)==keys.end()) throw std::invalid_argument("Unknown option: "+key);
        if ((key=="collect" || key=="calibration") ? equal!=std::string::npos : equal==std::string::npos)
            throw std::invalid_argument("Use bare mode flags and --name=value options");
    }
    if (parser.has("collect")==parser.has("calibration")) throw std::invalid_argument("Choose exactly one of --collect or --calibration");
    Options o;
    o.mode=parser.has("collect")?Options::Mode::Collect:Options::Mode::Calibration;
    if (parser.has("config")) o.config=parser.get<std::string>("config");
    o.side=parser.get<std::string>("side"); o.model=parser.get<std::string>("model");
    o.board={parser.get<int>("cols"),parser.get<int>("rows")}; o.squareMm=parser.get<double>("square-mm");
    o.interval=parser.get<double>("interval"); o.minViews=parser.get<int>("min-views"); o.maxViews=parser.get<int>("max-views");
    o.automatic=parser.get<bool>("auto");
    if (!parser.check()) {parser.printErrors();throw std::invalid_argument("Invalid command line");}
    if (o.side!="left" && o.side!="right") throw std::invalid_argument("--side=left or --side=right is required");
    if (o.model!="auto" && o.model!="full" && o.model!="narrow" && o.model!="k1") throw std::invalid_argument("Invalid --model");
    if (o.board.width<3 || o.board.height<3 || o.board.width>100 || o.board.height>100 ||
        !std::isfinite(o.squareMm) || o.squareMm<=0 || o.squareMm>10000 ||
        !std::isfinite(o.interval) || o.interval<.1 || o.interval>3600 ||
        o.minViews<3 || o.maxViews<o.minViews || o.maxViews>100)
        throw std::invalid_argument("Invalid board, interval or view count limits");
    o.input=parser.has("input")?fs::path(parser.get<std::string>("input")):fs::path(DART_PROJECT_ROOT)/"sub_module/Calibration/Input"/o.side;
    if (o.input.empty()) throw std::invalid_argument("--input cannot be empty");
    o.input=fs::absolute(o.input);
    if (parser.has("output")) {
        o.output=parser.get<std::string>("output");
        if (o.output.empty()) throw std::invalid_argument("--output cannot be empty");
    } else {
        const auto now=std::time(nullptr);std::tm local{};localtime_r(&now,&local);
        std::ostringstream name;name<<o.side<<'_'<<std::put_time(&local,"%Y%m%d_%H%M%S");
        o.output=fs::path(DART_PROJECT_ROOT)/"sub_module/Calibration/Results"/name.str();
    }
    o.output=fs::absolute(o.output);
    if (o.mode==Options::Mode::Calibration && fs::exists(o.output))
        throw std::invalid_argument("Output already exists; choose a new directory: "+o.output.string());
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
         << "source_directory" << o.input.string() << "capture_mode" << "offline_manual_roi_native_full_frames" << "calibration_model" << o.model << "calibration_flags" << solution.flags
         << "calibration_backend" << "OpenCV interactive calibration (offline ROI)"
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
    calibration_ui::text(canvas,value,{x,y},18,color);
}

void drawCoverage(cv::Mat& canvas, cv::Size size, const std::vector<Sample>& samples) {
    // Native sensor corner occupancy; preview scaling never enters calibration.
    cv::Mat coverage(200,400,CV_8UC3,cv::Scalar(40,40,40));
    if (!size.empty()) for (const auto& sample : samples) for (const auto& p : sample.points) {
        const int gx = std::clamp(static_cast<int>(p.x*10/size.width),0,9);
        const int gy = std::clamp(static_cast<int>(p.y*10/size.height),0,9);
        cv::rectangle(coverage,{gx*40,gy*20,40,20},{55,140,55},cv::FILLED);
    }
    for (int i=0;i<=10;++i) { cv::line(coverage,{i*40,0},{i*40,199},{85,85,85}); cv::line(coverage,{0,i*20},{399,i*20},{85,85,85}); }
    coverage.copyTo(canvas(cv::Rect(10,675,400,200)));
    text(canvas,"已覆盖区域（原始图像坐标）",10,660);
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
    drawCoverage(canvas,solved.data?solved.data->imageSize:annotated.size(),samples);
    text(canvas,status.substr(0,95),435,690,{80,220,255});
    text(canvas,"SPACE: capture | A: auto/manual | R: undo | D: clear",435,735);
    text(canvas,"S: save | U: undistort preview | ESC/Q: quit",435,770);
    text(canvas,"Pause the board; vary position, scale and tilt.",435,815);
    text(canvas,"RMS/quality OK does not validate stereo ranging.",435,855);
    return canvas;
}

} // namespace livecalib

#include "monocular_workflow.hpp"

#ifndef DART_CALIBRATION_NO_MAIN
int main(int argc,char** argv) {
    try { const auto options = livecalib::parseOptions(argc,argv); return options.side.empty()?0:(options.mode==livecalib::Options::Mode::Collect?livecalib::collect(options):livecalib::calibrate(options)); }
    catch (const std::exception& e) { std::cerr << "Calibration failed: " << e.what() << '\n'; return 1; }
}
#endif
