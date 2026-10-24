// 实时/离线双目标定：固定各自原生分辨率的 K/D，求 X_right = R * X_left + T。
// 默认实时：stereo_calibrate [--config=...] [--left-intrinsics=... --right-intrinsics=...]
// 保留离线入口：stereo_calibrate --offline [--full-frame] left_dir right_dir NEW_output_dir [left_yaml right_yaml]
// 用法：stereo_calibrate [left_dir right_dir output_dir [left_yaml right_yaml]]
// 全幅模式：stereo_calibrate --full-frame left_image_dir right_image_dir output_dir [left_yaml right_yaml]
// 输出目录必须不存在。退出码：0=通过基本质量检查，2=已保存诊断结果但质量不合格，1=程序错误。
// 采集前提：相机之间的位姿固定，且每对左右图中的棋盘必须保持同一位姿。
#include <opencv2/opencv.hpp>
#include "stereo_core.hpp"
#include "Acquisition/AcquireFrame.h"
#include "Acquisition/CameraInput.h"
#include "Acquisition/Quit.h"
#include "common/DartConfig.h"
#include <atomic>
#include <chrono>
#include <ctime>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace stereo_calibration {
struct Group {
    std::string label;
    cv::Point2f leftOffset;
    bool offsetConfirmed = true;
    cv::Size leftSize = kFrameSize;
};
// 在这里设置各组左相机拍摄时的真实 OffsetX/OffsetY。
// 右侧同名分组仅用于配对，右相机偏移始终为(0,0)。
std::vector<Group> kGroups{
    {"1500-1400", {1500.F, 1400.F}},
    {"2000-1300", {2000.F, 1300.F}},
    {"2600-1400", {2600.F, 1400.F}, false},
};
// 工程初筛阈值，并非真实测距精度的保证；不自动剔除高残差图对。
constexpr double kMaxRmsPx = 0.5, kMaxEpipolarP95Px = 1.0;
constexpr std::size_t kMinQualityPairs = 10;
constexpr int kReferenceRowHeight = 1180; // 校正图 + 原图，便于核查异常拟合或视野裁剪。

void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

Camera ReadCamera(const fs::path& path) {
    cv::FileStorage file(path.string(), cv::FileStorage::READ);
    Require(file.isOpened(), "Cannot read camera parameters: " + path.string());
    Camera c;
    file["camera_matrix"] >> c.K; file["distortion_coefficients"] >> c.D;
    file["image_width"] >> c.referenceSize.width; file["image_height"] >> c.referenceSize.height;
    Require(c.K.rows == 3 && c.K.cols == 3 && c.D.total() == 5 &&
            c.referenceSize.width > 0 && c.referenceSize.height > 0, "Invalid camera parameters");
    c.K.convertTo(c.K, CV_64F); c.D.convertTo(c.D, CV_64F);
    Require(cv::checkRange(c.K) && cv::checkRange(c.D) &&
            c.K.at<double>(0,0) > 0 && c.K.at<double>(1,1) > 0, "Nonfinite/invalid intrinsics");
    Require(std::abs(c.K.at<double>(2,2) - 1.) < 1e-9, "Invalid homogeneous intrinsics");
    return c;
}
cv::Mat ReadImage(const fs::path& path, cv::Size expected = kFrameSize, bool allowUnreadable = false) {
    auto image = cv::imread(path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty() && allowUnreadable) return {};
    Require(!image.empty(), "Cannot decode image: " + path.string());
    Require(image.size() == expected,
            "Expected raw " + std::to_string(expected.width) + "x" + std::to_string(expected.height) + " image: " + path.string());
    return image;
}
std::map<int, fs::path> IndexFiles(const fs::path& dir, const std::string& side) {
    Require(fs::is_directory(dir), "Missing image directory: " + dir.string());
    const std::regex pattern("^" + side + "([0-9]+)\\.(bmp|png|jpg|jpeg|tif|tiff)$", std::regex::icase);
    std::map<int, fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::smatch match;
        const auto name = entry.path().filename().string();
        if (entry.is_regular_file() && std::regex_match(name, match, pattern)) {
            const int number = std::stoi(match[1]);
            Require(files.emplace(number, entry.path()).second, "Duplicate frame number: " + name);
        }
    }
    return files;
}
bool Detect(const cv::Mat& image, std::vector<cv::Point2f>& points, bool realtime = false, int realtimeMaxEdge = 1600) {
    cv::Mat gray; cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    if (realtime) {
        // Try the cheap overview first. A second, finer search preserves distant-board cells on the left.
        // Keep both searches bounded: no exhaustive full-sensor fallback when the board is missing.
        for (const int edge : {1600,realtimeMaxEdge}) {
            const double scale=std::min(1.,static_cast<double>(edge)/std::max(gray.cols,gray.rows));
            cv::Mat coarse;
            if (scale<1.) cv::resize(gray,coarse,{},scale,scale,cv::INTER_AREA);
            else coarse=gray;
            points.clear();
            bool found=cv::findChessboardCorners(coarse,{kColumns,kRows},points,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
            if (!found && realtimeMaxEdge>1600 && (edge>=realtimeMaxEdge || scale==1.)) {
                // The sector-based detector recovers small boards missed by threshold/contour search.
                points.clear();
                found=cv::findChessboardCornersSB(coarse,{kColumns,kRows},points,cv::CALIB_CB_NORMALIZE_IMAGE);
            }
            if (found && points.size()==kColumns*kRows) {
                for (auto& p:points) {
                    p.x*=static_cast<float>(gray.cols)/coarse.cols;
                    p.y*=static_cast<float>(gray.rows)/coarse.rows;
                }
                int window=11;
                if (realtimeMaxEdge>1600) {
                    // An 11-pixel half-window crosses neighbouring corners on a small, distant board.
                    double spacing=std::numeric_limits<double>::infinity();
                    for (int row=0;row<kRows;++row) for (int col=0;col<kColumns;++col) {
                        const int i=row*kColumns+col;
                        if (col+1<kColumns) spacing=std::min(spacing,cv::norm(points[i+1]-points[i]));
                        if (row+1<kRows) spacing=std::min(spacing,cv::norm(points[i+kColumns]-points[i]));
                    }
                    window=std::clamp(static_cast<int>(std::floor(spacing*.4)),1,11);
                }
                cv::cornerSubPix(gray,points,{window,window},{-1,-1},
                    {cv::TermCriteria::COUNT | cv::TermCriteria::EPS,50,.001});
                return true;
            }
            if (edge>=realtimeMaxEdge || scale==1.) break;
        }
        points.clear(); return false;
    }
    cv::Rect search(0,0,gray.cols,gray.rows);
    // 全幅图先低分辨率定位棋盘，再回到原分辨率局部重检，最终角点未经缩放。
    if (gray.cols > 2000 || gray.rows > 2000) {
        const double scale=1600./std::max(gray.cols,gray.rows);
        cv::Mat coarse; cv::resize(gray,coarse,{},scale,scale,cv::INTER_AREA);
        std::vector<cv::Point2f> initial;
        if (cv::findChessboardCornersSB(coarse,{kColumns,kRows},initial,
            cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY)) {
            for (auto& p:initial) { p.x*=static_cast<float>(gray.cols)/coarse.cols; p.y*=static_cast<float>(gray.rows)/coarse.rows; }
            auto box=cv::boundingRect(initial);
            const int margin=std::max(64, std::max(box.width,box.height)/4);
            search=cv::Rect(box.x-margin,box.y-margin,box.width+2*margin,box.height+2*margin)&search;
        }
    }
    const auto region=gray(search);
    bool found = cv::findChessboardCornersSB(region, {kColumns, kRows}, points,
        cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
    if (!found) {
        points.clear();
        found = cv::findChessboardCorners(region, {kColumns, kRows}, points,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    }
    if (found) {
        for (auto& p:points) p+=cv::Point2f(static_cast<float>(search.x),static_cast<float>(search.y));
        cv::cornerSubPix(gray, points, {7,7}, {-1,-1},
            {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, .001});
    }
    return found && points.size() == kColumns * kRows;
}

struct Rectification { cv::Mat leftK, R1, R2, P1, P2, Q; };
double Quantile(std::vector<double> values, double q) {
    Require(!values.empty(), "No error samples");
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(q * (values.size()-1)))];
}
std::string Number(double v, int digits) {
    std::ostringstream s; s << std::fixed << std::setprecision(digits) << v; return s.str();
}

// 原尺寸双目校正后，缩小到每侧720x540，仅用于参考图显示。
cv::Mat ReferenceRow(const Pair& pair, const Rectification& rect,
                     const Camera& left, const Camera& right, const std::string& label) {
    cv::Mat mx, my, li, ri;
    cv::initUndistortRectifyMap(rect.leftK, left.D, rect.R1, rect.P1, kFrameSize, CV_32FC1, mx, my);
    cv::remap(ReadImage(pair.leftPath), li, mx, my, cv::INTER_LINEAR);
    cv::initUndistortRectifyMap(right.K, right.D, rect.R2, rect.P2, kFrameSize, CV_32FC1, mx, my);
    cv::remap(ReadImage(pair.rightPath), ri, mx, my, cv::INTER_LINEAR);
    std::vector<cv::Point2f> local = pair.left, pl, pr;
    for (auto& p : local) p -= kGroups[pair.group].leftOffset;
    cv::undistortPoints(local, pl, rect.leftK, left.D, rect.R1, rect.P1);
    cv::undistortPoints(pair.right, pr, right.K, right.D, rect.R2, rect.P2);
    cv::resize(li, li, {720,540}); cv::resize(ri, ri, {720,540});
    cv::Mat canvas(590, 1440, CV_8UC3, cv::Scalar(20,20,20));
    li.copyTo(canvas(cv::Rect(0,50,720,540))); ri.copyTo(canvas(cv::Rect(720,50,720,540)));
    const bool vertical = std::abs(rect.P2.at<double>(1,3)) > std::abs(rect.P2.at<double>(0,3));
    for (int y=50; y<590 && !vertical; y+=40) cv::line(canvas,{0,y},{1439,y},{40,190,40},1);
    if (vertical) for (int x=0; x<1440; x+=40) cv::line(canvas,{x,50},{x,589},{40,190,40},1);
    // 在各自面板中画点并裁剪，避免校正后落在画面外的角点串到另一侧。
    auto leftPanel=canvas(cv::Rect(0,50,720,540));
    auto rightPanel=canvas(cv::Rect(720,50,720,540));
    const cv::Rect2f frame(0,0,static_cast<float>(kFrameSize.width),static_cast<float>(kFrameSize.height));
    for (std::size_t i=0; i<pl.size(); ++i) {
        if (frame.contains(pl[i])) cv::circle(leftPanel,{cvRound(pl[i].x*.5),cvRound(pl[i].y*.5)},2,{0,0,255},1);
        if (frame.contains(pr[i])) cv::circle(rightPanel,{cvRound(pr[i].x*.5),cvRound(pr[i].y*.5)},2,{0,0,255},1);
    }
    cv::putText(canvas,label,{10,21},cv::FONT_HERSHEY_SIMPLEX,.55,{255,255,255},1);
    cv::putText(canvas,"LEFT | RIGHT   red: corresponding corners; green: epipolar guides; display scale=0.5",
                {10,42},cv::FONT_HERSHEY_SIMPLEX,.43,{200,200,200},1);
    // 诊断结果可能把棋盘校正到视野外；同时提供原图角点，保留可核查证据。
    auto rawLeft=ReadImage(pair.leftPath), rawRight=ReadImage(pair.rightPath);
    cv::drawChessboardCorners(rawLeft,{kColumns,kRows},local,true);
    cv::drawChessboardCorners(rawRight,{kColumns,kRows},pair.right,true);
    cv::resize(rawLeft,rawLeft,{720,540}); cv::resize(rawRight,rawRight,{720,540});
    cv::Mat rawCanvas(590,1440,CV_8UC3,cv::Scalar(20,20,20));
    rawLeft.copyTo(rawCanvas(cv::Rect(0,50,720,540)));
    rawRight.copyTo(rawCanvas(cv::Rect(720,50,720,540)));
    cv::putText(rawCanvas,"RAW INPUT (NOT RECTIFIED) - LEFT | RIGHT; detected corner order",
                {10,30},cv::FONT_HERSHEY_SIMPLEX,.55,{255,255,255},1);
    cv::Mat combined; cv::vconcat(canvas,rawCanvas,combined);
    return combined;
}
cv::Mat FitPanel(const cv::Mat& source) {
    const double scale=std::min(720./source.cols,540./source.rows);
    cv::Mat resized, panel(540,720,CV_8UC3,cv::Scalar(20,20,20));
    cv::resize(source,resized,{},scale,scale,cv::INTER_AREA);
    resized.copyTo(panel(cv::Rect((720-resized.cols)/2,(540-resized.rows)/2,resized.cols,resized.rows)));
    return panel;
}

// 左右尺寸不同：直接在各自去畸变像素坐标下画极线，不把全幅左图当作1440x1080。
// 展示时裁剪/等比缩放仅影响参考图，不修改K/D、原图或参与拟合的角点。
cv::Mat FullReferenceRow(const Pair& pair, const Camera& left, const Camera& right,
                         const cv::Mat& F, const std::string& label) {
    auto rawLeft=ReadImage(pair.leftPath,left.referenceSize), rawRight=ReadImage(pair.rightPath,right.referenceSize);
    cv::Mat li,ri; cv::undistort(rawLeft,li,left.K,left.D); cv::undistort(rawRight,ri,right.K,right.D);
    std::vector<cv::Point2f> pl,pr;
    cv::undistortPoints(pair.left,pl,left.K,left.D,cv::noArray(),left.K);
    cv::undistortPoints(pair.right,pr,right.K,right.D,cv::noArray(),right.K);
    std::vector<cv::Vec3f> linesLeft,linesRight;
    cv::computeCorrespondEpilines(pr,2,F,linesLeft); cv::computeCorrespondEpilines(pl,1,F,linesRight);
    const auto drawLine=[](cv::Mat& img,const cv::Vec3f& line,cv::Scalar color) {
        const double a=line[0],b=line[1],c=line[2];
        if (std::abs(b)>std::abs(a) && std::abs(b)>1e-12)
            cv::line(img,{0,cvRound(-c/b)},{img.cols-1,cvRound(-(c+a*(img.cols-1))/b)},color,1);
        else if(std::abs(a)>1e-12)
            cv::line(img,{cvRound(-c/a),0},{cvRound(-(c+b*(img.rows-1))/a),img.rows-1},color,1);
    };
    for (std::size_t i=0;i<pl.size();++i) {
        if (i%12==0 || i+1==pl.size()) { drawLine(li,linesLeft[i],{40,190,40}); drawLine(ri,linesRight[i],{40,190,40}); }
        cv::circle(li,pl[i],3,{0,0,255},1); cv::circle(ri,pr[i],3,{0,0,255},1);
    }
    const auto crop=[](const cv::Mat& image,const std::vector<cv::Point2f>& points) {
        auto box=cv::boundingRect(points); box=cv::Rect(box.x-80,box.y-80,box.width+160,box.height+160)&cv::Rect(0,0,image.cols,image.rows);
        Require(box.area()>0,"Undistorted board outside source image");
        return FitPanel(image(box));
    };
    cv::Mat canvas(kReferenceRowHeight,1440,CV_8UC3,cv::Scalar(20,20,20));
    crop(li,pl).copyTo(canvas(cv::Rect(0,50,720,540))); crop(ri,pr).copyTo(canvas(cv::Rect(720,50,720,540)));
    cv::putText(canvas,label,{10,21},cv::FONT_HERSHEY_SIMPLEX,.55,{255,255,255},1);
    cv::putText(canvas,"UNDISTORTED CROPS - LEFT | RIGHT; green: epilines; red: corners; NOT stereo-rectified",
                {10,42},cv::FONT_HERSHEY_SIMPLEX,.43,{200,200,200},1);
    cv::drawChessboardCorners(rawLeft,{kColumns,kRows},pair.left,true);
    cv::drawChessboardCorners(rawRight,{kColumns,kRows},pair.right,true);
    FitPanel(rawLeft).copyTo(canvas(cv::Rect(0,640,720,540))); FitPanel(rawRight).copyTo(canvas(cv::Rect(720,640,720,540)));
    cv::putText(canvas,"RAW INPUT - LEFT full sensor | RIGHT; aspect ratio preserved; display scales differ",
                {10,620},cv::FONT_HERSHEY_SIMPLEX,.5,{255,255,255},1);
    return canvas;
}
} // namespace stereo_calibration
using namespace stereo_calibration;

int OfflineMain(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0]
                << " [left_dir right_dir NEW_output_dir [left_intrinsics.yaml right_intrinsics.yaml]]\n"
                << "Use --full-frame before paths for LEFT 5472x3648 / RIGHT 1440x1080, both offsets zero.\n"
                << "Edit kGroups for LEFT capture offsets; RIGHT offsets are always zero.\n"
                << "Exit 0: basic checks passed; 2: diagnostics saved but quality failed; 1: error.\n";
            return 0;
        }
        const bool fullFrame=argc>1 && std::string(argv[1])=="--full-frame";
        if (fullFrame) { --argc; ++argv; Require(argc==4 || argc==6,"Full-frame mode requires explicit left/right image directories and NEW output directory"); }
        Require(argc == 1 || argc == 4 || argc == 6, "Use --help for arguments");
        const fs::path project(DART_PROJECT_ROOT);
        const fs::path leftDir = argc>=4 ? fs::path(argv[1]) : project/"Input/Calibration/left";
        const fs::path rightDir = argc>=4 ? fs::path(argv[2]) : project/"Input/Calibration/right";
        const fs::path output = argc>=4 ? fs::path(argv[3]) : project/"CalibrationResults/stereo_20261006";
        const fs::path leftYaml = argc==6 ? fs::path(argv[4]) : project/"CalibrationResults/monocular_20261006/left/intrinsics.yaml";
        const fs::path rightYaml = argc==6 ? fs::path(argv[5]) : project/"CalibrationResults/monocular_20261006/right/intrinsics.yaml";
        Require(!fs::exists(output), "Output already exists; choose a new directory: " + output.string());
        const Camera left = ReadCamera(leftYaml), right = ReadCamera(rightYaml);
        Require(left.referenceSize == cv::Size(5472,3648) && right.referenceSize == kFrameSize,
                "Expected LEFT full-sensor 5472x3648 and RIGHT raw 1440x1080 calibration");
        if (fullFrame) kGroups={{"full_frame",{0,0},true,left.referenceSize}};
        cv::setNumThreads(4);
        std::vector<Pair> pairs;
        std::vector<std::string> rejected;
        int discovered = 0;
        // 1. 分组后按数值编号配对；禁止将两组的left0/right0混到一起。
        for (std::size_t g=0; g<kGroups.size(); ++g) {
            const auto& group = kGroups[g];
            const auto ld=fullFrame?leftDir:leftDir/("left-"+group.label);
            const auto rd=fullFrame?rightDir:rightDir/("right-"+group.label);
            if (!fs::exists(ld) && !fs::exists(rd)) continue;
            Require(group.leftOffset.x>=0 && group.leftOffset.y>=0 &&
                    group.leftOffset.x+group.leftSize.width<=left.referenceSize.width &&
                    group.leftOffset.y+group.leftSize.height<=left.referenceSize.height, "LEFT ROI outside sensor");
            const auto lf=IndexFiles(ld,"left"), rf=IndexFiles(rd,"right");
            Require(lf.size()==rf.size(), "Unmatched frame counts in group "+group.label);
            for (const auto& item : lf) {
                Require(rf.count(item.first)==1, "Missing right frame "+std::to_string(item.first));
                ++discovered;
                Pair p{g,item.first,item.second,rf.at(item.first),{},{},false,0.};
                const auto li=ReadImage(p.leftPath,group.leftSize,true), ri=ReadImage(p.rightPath,kFrameSize,true);
                if (li.empty() || ri.empty()) {
                    const auto reason=group.label+"/"+std::to_string(item.first)+" (image decode failure: "+
                                      (li.empty()?"left":"right")+")";
                    rejected.push_back(reason); std::cerr << "Skipping " << reason << '\n';
                    continue;
                }
                if (!Detect(li,p.left) || !Detect(ri,p.right)) {
                    rejected.push_back(group.label+"/"+std::to_string(item.first));
                    continue;
                }
                // 当前近似同向安装的双目：纠正棋盘检测器可能产生的180度序号反转。
                // 不旋转图片；仅重排右侧角点索引，且在YAML中记录。
                const auto a=p.left[kColumns-1]-p.left[0], b=p.right[kColumns-1]-p.right[0];
                if (a.dot(b)<0) { std::reverse(p.right.begin(),p.right.end()); p.rightReversed=true; }
                for (auto& point:p.left) point+=group.leftOffset;
                pairs.push_back(std::move(p));
                std::cout << "Detected " << group.label << '/' << item.first << " (88+88 corners)" << std::endl;
            }
        }
        Require(pairs.size()>=3, "Need at least 3 valid image pairs with distinct board poses");
        std::vector<cv::Point3f> board;
        for (int y=0;y<kRows;++y) for (int x=0;x<kColumns;++x) board.emplace_back(x*kSquareMm,y*kSquareMm,0);
        std::vector<std::vector<cv::Point3f>> objects(pairs.size(),board);
        std::vector<std::vector<cv::Point2f>> lp,rp;
        for (const auto& p:pairs) { lp.push_back(p.left);rp.push_back(p.right); }

        // 2. 固定K/D，仅优化共同的R/T及每对棋盘位姿。
        // imageSize只影响内参初始化；FIX_INTRINSIC下左右图无需拥有相同参考尺寸。
        cv::Mat kl=left.K.clone(), dl=left.D.clone(), kr=right.K.clone(), dr=right.D.clone();
        cv::Mat R,T,E,F,errors;
        double rms=cv::stereoCalibrate(objects,lp,rp,kl,dl,kr,dr,left.referenceSize,
            R,T,E,F,errors,cv::CALIB_FIX_INTRINSIC,
            {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,150,1e-9});
        Require(std::isfinite(rms) && cv::checkRange(R) && cv::checkRange(T) && cv::norm(T)>1e-6,
                "Invalid stereo calibration result");
        Require(cv::norm(kl,left.K,cv::NORM_INF)==0 && cv::norm(kr,right.K,cv::NORM_INF)==0 &&
                cv::norm(dl,left.D,cv::NORM_INF)==0 && cv::norm(dr,right.D,cv::NORM_INF)==0,
                "Fixed intrinsics unexpectedly changed");

        const auto translationConstraint=RefineStereoTranslation(board,lp,rp,{left.K,left.D,right.K,right.D},R,T,E,F,errors,rms);

        // 3. 用去畸变后的原像素坐标计算左右点到极线的对称距离。
        std::vector<double> epipolarDistances;
        for (auto& p:pairs) {
            std::vector<cv::Point2f> ul,ur;
            cv::undistortPoints(p.left,ul,left.K,left.D,cv::noArray(),left.K);
            cv::undistortPoints(p.right,ur,right.K,right.D,cv::noArray(),right.K);
            double sum=0.;
            for (std::size_t i=0;i<ul.size();++i) {
                const cv::Mat a=(cv::Mat_<double>(3,1)<<ul[i].x,ul[i].y,1.);
                const cv::Mat b=(cv::Mat_<double>(3,1)<<ur[i].x,ur[i].y,1.);
                const cv::Mat lineR=F*a, lineL=F.t()*b, value=b.t()*lineR;
                const double numerator=std::abs(value.at<double>(0));
                for (const cv::Mat& line : {lineL,lineR}) {
                    const double distance=numerator/std::max(1e-15,std::hypot(line.at<double>(0),line.at<double>(1)));
                    epipolarDistances.push_back(distance);sum+=distance*distance;
                }
            }
            p.epipolarRms=std::sqrt(sum/(2*ul.size()));
        }
        const double epiRms=std::sqrt(std::inner_product(epipolarDistances.begin(),epipolarDistances.end(),
                                      epipolarDistances.begin(),0.)/epipolarDistances.size());
        const double epiP95=Quantile(epipolarDistances,.95);
        const bool residualChecksPassed=translationConstraint.converged && rms<=kMaxRmsPx && epiP95<=kMaxEpipolarP95Px && pairs.size()>=kMinQualityPairs;
        const bool offsetsConfirmed=std::all_of(pairs.begin(),pairs.end(),
            [](const Pair& p){ return kGroups[p.group].offsetConfirmed; });
        const bool passes=residualChecksPassed && offsetsConfirmed;

        // 4. 每个LEFT offset分别生成原始1440x1080图像的校正参数，R/T对所有组共用。
        std::map<std::size_t,Rectification> rectifications;
        for (const auto& p:pairs) {
            if (rectifications.count(p.group)) continue;
            Rectification rect;rect.leftK=left.K.clone();
            rect.leftK.at<double>(0,2)-=kGroups[p.group].leftOffset.x;
            rect.leftK.at<double>(1,2)-=kGroups[p.group].leftOffset.y;
            if (!fullFrame) cv::stereoRectify(rect.leftK,left.D,right.K,right.D,kFrameSize,R,T,
                             rect.R1,rect.R2,rect.P1,rect.P2,rect.Q,cv::CALIB_ZERO_DISPARITY,0.,kFrameSize);
            rectifications.emplace(p.group,std::move(rect));
        }
        fs::create_directories(output);
        cv::FileStorage yaml((output/"stereo.yaml").string(),cv::FileStorage::WRITE);
        Require(yaml.isOpened(), "Cannot write stereo.yaml");
        yaml << "status" << (!offsetsConfirmed?"provisional_offset_unconfirmed":
                             (passes?"passes_basic_checks_not_distance_validated":"needs_recapture_or_investigation"))
             << "quality_passed" << static_cast<int>(passes) << "distance_validated" << 0
             << "residual_checks_passed" << static_cast<int>(residualChecksPassed)
             << "left_offsets_confirmed" << static_cast<int>(offsetsConfirmed)
             << "capture_mode" << (fullFrame?"left_full_frame_right_native":"left_roi_right_native")
             << "rectification_parameters_available" << static_cast<int>(!fullFrame)
             << "transform_convention" << "X_right_mm = R * X_left_mm + T_mm"
             << "camera_axes" << "x_right,y_down,z_forward"
             << "intrinsics_policy" << "CALIB_FIX_INTRINSIC" << "exif_orientation" << "ignored"
             << "capture_requirement" << "fixed camera rig; identical board pose during both exposures of each pair"
             << "board_columns" << kColumns << "board_rows" << kRows << "square_size_mm" << kSquareMm
             << "left_intrinsics_source" << fs::absolute(leftYaml).string()
             << "right_intrinsics_source" << fs::absolute(rightYaml).string()
             << "left_image_directory" << fs::absolute(leftDir).string()
             << "right_image_directory" << fs::absolute(rightDir).string()
             << "left_reference_width" << left.referenceSize.width << "left_reference_height" << left.referenceSize.height
             << "right_reference_width" << right.referenceSize.width << "right_reference_height" << right.referenceSize.height
             << "K_left_reference" << left.K << "D_left" << left.D
             << "K_right" << right.K << "D_right" << right.D
             << "R" << R << "T_mm" << T << "baseline_mm" << cv::norm(T)
             << "E" << E << "F_reference" << F
             << "F_convention" << "undistorted_right_pixel^T * F_reference * undistorted_left_full_pixel = 0"
             << "rms_reprojection_error_px" << rms << "epipolar_symmetric_rms_px" << epiRms
             << "epipolar_distance_p95_px" << epiP95 << "quality_rms_limit_px" << kMaxRmsPx
             << "quality_epipolar_p95_limit_px" << kMaxEpipolarP95Px
             << "quality_minimum_pairs" << static_cast<int>(kMinQualityPairs)
             << "discovered_pairs" << discovered << "used_pairs" << static_cast<int>(pairs.size())
             << "rejected_detection_pairs" << rejected;
        WriteTranslationConstraint(yaml,translationConstraint);
        yaml << "pairs" << "[";
        for (std::size_t i=0;i<pairs.size();++i) {
            const auto& p=pairs[i];const auto& g=kGroups[p.group];
            yaml << "{" << "group" << g.label << "number" << p.number
                 << "left_image" << p.leftPath.string() << "right_image" << p.rightPath.string()
                 << "left_offset_x" << static_cast<int>(g.leftOffset.x) << "left_offset_y" << static_cast<int>(g.leftOffset.y)
                 << "left_offset_confirmed" << static_cast<int>(g.offsetConfirmed)
                 << "right_offset_x" << 0 << "right_offset_y" << 0
                 << "left_image_width" << g.leftSize.width << "left_image_height" << g.leftSize.height
                 << "right_image_width" << kFrameSize.width << "right_image_height" << kFrameSize.height
                 << "right_corner_order_reversed" << static_cast<int>(p.rightReversed)
                 << "image_points_left_reference" << p.left << "image_points_right" << p.right
                 << "left_rms_px" << errors.at<double>(static_cast<int>(i),0)
                 << "right_rms_px" << errors.at<double>(static_cast<int>(i),1)
                 << "epipolar_rms_px" << p.epipolarRms << "}";
        }
        yaml << "]" << "rectification_by_left_roi" << "[";
        for (const auto& item:rectifications) {
            if (fullFrame) continue; // 原生尺寸不同，保留K/D/R/T/E/F；不输出假定同尺寸的P/Q。
            const auto& g=kGroups[item.first];const auto& rect=item.second;
            yaml << "{" << "group" << g.label << "left_offset_x" << static_cast<int>(g.leftOffset.x)
                 << "left_offset_confirmed" << static_cast<int>(g.offsetConfirmed)
                 << "left_offset_y" << static_cast<int>(g.leftOffset.y) << "right_offset_x" << 0 << "right_offset_y" << 0
                 << "image_width" << kFrameSize.width << "image_height" << kFrameSize.height
                 << "K_left_roi" << rect.leftK << "R1" << rect.R1 << "R2" << rect.R2
                 << "P1" << rect.P1 << "P2" << rect.P2 << "Q" << rect.Q
                 << "rectification_alpha" << 0. << "}";
        }
        yaml << "]";

        // 5. 一张参考图包含各组中位误差样本及全局最差样本，避免只展示最好的一对。
        std::vector<std::size_t> referenceIndices;
        for (const auto& group:rectifications) {
            std::vector<std::size_t> ids;
            for (std::size_t i=0;i<pairs.size();++i) if (pairs[i].group==group.first) ids.push_back(i);
            std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return pairs[a].epipolarRms<pairs[b].epipolarRms;});
            referenceIndices.push_back(ids[ids.size()/2]);
        }
        const auto worst=static_cast<std::size_t>(std::max_element(pairs.begin(),pairs.end(),
            [](const auto& a,const auto& b){return a.epipolarRms<b.epipolarRms;})-pairs.begin());
        if (std::find(referenceIndices.begin(),referenceIndices.end(),worst)==referenceIndices.end()) referenceIndices.push_back(worst);
        cv::Mat reference(75+kReferenceRowHeight*static_cast<int>(referenceIndices.size()),1440,CV_8UC3,cv::Scalar(20,20,20));
        cv::putText(reference,!offsetsConfirmed?"OFFSET UNCONFIRMED - DIAGNOSTIC ONLY":
                    (passes?"BASIC CHECKS PASSED - DISTANCE NOT VALIDATED":"QUALITY FAILED - DIAGNOSTIC ONLY"),
                    {10,29},cv::FONT_HERSHEY_SIMPLEX,.8,passes?cv::Scalar(0,220,0):cv::Scalar(0,80,255),2);
        cv::putText(reference,"Stereo RMS="+Number(rms)+" px; epipolar P95="+Number(epiP95)+" px; pairs="+std::to_string(pairs.size()),
                    {10,57},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},1);
        yaml << "reference_image" << "reference.png"
             << "reference_layout" << (fullFrame?
                 "per selected pair: independently scaled undistorted board crops with epilines, then native raw inputs; not stereo-rectified":
                 "per selected pair: rectified view followed by raw input with detected corner order")
             << "reference_pairs" << "[";
        for (std::size_t row=0;row<referenceIndices.size();++row) {
            const auto i=referenceIndices[row];const auto& p=pairs[i];
            const auto label=kGroups[p.group].label+" / pair "+std::to_string(p.number)+
                (i==worst?" (WORST)":" (GROUP MEDIAN)")+"  epipolar RMS="+Number(p.epipolarRms)+" raw px";
            const auto rowImage=fullFrame?FullReferenceRow(p,left,right,F,label):
                                         ReferenceRow(p,rectifications.at(p.group),left,right,label);
            rowImage.copyTo(
                reference(cv::Rect(0,75+kReferenceRowHeight*static_cast<int>(row),1440,kReferenceRowHeight)));
            yaml << "{" << "group" << kGroups[p.group].label << "number" << p.number << "}";
        }
        yaml << "]";yaml.release();
        Require(cv::imwrite((output/"reference.png").string(),reference),"Cannot save reference image");
        std::cout << "Unconstrained T_mm=" << translationConstraint.initialT.t()
                  << " RMS=" << translationConstraint.initialRms << " px; bounded T_mm=" << T.t()
                  << " RMS=" << rms << " px; converged=" << translationConstraint.converged
                  << " at_limit=" << translationConstraint.atLimit << '\n';
        std::cout << "RMS: " << rms << " px\nEpipolar P95: " << epiP95
                  << " px\nBaseline (diagnostic unless validated): " << cv::norm(T) << " mm\nR:\n" << R
                  << "\nT_mm:\n" << T << "\nResidual checks passed: " << residualChecksPassed
                  << "\nLEFT offsets confirmed: " << offsetsConfirmed
                  << "\nQuality passed: " << passes << "\nSaved to: " << output << '\n';
        return passes?0:2;
    } catch (const std::exception& e) { std::cerr << "Stereo calibration failed: " << e.what() << '\n'; return 1; }
}


namespace stereo_live {
using Clock = std::chrono::steady_clock;
constexpr const char* kWindow = "Live stereo calibration - native coordinates";

Options Parse(int argc, char** argv) {
    cv::CommandLineParser parser(argc,argv,
        "{help | | Show live/offline usage without opening cameras}"
        "{config | | dart.yaml, default: project config/dart.yaml}"
        "{left-intrinsics | | Optional left monocular intrinsics.yaml; otherwise cameras.left}"
        "{right-intrinsics | | Optional right monocular intrinsics.yaml; otherwise cameras.right}"
        "{output | | NEW session directory; existing directories are rejected}"
        "{pair-ms | -1 | Maximum host receive timestamp difference; default from dart.yaml}"
        "{hold | 0.8 | Both boards must remain still for this many seconds}"
        "{min-pairs | 10 | Minimum pairs for basic quality checks}"
        "{max-pairs | 30 | Stop accepting at this count; R/X can remove pairs}"
        "{auto | true | Enable automatic capture after stable, diverse poses}");
    Options o;
    if (parser.has("help")) {
        parser.printMessage();
        std::cout << "Live: stereo_calibrate [--config=...] [--left-intrinsics=... --right-intrinsics=...]\n"
                  << "Board: 11x8 INNER corners, 30 mm squares. Both cameras must see the same stationary board.\n"
                  << "Offline: stereo_calibrate --offline --full-frame LEFT_DIR RIGHT_DIR NEW_OUTPUT [LEFT_YAML RIGHT_YAML]\n"
                  << "Keys: SPACE=capture (stable required), A=auto/manual, R=undo, X=remove worst pair, D=clear, S=save, E=epilines, ESC/Q=exit.\n"
                  << "Exit: 0=basic checks passed; 2=diagnostics/incomplete session; 1=error.\n";
        o.help=true; return o;
    }
    if (parser.has("config")) o.config=parser.get<std::string>("config");
    o.leftIntrinsics=parser.get<std::string>("left-intrinsics");
    o.rightIntrinsics=parser.get<std::string>("right-intrinsics");
    o.pairMs=parser.get<int>("pair-ms"); o.holdSeconds=parser.get<double>("hold");
    o.minPairs=parser.get<int>("min-pairs"); o.maxPairs=parser.get<int>("max-pairs");
    o.automatic=parser.get<bool>("auto");
    if (!parser.check()) { parser.printErrors(); throw std::invalid_argument("Invalid command line"); }
    const std::vector<std::string> keys{"config","left-intrinsics","right-intrinsics","output","pair-ms","hold","min-pairs","max-pairs","auto"};
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        Require(arg.rfind("--",0)==0 && arg.find('=')!=std::string::npos,"Use --name=value options");
        const auto key=arg.substr(2,arg.find('=')-2);
        Require(std::find(keys.begin(),keys.end(),key)!=keys.end(),"Unknown option: "+key);
    }
    Require(!o.config.empty() && std::isfinite(o.holdSeconds) && o.holdSeconds>=0.1 && o.holdSeconds<=60 &&
            o.pairMs>=-1 && o.pairMs<=1000 && o.minPairs>=3 && o.maxPairs>=o.minPairs && o.maxPairs<=100,
            "Invalid config, pair time, hold duration or sample limits");
    if (parser.has("output")) {
        o.output=parser.get<std::string>("output"); Require(!o.output.empty(),"Output cannot be empty");
    } else {
        const auto now=std::time(nullptr); std::tm local{}; localtime_r(&now,&local);
        std::ostringstream name; name<<"stereo_live_"<<std::put_time(&local,"%Y%m%d_%H%M%S");
        o.output=fs::path(DART_PROJECT_ROOT)/"CalibrationResults"/name.str();
    }
    o.output=fs::absolute(o.output);
    Require(!fs::exists(o.output),"Output already exists; choose a NEW directory: "+o.output.string());
    return o;
}

Camera Intrinsics(const CameraParams& p, const fs::path& overrideFile) {
    Camera camera=overrideFile.empty()?Camera{p.cameraMatrix.clone(),p.distCoeffs.clone(),{p.width,p.height}}:ReadCamera(overrideFile);
    Require(camera.referenceSize==cv::Size(p.width,p.height),"Intrinsic image size differs from selected camera SDK settings");
    Require(camera.K.type()==CV_64FC1 && camera.K.rows==3 && camera.K.cols==3 &&
            camera.D.channels()==1 && camera.D.total()==5 && cv::checkRange(camera.K) && cv::checkRange(camera.D) &&
            camera.K.at<double>(0,0)>0 && camera.K.at<double>(1,1)>0 &&
            std::abs(camera.K.at<double>(0,1))<1e-12 && std::abs(camera.K.at<double>(1,0))<1e-12 &&
            std::abs(camera.K.at<double>(2,0))<1e-12 && std::abs(camera.K.at<double>(2,1))<1e-12 &&
            std::abs(camera.K.at<double>(2,2)-1)<1e-12,"Invalid standard pinhole K / five-coefficient Brown distortion");
    camera.D=camera.D.reshape(1,5).clone(); return camera;
}

std::uint64_t TimestampDifference(std::int64_t a, std::int64_t b) {
    return a>=b?static_cast<std::uint64_t>(a)-static_cast<std::uint64_t>(b)
               :static_cast<std::uint64_t>(b)-static_cast<std::uint64_t>(a);
}

// Leave GetFrame's one-attempt contract unchanged. This background consumer drains stale pairs.
class LatestPair {
    CameraInput input_;
    std::atomic<bool> stop_{false};
    std::thread matcher_;
    std::mutex mutex_;
    std::optional<StereoFrame> latest_;
    std::exception_ptr error_;
public:
    LatestPair(const DartCongfig& config,std::int64_t maxMs)
        :input_(config.leftCamera,config.rightCamera,config.frameQueue.capacity) {
        input_.StartCamera();
        matcher_=std::thread([this,maxMs] {
            try {
                while(!stop_.load()) {
                    auto pair=input_.frameQueue().GetFrame(maxMs);
                    if(pair) { std::lock_guard<std::mutex> lock(mutex_); latest_=std::move(pair); }
                    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } catch(...) { std::lock_guard<std::mutex> lock(mutex_); error_=std::current_exception(); }
        });
    }
    void stop() { stop_.store(true); if(matcher_.joinable()) matcher_.join(); input_.StopCamera(); }
    ~LatestPair() { stop(); }
    std::optional<StereoFrame> take() {
        std::lock_guard<std::mutex> lock(mutex_); if(error_) std::rethrow_exception(error_);
        auto result=std::move(latest_); latest_.reset(); return result;
    }
};

// Same near-parallel-rig convention as the existing offline calibration, with recorded reordering.
bool AlignRight(const std::vector<cv::Point2f>& left,std::vector<cv::Point2f>& right) {
    Require(left.size()==kColumns*kRows && right.size()==left.size(),"Incomplete corresponding checkerboards");
    if((left[kColumns-1]-left[0]).dot(right[kColumns-1]-right[0])<0) {
        std::reverse(right.begin(),right.end()); return true;
    }
    return false;
}
struct Observation {
    StereoFrame frame;
    std::vector<cv::Point2f> left,right;
    bool leftFound=false,rightFound=false,rightReversed=false;
};
Observation Observe(StereoFrame frame) {
    Observation result; result.frame=std::move(frame);
    result.leftFound=Detect(result.frame.left.image,result.left,true,3200);
    result.rightFound=Detect(result.frame.right.image,result.right,true);
    if(result.leftFound && result.rightFound) result.rightReversed=AlignRight(result.left,result.right);
    return result;
}


double Motion(const std::vector<cv::Point2f>& a,const std::vector<cv::Point2f>& b) {
    if(a.size()!=b.size() || a.empty()) return std::numeric_limits<double>::infinity();
    double forward=0,reverse=0;
    for(std::size_t i=0;i<a.size();++i) {
        const auto da=a[i]-b[i],db=a[i]-b[a.size()-1-i]; forward+=da.dot(da); reverse+=db.dot(db);
    }
    return std::sqrt(std::min(forward,reverse)/a.size());
}
bool Diverse(const Observation& current,const std::vector<LivePair>& pairs,const Camera& left,const Camera& right) {
    const double dl=std::hypot(left.referenceSize.width,left.referenceSize.height);
    const double dr=std::hypot(right.referenceSize.width,right.referenceSize.height);
    for(const auto& previous:pairs)
        if(Motion(current.left,previous.pair.left)/dl<0.008 && Motion(current.right,previous.pair.right)/dr<0.008) return false;
    return true;
}

// Manual capture also obeys this gate: receive-time proximity cannot establish exposure synchronization.
class Stability {
    std::vector<cv::Point2f> left_,right_;
    std::int64_t first_=0,last_=0;
    int count_=0;
    double elapsed_=0;
public:
    void reset() { left_.clear(); right_.clear(); count_=0; elapsed_=0; first_=last_=0; }
    double elapsed() const { return elapsed_; }
    bool update(const Observation& o,const Camera& left,const Camera& right,double seconds,std::int64_t pairMs) {
        if(!o.leftFound || !o.rightFound || TimestampDifference(o.frame.left.timestamp_ms,o.frame.right.timestamp_ms)>static_cast<std::uint64_t>(pairMs)) {
            reset(); return false;
        }
        const auto time=std::max(o.frame.left.timestamp_ms,o.frame.right.timestamp_ms);
        const double dl=std::hypot(left.referenceSize.width,left.referenceSize.height);
        const double dr=std::hypot(right.referenceSize.width,right.referenceSize.height);
        if(count_==0 || time<=last_ || TimestampDifference(time,last_)>1500 ||
           Motion(o.left,left_)/dl>0.0006 || Motion(o.right,right_)/dr>0.0006) {
            left_=o.left; right_=o.right; first_=time; count_=1;
        } else ++count_;
        last_=time; elapsed_=static_cast<double>(time-first_)/1000;
        return count_>=3 && elapsed_>=seconds;
    }
};


Epipolar MeasureEpipolar(const std::vector<cv::Point2f>& leftPoints,const std::vector<cv::Point2f>& rightPoints,
                        const Camera& left,const Camera& right,const cv::Mat& fundamental) {
    Require(!leftPoints.empty() && leftPoints.size()==rightPoints.size(),"Missing corresponding points");
    std::vector<cv::Point2d> pl(leftPoints.begin(),leftPoints.end()),pr(rightPoints.begin(),rightPoints.end()),ul,ur;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,1e-12);
    cv::undistortPoints(pl,ul,left.K,left.D,cv::noArray(),left.K,criteria);
    cv::undistortPoints(pr,ur,right.K,right.D,cv::noArray(),right.K,criteria);
    const cv::Matx33d F=fundamental;
    Epipolar result; double sum=0;
    for(std::size_t i=0;i<ul.size();++i) {
        const cv::Vec3d a(ul[i].x,ul[i].y,1),b(ur[i].x,ur[i].y,1);
        const cv::Vec3d lineRight=F*a,lineLeft=F.t()*b;
        const double nr=std::hypot(lineRight[0],lineRight[1]),nl=std::hypot(lineLeft[0],lineLeft[1]);
        Require(nr>1e-15 && nl>1e-15,"Degenerate epipolar lines");
        const double numerator=std::abs(b.dot(lineRight));
        const double el=numerator/nl,er=numerator/nr;
        Require(std::isfinite(el) && std::isfinite(er),"Non-finite epipolar residual");
        sum+=el*el+er*er; result.maxSide.push_back(std::max(el,er));
    }
    result.rms=std::sqrt(sum/(2*ul.size())); result.p95=Quantile(result.maxSide,.95);
    result.maximum=*std::max_element(result.maxSide.begin(),result.maxSide.end()); return result;
}

Solution Solve(std::vector<LivePair> pairs,const Camera& left,const Camera& right,int minimumPairs) {
    Solution result; result.pairs=std::move(pairs); if(result.pairs.size()<3) return result;
    std::vector<cv::Point3f> board;
    for(int y=0;y<kRows;++y) for(int x=0;x<kColumns;++x) board.emplace_back(x*kSquareMm,y*kSquareMm,0);
    std::vector<std::vector<cv::Point3f>> objects(result.pairs.size(),board);
    std::vector<std::vector<cv::Point2f>> lp,rp;
    for(const auto& p:result.pairs) {
        Require(p.pair.left.size()==board.size() && p.pair.right.size()==board.size(),"Incomplete stereo observations");
        lp.push_back(p.pair.left); rp.push_back(p.pair.right);
    }
    cv::Mat kl=left.K.clone(),dl=left.D.clone(),kr=right.K.clone(),dr=right.D.clone();
    result.rms=cv::stereoCalibrate(objects,lp,rp,kl,dl,kr,dr,left.referenceSize,
        result.R,result.T,result.E,result.F,result.errors,cv::CALIB_FIX_INTRINSIC,
        {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,150,1e-9});
    Require(std::isfinite(result.rms) && cv::checkRange(result.R) && cv::checkRange(result.T) && cv::checkRange(result.F) &&
            cv::norm(result.T)>1e-6,"Invalid stereo calibration");
    Require(cv::norm(kl,left.K,cv::NORM_INF)==0 && cv::norm(dl,left.D,cv::NORM_INF)==0 &&
            cv::norm(kr,right.K,cv::NORM_INF)==0 && cv::norm(dr,right.D,cv::NORM_INF)==0,"Fixed intrinsics changed unexpectedly");
    result.translationConstraint=RefineStereoTranslation(board,lp,rp,{left.K,left.D,right.K,right.D},
        result.R,result.T,result.E,result.F,result.errors,result.rms);
    Require(result.errors.rows==static_cast<int>(result.pairs.size()) && result.errors.cols==2 && cv::checkRange(result.errors),"Invalid per-camera pair errors");
    std::vector<double> maxErrors; double squared=0;
    for(auto& pair:result.pairs) {
        auto error=MeasureEpipolar(pair.pair.left,pair.pair.right,left,right,result.F);
        squared+=error.rms*error.rms*error.maxSide.size(); pair.pair.epipolarRms=error.rms;
        maxErrors.insert(maxErrors.end(),error.maxSide.begin(),error.maxSide.end());
        result.perPair.push_back(std::move(error));
    }
    result.epiRms=std::sqrt(squared/maxErrors.size()); result.epiP95=Quantile(maxErrors,.95);
    result.epiMax=*std::max_element(maxErrors.begin(),maxErrors.end());
    result.passed=result.translationConstraint.converged && result.pairs.size()>=static_cast<std::size_t>(minimumPairs) && result.rms<=kMaxRmsPx && result.epiP95<=kMaxEpipolarP95Px;
    return result;
}
std::size_t Worst(const Solution& s) {
    Require(!s.R.empty() && !s.perPair.empty(),"No valid solution to select a worst pair");
    std::size_t worst=0; double maximum=-1;
    for(std::size_t i=0;i<s.pairs.size();++i) {
        const double score=std::max({s.perPair[i].p95,s.errors.at<double>(static_cast<int>(i),0),s.errors.at<double>(static_cast<int>(i),1)});
        if(score>maximum) { maximum=score; worst=i; }
    }
    return worst;
}

void Save(const Options& o,const Camera& left,const Camera& right,const Solution& s) {
    Require(!s.R.empty(),"Need at least 3 solved pairs before saving stereo.yaml");
    std::vector<std::size_t> referenceIndices;
    std::vector<std::size_t> order(s.pairs.size()); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto a,auto b){return s.perPair[a].p95<s.perPair[b].p95;});
    referenceIndices.push_back(order[order.size()/2]);
    const auto worst=Worst(s); if(worst!=referenceIndices.front()) referenceIndices.push_back(worst);
    cv::Mat reference(75+kReferenceRowHeight*static_cast<int>(referenceIndices.size()),1440,CV_8UC3,cv::Scalar(20,20,20));
    cv::putText(reference,s.passed?"BASIC CHECKS PASSED - DISTANCE NOT VALIDATED":"QUALITY FAILED / INSUFFICIENT PAIRS - DIAGNOSTIC ONLY",
        {10,27},cv::FONT_HERSHEY_SIMPLEX,.65,s.passed?cv::Scalar(0,220,0):cv::Scalar(0,80,255),1);
    cv::putText(reference,"RMS="+Number(s.rms)+" px; max-side epipolar P95="+Number(s.epiP95)+" px; pairs="+std::to_string(s.pairs.size()),
        {10,57},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},1);
    for(std::size_t row=0;row<referenceIndices.size();++row) {
        const auto i=referenceIndices[row];
        const auto label="Pair "+std::to_string(s.pairs[i].pair.number)+(i==worst?" WORST":" MEDIAN")+"; epipolar P95="+Number(s.perPair[i].p95)+" px";
        FullReferenceRow(s.pairs[i].pair,left,right,s.F,label).copyTo(reference(cv::Rect(0,75+kReferenceRowHeight*static_cast<int>(row),1440,kReferenceRowHeight)));
    }
    Require(cv::imwrite((o.output/"reference.pending.png").string(),reference),"Cannot write reference image");
    cv::FileStorage yaml((o.output/"stereo.pending.yaml").string(),cv::FileStorage::WRITE);
    Require(yaml.isOpened(),"Cannot write stereo report");
    yaml<<"status"<<(s.passed?"passes_basic_checks_not_distance_validated":"needs_recapture_or_investigation")
        <<"quality_passed"<<static_cast<int>(s.passed)<<"distance_validated"<<0
        <<"capture_mode"<<o.captureMode<<"rectification_parameters_available"<<0
        <<"transform_convention"<<"X_right_mm = R * X_left_mm + T_mm"<<"camera_axes"<<"x_right,y_down,z_forward"
        <<"intrinsics_policy"<<"CALIB_FIX_INTRINSIC"<<"config_source"<<fs::absolute(o.config).string()
        <<"left_intrinsics_source"<<(o.leftIntrinsics.empty()?"dart.yaml:cameras.left":fs::absolute(o.leftIntrinsics).string())
        <<"right_intrinsics_source"<<(o.rightIntrinsics.empty()?"dart.yaml:cameras.right":fs::absolute(o.rightIntrinsics).string())
        <<"board_columns"<<kColumns<<"board_rows"<<kRows<<"square_size_mm"<<kSquareMm
        <<"left_reference_width"<<left.referenceSize.width<<"left_reference_height"<<left.referenceSize.height
        <<"right_reference_width"<<right.referenceSize.width<<"right_reference_height"<<right.referenceSize.height
        <<"K_left_reference"<<left.K<<"D_left"<<left.D<<"K_right"<<right.K<<"D_right"<<right.D
        <<"R"<<s.R<<"T_mm"<<s.T<<"E"<<s.E<<"F_reference"<<s.F<<"baseline_mm"<<cv::norm(s.T)
        <<"F_convention"<<"undistorted_right_pixel^T * F_reference * undistorted_left_full_pixel = 0"
        <<"rms_reprojection_error_px"<<s.rms<<"epipolar_symmetric_rms_px"<<s.epiRms
        <<"epipolar_distance_p95_px"<<s.epiP95<<"epipolar_max_px"<<s.epiMax
        <<"epipolar_p95_definition"<<"P95 of max(left point-to-line distance, right point-to-line distance) per correspondence"
        <<"quality_rms_limit_px"<<kMaxRmsPx<<"quality_epipolar_p95_limit_px"<<kMaxEpipolarP95Px
        <<"quality_minimum_pairs"<<o.minPairs<<"used_pairs"<<static_cast<int>(s.pairs.size())
        <<"max_host_timestamp_diff_ms"<<static_cast<int>(o.pairMs)<<"stationary_hold_seconds"<<o.holdSeconds
        <<"timestamp_semantics"<<"host steady_clock receive time; not exposure synchronization"
        <<"capture_requirement"<<"fixed rig; same stationary board pose during both exposures"
        <<"corner_alignment"<<"near-parallel rig: reverse right 180-degree ordering if row directions oppose"
        <<"reference_image"<<"reference.png"<<"rectification_by_left_roi"<<"["<<"]";
    WriteTranslationConstraint(yaml,s.translationConstraint);
    yaml<<"pairs"<<"[";
    for(std::size_t i=0;i<s.pairs.size();++i) {
        const auto& p=s.pairs[i];
        yaml<<"{"<<"number"<<p.pair.number<<"left_image"<<p.pair.leftPath.string()<<"right_image"<<p.pair.rightPath.string()
            <<"left_frame_id"<<std::to_string(p.leftId)<<"right_frame_id"<<std::to_string(p.rightId)
            <<"left_receive_timestamp_ms"<<std::to_string(p.leftTime)<<"right_receive_timestamp_ms"<<std::to_string(p.rightTime)
            <<"frame_timestamps_available"<<static_cast<int>(p.timestampsKnown)
            <<"host_timestamp_diff_ms"<<(p.timestampsKnown?static_cast<int>(TimestampDifference(p.leftTime,p.rightTime)):-1)
            <<"left_offset_x"<<0<<"left_offset_y"<<0<<"right_offset_x"<<0<<"right_offset_y"<<0
            <<"right_corner_order_reversed"<<static_cast<int>(p.pair.rightReversed)
            <<"image_points_left_reference"<<p.pair.left<<"image_points_right"<<p.pair.right
            <<"left_rms_px"<<s.errors.at<double>(static_cast<int>(i),0)<<"right_rms_px"<<s.errors.at<double>(static_cast<int>(i),1)
            <<"epipolar_rms_px"<<s.perPair[i].rms<<"epipolar_p95_px"<<s.perPair[i].p95<<"}";
    }
    yaml<<"]"; yaml.release();
    fs::rename(o.output/"reference.pending.png",o.output/"reference.png");
    fs::rename(o.output/"stereo.pending.yaml",o.output/"stereo.yaml");
    std::cout<<"Saved "<<o.output/"stereo.yaml"<<"; pairs="<<s.pairs.size()<<" RMS="<<s.rms
             <<" px epipolar P95="<<s.epiP95<<" px baseline="<<cv::norm(s.T)<<" mm; quality="<<s.passed<<'\n';
}


cv::Mat Preview(const cv::Mat& raw,const std::vector<cv::Point2f>& points,const Camera& camera,
                const std::vector<cv::Point2f>& peerPoints,const Camera& peerCamera,
                const cv::Mat& F,bool isLeft,bool epilines) {
    cv::Mat pane(480,720,CV_8UC3,cv::Scalar(30,30,30)); if(raw.empty()) return pane;
    const double scale=std::min(720.0/raw.cols,480.0/raw.rows);
    cv::Mat image; cv::resize(raw,image,{},scale,scale,cv::INTER_AREA);
    const double sx=static_cast<double>(image.cols)/raw.cols,sy=static_cast<double>(image.rows)/raw.rows;
    std::vector<cv::Point2f> displayed=points;
    if(epilines && !F.empty()) {
        cv::Mat k=camera.K.clone(); k.row(0)*=sx; k.row(1)*=sy;
        cv::Mat undistorted; cv::undistort(image,undistorted,k,camera.D); image=undistorted;
        if(!points.empty()) cv::undistortPoints(points,displayed,camera.K,camera.D,cv::noArray(),camera.K);
        if(!peerPoints.empty()) {
            std::vector<cv::Point2f> peer; cv::undistortPoints(peerPoints,peer,peerCamera.K,peerCamera.D,cv::noArray(),peerCamera.K);
            std::vector<cv::Vec3f> lines; cv::computeCorrespondEpilines(peer,isLeft?2:1,F,lines);
            for(std::size_t i=0;i<lines.size();i+=12) {
                const double a=lines[i][0]/sx,b=lines[i][1]/sy,c=lines[i][2];
                if(std::abs(b)>=std::abs(a) && std::abs(b)>1e-12)
                    cv::line(image,{0,cvRound(-c/b)},{image.cols-1,cvRound(-(c+a*(image.cols-1))/b)},{30,210,30},1,cv::LINE_AA);
                else if(std::abs(a)>1e-12)
                    cv::line(image,{cvRound(-c/a),0},{cvRound(-(c+b*(image.rows-1))/a),image.rows-1},{30,210,30},1,cv::LINE_AA);
            }
        }
    }
    for(auto& point:displayed) { point.x*=sx; point.y*=sy; }
    if(displayed.size()==kColumns*kRows) cv::drawChessboardCorners(image,{kColumns,kRows},displayed,true);
    image.copyTo(pane(cv::Rect((pane.cols-image.cols)/2,(pane.rows-image.rows)/2,image.cols,image.rows)));
    return pane;
}
void Label(cv::Mat& canvas,const std::string& value,int x,int y,cv::Scalar color={230,230,230}) {
    cv::putText(canvas,value,{x,y},cv::FONT_HERSHEY_SIMPLEX,.52,color,1,cv::LINE_AA);
}
void Coverage(cv::Mat& canvas,const std::vector<LivePair>& pairs,const Camera& camera,bool isLeft,int x,int y) {
    cv::Mat grid(180,360,CV_8UC3,cv::Scalar(40,40,40));
    for(const auto& pair:pairs) for(const auto& p:(isLeft?pair.pair.left:pair.pair.right)) {
        const int gx=std::clamp(static_cast<int>(p.x*10/camera.referenceSize.width),0,9);
        const int gy=std::clamp(static_cast<int>(p.y*10/camera.referenceSize.height),0,9);
        cv::rectangle(grid,{gx*36,gy*18,36,18},{55,140,55},cv::FILLED);
    }
    for(int i=0;i<=10;++i) { cv::line(grid,{i*36,0},{i*36,179},{85,85,85}); cv::line(grid,{0,i*18},{359,i*18},{85,85,85}); }
    grid.copyTo(canvas(cv::Rect(x,y,360,180)));
}
cv::Mat Dashboard(const Observation& observed,const Camera& left,const Camera& right,const Options& o,
                  const std::vector<LivePair>& pairs,const Solution& solved,bool busy,bool automatic,bool epilines,
                  double held,const std::optional<Epipolar>& current,const std::string& status,
                  double rotationChange=0,double translationChange=0) {
    cv::Mat canvas(1020,1440,CV_8UC3,cv::Scalar(24,24,24));
    Label(canvas,"LEFT "+std::to_string(left.referenceSize.width)+"x"+std::to_string(left.referenceSize.height)+
        "  #"+std::to_string(observed.frame.left.frame_id),12,27);
    Label(canvas,"RIGHT "+std::to_string(right.referenceSize.width)+"x"+std::to_string(right.referenceSize.height)+
        "  #"+std::to_string(observed.frame.right.frame_id),732,27);
    Preview(observed.frame.left.image,observed.left,left,observed.right,right,solved.F,true,epilines).copyTo(canvas(cv::Rect(0,40,720,480)));
    Preview(observed.frame.right.image,observed.right,right,observed.left,left,solved.F,false,epilines).copyTo(canvas(cv::Rect(720,40,720,480)));
    Label(canvas,"Pairs: "+std::to_string(pairs.size())+" / "+std::to_string(o.maxPairs)+
        "   Capture: "+(automatic?"AUTO":"MANUAL")+"   Worker: "+(busy?"BUSY":"READY"),15,550);
    Label(canvas,epilines?"View: UNDISTORTED + EPILINES (NOT rectified)":"View: RAW + detected corners; display scales differ",15,578);
    if(!solved.R.empty()) {
        Label(canvas,"Fit: "+std::to_string(solved.pairs.size())+" pairs; RMS="+Number(solved.rms)+" px; baseline="+Number(cv::norm(solved.T),2)+" mm",15,607);
        Label(canvas,"Fit epipolar P95="+Number(solved.epiP95)+" px; max="+Number(solved.epiMax)+" px (max of L/R point-to-line)",15,636);
        Label(canvas,"Change after last solve: rotation="+Number(rotationChange,4)+" deg; translation="+Number(translationChange,3)+" mm",15,665);
        Label(canvas,solved.passed?"Basic checks: PASS; physical distance NOT validated":"Basic checks: NEED MORE PAIRS / HIGH ERROR / SOLVER NOT CONVERGED",15,694,
              solved.passed?cv::Scalar(70,220,70):cv::Scalar(80,170,255));
        Label(canvas,"Worst pair ID="+std::to_string(solved.pairs[Worst(solved)].pair.number)+" (X removes it after solver is ready)",860,665);
    } else Label(canvas,"Need at least 3 accepted pairs to solve R/T",15,608);
    Label(canvas,"Board L/R: "+std::string(observed.leftFound?"FOUND":"MISSING")+" / "+(observed.rightFound?"FOUND":"MISSING"),860,550);
    Label(canvas,"Receive dt="+std::to_string(TimestampDifference(observed.frame.left.timestamp_ms,observed.frame.right.timestamp_ms))+
        " ms (limit "+std::to_string(o.pairMs)+")",860,578);
    Label(canvas,"Stable: "+Number(held,1)+" / "+Number(o.holdSeconds,1)+" s; >=3 observations",860,607);
    if(current) Label(canvas,"Current board epipolar P95="+Number(current->p95)+" px",860,636);
    Label(canvas,status.substr(0,155),15,730,{80,220,255});
    Label(canvas,"LEFT native corner coverage",15,759); Label(canvas,"RIGHT native corner coverage",435,759);
    Coverage(canvas,pairs,left,true,15,775); Coverage(canvas,pairs,right,false,435,775);
    Label(canvas,"Fixed K/D; |Tx|=450..570, |Ty|<=10, |Tz|<=80 mm",830,790);
    if (!solved.T.empty()) {
        Label(canvas,"T mm: "+Number(solved.T.at<double>(0),2)+", "+Number(solved.T.at<double>(1),2)+", "+Number(solved.T.at<double>(2),2),830,914);
        Label(canvas,"Free RMS="+Number(solved.translationConstraint.initialRms)+"; bounded RMS="+Number(solved.rms)+
            (solved.translationConstraint.atLimit?"; AT LIMIT":""),830,945);
    }
    Label(canvas,"11x8 inner corners; 30 mm squares",830,821);
    Label(canvas,"Pause the SAME board during both exposures",830,852);
    Label(canvas,"Receive timestamps are NOT exposure sync",830,883,{80,170,255});
    Label(canvas,"SPACE: capture (stable required) | A: auto/manual | R: undo | X: drop worst | D: clear",15,979);
    Label(canvas,"S: save stereo.yaml + reference.png | E/U: epilines/raw | ESC/Q: exit and save valid solution",15,1007);
    return canvas;
}

struct FrameWriteError:std::runtime_error { using std::runtime_error::runtime_error; };
void WriteFrames(const LivePair& pair,const StereoFrame& raw) {
    try {
        if(!cv::imwrite(pair.pair.leftPath.string(),raw.left.image,{cv::IMWRITE_PNG_COMPRESSION,1}) ||
           !cv::imwrite(pair.pair.rightPath.string(),raw.right.image,{cv::IMWRITE_PNG_COMPRESSION,1}))
            throw FrameWriteError("Cannot write original paired images");
    } catch(const cv::Exception& e) { throw FrameWriteError(e.what()); }
}

int Run(Options o) {
    const DartCongfig config(o.config.string());
    if(o.pairMs<0) o.pairMs=config.frameQueue.maxTimestampDiffMs;
    Require(o.pairMs>=0 && o.pairMs<=1000,"Invalid configured frame_queue.max_timestamp_diff_ms");
    const Camera left=Intrinsics(config.leftCamera,o.leftIntrinsics),right=Intrinsics(config.rightCamera,o.rightIntrinsics);
    cv::setNumThreads(4);
    CameraStopSignals signals;
    LatestPair camera(config,o.pairMs);
    Require(fs::create_directories(o.output),"Output already exists / cannot create a new session directory");
    fs::create_directory(o.output/"left"); fs::create_directory(o.output/"right");
    std::cout<<"Live stereo: left "<<config.leftCamera.deviceIp<<' '<<left.referenceSize<<"; right "<<config.rightCamera.deviceIp<<' '<<right.referenceSize
        <<"; pair receive-time limit="<<o.pairMs<<" ms\nK_left:\n"<<left.K<<"\nK_right:\n"<<right.K
        <<"\nIntrinsics: "<<(o.leftIntrinsics.empty()?"dart.yaml:cameras.left":o.leftIntrinsics.string())<<" | "
        <<(o.rightIntrinsics.empty()?"dart.yaml:cameras.right":o.rightIntrinsics.string())
        <<"\nBoard: 11x8 INNER corners, 30 mm squares. Fixed K/D; no equal-focal constraint.\n"
        <<"Translation bounds (right optical frame, mm): |Tx|=450..570 (initial sign), |Ty|<=10, |Tz|<=80.\n"
        <<"Keep rig fixed, pause board for sampling. Receive-time pairing is NOT hardware exposure sync.\nOutput: "<<o.output<<'\n';
    cv::namedWindow(kWindow,cv::WINDOW_NORMAL); cv::resizeWindow(kWindow,1440,1020);
    Observation observed;
    std::future<Observation> detecting;
    std::future<Solution> pending;
    std::vector<LivePair> pairs;
    Solution solved;
    Stability stability;
    std::optional<Epipolar> current;
    bool automatic=o.automatic,epilines=false,captureRequested=false,jobIsSave=false;
    std::uint64_t revision=0,pendingRevision=0,solvedRevision=std::numeric_limits<std::uint64_t>::max(),savedRevision=solvedRevision;
    int serial=0;
    double rotationChange=0,translationChange=0;
    auto lastPairAt=Clock::now();
    std::string status="Waiting for a paired frame / full board in both images",lastError;
    auto launchSolve=[&](std::optional<LivePair> added={},StereoFrame raw={}) {
        Require(!pending.valid(),"Worker already busy"); pendingRevision=revision; jobIsSave=false;
        const auto snapshot=pairs;
        pending=std::async(std::launch::async,[snapshot,left,right,o,added,raw] {
            if(added) WriteFrames(*added,raw);
            return Solve(snapshot,left,right,o.minPairs);
        });
        stability.reset(); lastError.clear();
    };
    auto finishJob=[&] {
        Solution completed=pending.get();
        if(pendingRevision!=revision) return;
        if(!jobIsSave && !solved.R.empty() && !completed.R.empty()) {
            const cv::Mat delta=completed.R*solved.R.t();
            rotationChange=std::acos(std::clamp((cv::trace(delta)[0]-1)/2,-1.0,1.0))*180/CV_PI;
            translationChange=cv::norm(completed.T-solved.T);
        }
        solved=std::move(completed); solvedRevision=revision;
        if(jobIsSave) savedRevision=revision;
        if(!solved.R.empty()) {
            std::cout<<"Pairs="<<solved.pairs.size()<<" RMS="<<solved.rms<<" px epi P95="<<solved.epiP95
                     <<" px max="<<solved.epiMax<<" px baseline="<<cv::norm(solved.T)<<" mm quality="<<solved.passed<<'\n';
            std::cout<<"Unconstrained T_mm="<<solved.translationConstraint.initialT.t()<<" RMS="<<solved.translationConstraint.initialRms
                     <<" px; bounded T_mm="<<solved.T.t()<<" RMS="<<solved.rms<<" px; converged="<<solved.translationConstraint.converged
                     <<" at_limit="<<solved.translationConstraint.atLimit<<'\n';
            if(observed.leftFound && observed.rightFound) current=MeasureEpipolar(observed.left,observed.right,left,right,solved.F);
        }
        status=jobIsSave?"Saved stereo.yaml and median/worst reference images":"Updated; move board to another pose";
        stability.reset();
    };
    try {
        while(!signals.stopRequested()) {
            if(pending.valid() && pending.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
                try { finishJob(); }
                catch(const FrameWriteError&) { throw; }
                catch(const std::exception& e) { lastError=e.what(); std::cerr<<"Stereo worker: "<<lastError<<'\n'; }
            }
            if(detecting.valid() && detecting.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
                observed=detecting.get();
                const auto receiveTime=std::max(observed.frame.left.timestamp_ms,observed.frame.right.timestamp_ms);
                const auto nowMs=std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
                const bool fresh=nowMs>=receiveTime && nowMs-receiveTime<=2000;
                const bool complete=observed.leftFound && observed.rightFound;
                current.reset();
                if(complete && !solved.R.empty()) current=MeasureEpipolar(observed.left,observed.right,left,right,solved.F);
                if(pending.valid() || !fresh) {
                    stability.reset(); status=pending.valid()?"Saving/solving in background; capture paused":"Processed pair too old; wait for a fresh stationary board";
                } else {
                    const bool stable=stability.update(observed,left,right,o.holdSeconds,o.pairMs);
                    const bool novel=complete && Diverse(observed,pairs,left,right);
                    status=!complete?"Both cameras must see the complete same board":(!novel?"Repeated pose; move/tilt the board":
                        "Hold both boards steady: "+Number(stability.elapsed(),1)+" / "+Number(o.holdSeconds,1)+" s");
                    if(pairs.size()>=static_cast<std::size_t>(o.maxPairs)) status="Pair limit reached; R=undo or X=remove worst pair before adding more";
                    if((captureRequested || automatic) && stable && novel && pairs.size()<static_cast<std::size_t>(o.maxPairs)) {
                        LivePair added; added.pair={0,++serial,o.output/"left"/("left"+std::to_string(serial)+".png"),
                            o.output/"right"/("right"+std::to_string(serial)+".png"),observed.left,observed.right,observed.rightReversed,0};
                        added.leftId=observed.frame.left.frame_id; added.rightId=observed.frame.right.frame_id;
                        added.leftTime=observed.frame.left.timestamp_ms; added.rightTime=observed.frame.right.timestamp_ms;
                        pairs.push_back(added); ++revision; launchSolve(added,observed.frame);
                        status="Accepted pair "+std::to_string(serial)+"; saving original images and updating R/T";
                    }
                }
                captureRequested=false;
            }
            if(!detecting.valid()) {
                if(auto pair=camera.take()) {
                    lastPairAt=Clock::now();
                    detecting=std::async(std::launch::async,[pair=std::move(*pair)]() mutable { return Observe(std::move(pair)); });
                } else if(Clock::now()-lastPairAt>std::chrono::seconds(10))
                    throw std::runtime_error("No paired frames for 10 s; check SDK logs/network and --pair-ms (receive-time tolerance)");
            }
            cv::imshow(kWindow,Dashboard(observed,left,right,o,pairs,solved,pending.valid(),automatic,epilines,stability.elapsed(),current,
                lastError.empty()?status:"Worker error: "+lastError,rotationChange,translationChange));
            const int key=cv::waitKey(5)&0xff;
            if(key==27 || key=='q' || cv::getWindowProperty(kWindow,cv::WND_PROP_VISIBLE)==0) break;
            if(key=='a') { automatic=!automatic; stability.reset(); }
            if(key=='e' || key=='u') epilines=!epilines;
            if(key==' ') { captureRequested=!pending.valid(); if(!captureRequested) status="Wait for the background worker"; }
            if(key=='s') {
                if(pending.valid() || solvedRevision!=revision || solved.R.empty()) status="Wait for an updated solution with at least 3 pairs";
                else {
                    pendingRevision=revision; jobIsSave=true; const auto snapshot=solved;
                    pending=std::async(std::launch::async,[o,left,right,snapshot] { Save(o,left,right,snapshot); return snapshot; });
                    status="Saving report and native-image reference panels"; stability.reset();
                }
            }
            if((key=='r' || key=='d' || key=='x') && !pending.valid()) {
                if(key=='d') pairs.clear();
                else if(key=='r' && !pairs.empty()) pairs.pop_back();
                else if(key=='x' && !solved.R.empty() && solvedRevision==revision) {
                    const int id=solved.pairs[Worst(solved)].pair.number;
                    pairs.erase(std::remove_if(pairs.begin(),pairs.end(),[id](const auto& p){return p.pair.number==id;}),pairs.end());
                } else continue;
                ++revision; solved={}; current.reset(); rotationChange=translationChange=0;
                launchSolve(); status="Rebuilding from retained paired observations";
            }
        }
        camera.stop();
        if(detecting.valid()) detecting.get();
        if(pending.valid()) finishJob();
        if(solvedRevision==revision && !solved.R.empty() && savedRevision!=revision) Save(o,left,right,solved);
        cv::destroyWindow(kWindow);
        if(solvedRevision!=revision || solved.R.empty()) {
            std::cout<<"No current stereo solution. Accepted original pairs remain in "<<o.output<<'\n'; return 2;
        }
        return solved.passed?0:2;
    } catch(...) { camera.stop(); cv::destroyWindow(kWindow); throw; }
}
} // namespace stereo_live

#ifndef DART_STEREO_CALIBRATION_NO_MAIN
int main(int argc,char** argv) {
    try {
        // Keep the old positional and --full-frame invocations for existing datasets.
        if(argc>1 && std::string(argv[1])=="--offline") {
            std::vector<char*> offlineArgs{argv[0]}; offlineArgs.insert(offlineArgs.end(),argv+2,argv+argc);
            return OfflineMain(static_cast<int>(offlineArgs.size()),offlineArgs.data());
        }
        if(argc>1 && (std::string(argv[1])=="--full-frame" || std::string(argv[1]).rfind("-",0)!=0)) return OfflineMain(argc,argv);
        const auto options=stereo_live::Parse(argc,argv); return options.help?0:stereo_live::Run(options);
    } catch(const std::exception& e) { std::cerr<<"Stereo calibration failed: "<<e.what()<<'\n'; return 1; }
}
#endif
