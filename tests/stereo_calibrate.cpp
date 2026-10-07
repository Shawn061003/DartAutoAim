// 双目标定：读取原图、固定已标定的 K/D，求 X_right = R * X_left + T。
// 用法：stereo_calibrate [left_dir right_dir output_dir [left_yaml right_yaml]]
// 输出目录必须不存在。退出码：0=通过基本质量检查，2=已保存诊断结果但质量不合格，1=程序错误。
// 采集前提：相机之间的位姿固定，且每对左右图中的棋盘必须保持同一位姿。
#include <opencv2/opencv.hpp>
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
namespace {
constexpr int kColumns = 11, kRows = 8;
constexpr float kSquareMm = 30.F;
const cv::Size kFrameSize(1440, 1080);
struct Group { std::string label; cv::Point2f leftOffset; bool offsetConfirmed = true; };
// 在这里设置各组左相机拍摄时的真实 OffsetX/OffsetY。
// 右侧同名分组仅用于配对，右相机偏移始终为(0,0)。
const std::vector<Group> kGroups{
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
struct Camera { cv::Mat K, D; cv::Size referenceSize; };
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
cv::Mat ReadImage(const fs::path& path) {
    auto image = cv::imread(path.string(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    Require(!image.empty() && image.size() == kFrameSize,
            "Expected raw 1440x1080 image: " + path.string());
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
bool Detect(const cv::Mat& image, std::vector<cv::Point2f>& points) {
    cv::Mat gray; cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    bool found = cv::findChessboardCornersSB(gray, {kColumns, kRows}, points,
        cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
    if (!found) {
        points.clear();
        found = cv::findChessboardCorners(gray, {kColumns, kRows}, points,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    }
    if (found) cv::cornerSubPix(gray, points, {7,7}, {-1,-1},
        {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, .001});
    return found && points.size() == kColumns * kRows;
}
struct Pair {
    std::size_t group;
    int number;
    fs::path leftPath, rightPath;
    std::vector<cv::Point2f> left, right; // 左全幅参考坐标 / 右原图坐标。
    bool rightReversed = false;
    double epipolarRms = 0;
};
struct Rectification { cv::Mat leftK, R1, R2, P1, P2, Q; };
double Quantile(std::vector<double> values, double q) {
    Require(!values.empty(), "No error samples");
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(q * (values.size()-1)))];
}
std::string Number(double v, int digits=3) {
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
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0]
                << " [left_dir right_dir NEW_output_dir [left_intrinsics.yaml right_intrinsics.yaml]]\n"
                << "Edit kGroups for LEFT capture offsets; RIGHT offsets are always zero.\n"
                << "Exit 0: basic checks passed; 2: diagnostics saved but quality failed; 1: error.\n";
            return 0;
        }
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
        cv::setNumThreads(4);
        std::vector<Pair> pairs;
        std::vector<std::string> rejected;
        int discovered = 0;
        // 1. 分组后按数值编号配对；禁止将两组的left0/right0混到一起。
        for (std::size_t g=0; g<kGroups.size(); ++g) {
            const auto& group = kGroups[g];
            const auto ld=leftDir/("left-"+group.label), rd=rightDir/("right-"+group.label);
            if (!fs::exists(ld) && !fs::exists(rd)) continue;
            Require(group.leftOffset.x>=0 && group.leftOffset.y>=0 &&
                    group.leftOffset.x+kFrameSize.width<=left.referenceSize.width &&
                    group.leftOffset.y+kFrameSize.height<=left.referenceSize.height, "LEFT ROI outside sensor");
            const auto lf=IndexFiles(ld,"left"), rf=IndexFiles(rd,"right");
            Require(lf.size()==rf.size(), "Unmatched frame counts in group "+group.label);
            for (const auto& item : lf) {
                Require(rf.count(item.first)==1, "Missing right frame "+std::to_string(item.first));
                ++discovered;
                Pair p{g,item.first,item.second,rf.at(item.first),{},{},false,0.};
                const auto li=ReadImage(p.leftPath), ri=ReadImage(p.rightPath);
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
                std::cout << "Detected " << group.label << '/' << item.first << " (88+88 corners)\n";
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
        const double rms=cv::stereoCalibrate(objects,lp,rp,kl,dl,kr,dr,left.referenceSize,
            R,T,E,F,errors,cv::CALIB_FIX_INTRINSIC,
            {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,150,1e-9});
        Require(std::isfinite(rms) && cv::checkRange(R) && cv::checkRange(T) && cv::norm(T)>1e-6,
                "Invalid stereo calibration result");
        Require(cv::norm(kl,left.K,cv::NORM_INF)==0 && cv::norm(kr,right.K,cv::NORM_INF)==0 &&
                cv::norm(dl,left.D,cv::NORM_INF)==0 && cv::norm(dr,right.D,cv::NORM_INF)==0,
                "Fixed intrinsics unexpectedly changed");

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
        const bool residualChecksPassed=rms<=kMaxRmsPx && epiP95<=kMaxEpipolarP95Px && pairs.size()>=kMinQualityPairs;
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
            cv::stereoRectify(rect.leftK,left.D,right.K,right.D,kFrameSize,R,T,
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
        yaml << "pairs" << "[";
        for (std::size_t i=0;i<pairs.size();++i) {
            const auto& p=pairs[i];const auto& g=kGroups[p.group];
            yaml << "{" << "group" << g.label << "number" << p.number
                 << "left_image" << p.leftPath.string() << "right_image" << p.rightPath.string()
                 << "left_offset_x" << static_cast<int>(g.leftOffset.x) << "left_offset_y" << static_cast<int>(g.leftOffset.y)
                 << "left_offset_confirmed" << static_cast<int>(g.offsetConfirmed)
                 << "right_offset_x" << 0 << "right_offset_y" << 0
                 << "right_corner_order_reversed" << static_cast<int>(p.rightReversed)
                 << "left_rms_px" << errors.at<double>(static_cast<int>(i),0)
                 << "right_rms_px" << errors.at<double>(static_cast<int>(i),1)
                 << "epipolar_rms_px" << p.epipolarRms << "}";
        }
        yaml << "]" << "rectification_by_left_roi" << "[";
        for (const auto& item:rectifications) {
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
             << "reference_layout" << "per selected pair: rectified view followed by raw input with detected corner order"
             << "reference_pairs" << "[";
        for (std::size_t row=0;row<referenceIndices.size();++row) {
            const auto i=referenceIndices[row];const auto& p=pairs[i];
            const auto label=kGroups[p.group].label+" / pair "+std::to_string(p.number)+
                (i==worst?" (WORST)":" (GROUP MEDIAN)")+"  epipolar RMS="+Number(p.epipolarRms)+" raw px";
            ReferenceRow(p,rectifications.at(p.group),left,right,label).copyTo(
                reference(cv::Rect(0,75+kReferenceRowHeight*static_cast<int>(row),1440,kReferenceRowHeight)));
            yaml << "{" << "group" << kGroups[p.group].label << "number" << p.number << "}";
        }
        yaml << "]";yaml.release();
        Require(cv::imwrite((output/"reference.png").string(),reference),"Cannot save reference image");
        std::cout << "RMS: " << rms << " px\nEpipolar P95: " << epiP95
                  << " px\nBaseline (diagnostic unless validated): " << cv::norm(T) << " mm\nR:\n" << R
                  << "\nT_mm:\n" << T << "\nResidual checks passed: " << residualChecksPassed
                  << "\nLEFT offsets confirmed: " << offsetsConfirmed
                  << "\nQuality passed: " << passes << "\nSaved to: " << output << '\n';
        return passes?0:2;
    } catch (const std::exception& e) { std::cerr << "Stereo calibration failed: " << e.what() << '\n'; return 1; }
}
