// Single-camera calibration: OpenCV 4.x, C++17.
// Defaults: inputs in ./CS200; parameters and previews in ./CS200_calibration
// Build: g++ -std=c++17 -O2 sub_module/demarcate.cpp -o demarcate $(pkg-config --cflags --libs opencv4)
// Run: ./demarcate [image_directory] [new_output_directory] [full|narrow|k1] [manifest.yaml]
// A manifest permits different crops of one sensor; zoom and focus must stay fixed.
// The output directory must not already exist, to preserve earlier results.
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
constexpr int BOARD_COLS = 11; // Inner corners, not number of squares.
constexpr int BOARD_ROWS = 8;
constexpr float SQUARE_MM = 30.0f;
constexpr int DETECTION_MAX_DIM = 1920;

bool isImage(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".bmp" || ext == ".png" || ext == ".jpg" ||
           ext == ".jpeg" || ext == ".tif" || ext == ".tiff";
}

void saveImage(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image))
        throw std::runtime_error("Cannot save image: " + path.string());
}

int main(int argc, char** argv) {
    try {
        if (argc > 5 || (argc > 1 && std::string(argv[1]) == "--help")) {
            std::cout << "Usage: " << argv[0]
                      << " [image_directory=CS200] [new_output_directory=CS200_calibration]"
                         " [model=full|narrow|k1] [optional_manifest.yaml]\n";
            return argc > 5 ? 1 : 0;
        }
        const fs::path input = argc > 1 ? argv[1] : "CS200";
        const fs::path output = argc > 2 ? argv[2] : "CS200_calibration";
        const std::string calibrationModel = argc > 3 ? argv[3] : "full";
        if (calibrationModel != "full" && calibrationModel != "narrow" && calibrationModel != "k1")
            throw std::runtime_error("Calibration model must be 'full', 'narrow' or 'k1'.");
        if (!fs::is_directory(input))
            throw std::runtime_error("Image directory does not exist: " + input.string());
        if (fs::exists(output))
            throw std::runtime_error("Output already exists; choose a new directory: " + output.string());

        std::vector<fs::path> files;
        std::vector<cv::Point2f> offsets;
        cv::Size imageSize;
        if (argc == 5) {
            // 清单声明统一的参考图像尺寸和每张图的真实裁剪偏移。
            // 右相机目录名只是配对标签；其清单 offset 必须为实际拍摄值零。
            cv::FileStorage manifest(argv[4], cv::FileStorage::READ);
            if (!manifest.isOpened()) throw std::runtime_error("Cannot read manifest.");
            manifest["image_width"] >> imageSize.width;
            manifest["image_height"] >> imageSize.height;
            if (imageSize.width <= 0 || imageSize.height <= 0 ||
                !manifest["images"].isSeq()) throw std::runtime_error("Invalid manifest schema.");
            std::set<std::string> seen;
            for (const auto& node : manifest["images"]) {
                std::string relative;
                node["path"] >> relative;
                const fs::path rel(relative);
                if (relative.empty() || rel.is_absolute() ||
                    std::any_of(rel.begin(), rel.end(), [](const auto& part) { return part == ".."; }) ||
                    !seen.insert(relative).second || !isImage(rel))
                    throw std::runtime_error("Invalid or duplicate manifest path: " + relative);
                if (!node["offset_x"].isInt() || !node["offset_y"].isInt())
                    throw std::runtime_error("Manifest offsets must be integer pixels.");
                const int x = static_cast<int>(node["offset_x"]);
                const int y = static_cast<int>(node["offset_y"]);
                if (x < 0 || y < 0 || x >= imageSize.width || y >= imageSize.height)
                    throw std::runtime_error("Invalid ROI offset: " + relative);
                files.push_back(input / rel);
                offsets.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        } else {
            for (const auto& entry : fs::directory_iterator(input))
                if (entry.is_regular_file() && isImage(entry.path()) &&
                    entry.path().filename().string().rfind("corners_", 0) != 0)
                    files.push_back(entry.path());
            std::sort(files.begin(), files.end());
            offsets.resize(files.size(), cv::Point2f(0, 0));
        }
        if (files.size() < 3)
            throw std::runtime_error("Need at least 3 images with distinct board poses.");

        fs::create_directories(output);
        const cv::Size board(BOARD_COLS, BOARD_ROWS);
        std::vector<cv::Point3f> objectTemplate;
        for (int row = 0; row < BOARD_ROWS; ++row)
            for (int col = 0; col < BOARD_COLS; ++col)
                objectTemplate.emplace_back(col * SQUARE_MM, row * SQUARE_MM, 0.0f);

        std::vector<std::vector<cv::Point2f>> imagePoints;
        std::vector<std::vector<cv::Point3f>> objectPoints;
        std::vector<std::string> usedFiles, rejectedFiles;
        std::vector<cv::Point2f> usedOffsets;
        std::vector<cv::Size> usedSizes;

        for (std::size_t index = 0; index < files.size(); ++index) {
            // 与采集入口保持一致：使用文件原始像素坐标，忽略 EXIF 旋转和镜像。
            // 仅解码为 8 位 BGR，不修改原图。
            cv::Mat image = cv::imread(files[index].string(),
                                       cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
            if (image.empty())
                throw std::runtime_error("Cannot read image: " + files[index].string());
            if (imageSize.empty()) imageSize = image.size();
            if (argc != 5 && image.size() != imageSize)
                throw std::runtime_error("Mixed image resolutions: " + files[index].string());
            if (offsets[index].x + image.cols > imageSize.width ||
                offsets[index].y + image.rows > imageSize.height)
                throw std::runtime_error("ROI extends outside the reference image: " + files[index].string());

            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> corners;
            std::cout << "[" << index + 1 << "/" << files.size() << "] "
                      << files[index].filename().string() << std::endl;

            // Detect on a reduced image when the source is very large. Searching a
            // 20 MP frame directly is both slow and less reliable when the board
            // occupies only a small part of the image. Calibration still uses
            // corner coordinates at the original resolution.
            const int maxDim = std::max(gray.cols, gray.rows);
            const double detectionScale = maxDim > DETECTION_MAX_DIM
                ? static_cast<double>(DETECTION_MAX_DIM) / maxDim
                : 1.0;
            cv::Mat detectionGray;
            if (detectionScale < 1.0) {
                cv::resize(gray, detectionGray, cv::Size(), detectionScale,
                           detectionScale, cv::INTER_AREA);
            } else {
                detectionGray = gray;
            }

            // SB returns subpixel corners in the detection image.
            bool found = cv::findChessboardCornersSB(
                detectionGray, board, corners,
                cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE |
                cv::CALIB_CB_ACCURACY);
            if (!found) {
                // Fallback: classic detector followed by subpixel refinement.
                corners.clear();
                found = cv::findChessboardCorners(
                    detectionGray, board, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
                if (found)
                    cv::cornerSubPix(detectionGray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                         50, 0.001));
            }
            if (found && detectionScale < 1.0) {
                for (auto& corner : corners)
                    corner *= static_cast<float>(1.0 / detectionScale);
            }
            if (found) {
                const int radius = detectionScale < 1.0 ? 11 : 7;
                cv::cornerSubPix(gray, corners, cv::Size(radius, radius), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                                     50, 0.001));
            }
            found = found && corners.size() == objectTemplate.size();
            cv::Mat preview = image.clone();
            cv::drawChessboardCorners(preview, board, corners, found);
            const fs::path relative = files[index].lexically_relative(input);
            const fs::path previewPath = output / "corners" / relative.parent_path() /
                ("corners_" + relative.filename().string() + ".jpg");
            fs::create_directories(previewPath.parent_path());
            saveImage(previewPath, preview);
            if (!found) {
                rejectedFiles.push_back(relative.generic_string());
                std::cerr << "  Rejected: complete 11 x 8 board not detected.\n";
                continue;
            }
            // 只平移角点，图片不旋转、不缩放；不同 ROI 由此共享同一套全幅内参。
            for (auto& corner : corners) corner += offsets[index];
            imagePoints.push_back(corners);
            objectPoints.push_back(objectTemplate);
            usedFiles.push_back(relative.generic_string());
            usedOffsets.push_back(offsets[index]);
            usedSizes.push_back(image.size());
            std::cout << "  Accepted: " << corners.size() << " corners.\n";
        }
        if (imagePoints.size() < 3)
            throw std::runtime_error("Fewer than 3 valid images; inspect output/corners and collect more.");
        if (imagePoints.size() < 15)
            std::cerr << "Limited dataset: treat calibration as preliminary. "
                         "More diverse views and independent validation are recommended.\n";

        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat D = cv::Mat::zeros(5, 1, CV_64F);
        std::vector<cv::Mat> rvecs, tvecs;
        // The narrow-FOV profile prevents weakly observable distortion terms from
        // trading off against the principal point. The full profile retains the
        // standard five-coefficient Brown model for well-covered datasets.
        int calibrationFlags = calibrationModel == "full" ? 0 :
            cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3;
        if (calibrationModel == "narrow") calibrationFlags |= cv::CALIB_FIX_PRINCIPAL_POINT;
        cv::Mat stdIntrinsics, stdExtrinsics, solverViewErrors;
        const double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize,
            K, D, rvecs, tvecs, stdIntrinsics, stdExtrinsics, solverViewErrors, calibrationFlags,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 100, 1e-9));
        if (!std::isfinite(rms) || !cv::checkRange(K) || !cv::checkRange(D) ||
            K.at<double>(0, 0) <= 0 || K.at<double>(1, 1) <= 0)
            throw std::runtime_error("Calibration produced invalid parameters.");

        std::vector<double> perViewRms;
        for (std::size_t i = 0; i < imagePoints.size(); ++i) {
            std::vector<cv::Point2f> projected;
            cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], K, D, projected);
            const double error = std::sqrt(
                cv::norm(imagePoints[i], projected, cv::NORM_L2SQR) / projected.size());
            perViewRms.push_back(error);
            std::cout << usedFiles[i] << ": RMS = " << error << " px\n";
        }

        cv::FileStorage yaml((output / "intrinsics.yaml").string(), cv::FileStorage::WRITE);
        if (!yaml.isOpened()) throw std::runtime_error("Cannot open output YAML.");
        yaml << "image_width" << imageSize.width << "image_height" << imageSize.height
             << "board_columns" << BOARD_COLS << "board_rows" << BOARD_ROWS
             << "square_size_mm" << SQUARE_MM
             << "camera_matrix" << K << "distortion_coefficients" << D
             << "distortion_order" << "k1,k2,p1,p2,k3"
             << "coordinate_system" << "file_pixels_plus_roi_offset"
             << "exif_orientation" << "ignored"
             << "source_directory" << fs::absolute(input).string()
             << "used_views" << static_cast<int>(usedFiles.size())
             << "source_view_count" << static_cast<int>(files.size())
             << "rejected_files" << rejectedFiles
             << "std_deviations_intrinsics" << stdIntrinsics
             << "calibration_model" << calibrationModel
             << "calibration_constraints" << (calibrationModel == "narrow"
                    ? "principal_point=image_center,tangential=zero,k2=zero,k3=zero"
                    : (calibrationModel == "k1" ? "tangential=zero,k2=zero,k3=zero" : "none"))
             << "rms_reprojection_error_px" << rms
             << "per_view_rms_px" << perViewRms;
        // Each pose maps board coordinates (mm) into camera coordinates.
        yaml << "board_poses" << "[";
        for (std::size_t i = 0; i < usedFiles.size(); ++i)
            yaml << "{" << "image" << usedFiles[i]
                 << "offset_x" << static_cast<int>(usedOffsets[i].x)
                 << "offset_y" << static_cast<int>(usedOffsets[i].y)
                 << "source_width" << usedSizes[i].width
                 << "source_height" << usedSizes[i].height
                 << "image_points_reference" << imagePoints[i]
                 << "rvec" << rvecs[i] << "tvec_mm" << tvecs[i] << "}";
        yaml << "]";
        // 纯裁剪不改变焦距和畸变；ROI 主点等于参考主点减去 OffsetX/OffsetY。
        yaml << "roi_intrinsics" << "[";
        std::set<std::tuple<int, int, int, int>> seenRois;
        for (std::size_t i = 0; i < usedFiles.size(); ++i) {
            const int x = static_cast<int>(usedOffsets[i].x), y = static_cast<int>(usedOffsets[i].y);
            if (!seenRois.emplace(x, y, usedSizes[i].width, usedSizes[i].height).second) continue;
            cv::Mat roiK = K.clone();
            roiK.at<double>(0, 2) -= x;
            roiK.at<double>(1, 2) -= y;
            yaml << "{" << "offset_x" << x << "offset_y" << y
                 << "image_width" << usedSizes[i].width << "image_height" << usedSizes[i].height
                 << "camera_matrix" << roiK << "distortion_coefficients" << D << "}";
        }
        yaml << "]";
        yaml.release();

        std::cout << "\nK (pixels):\n" << K << "\nD [k1,k2,p1,p2,k3]:\n" << D
                  << "\nOverall RMS: " << rms << " px"
                  << "\nUsed views: " << usedFiles.size() << "/" << files.size()
                  << "\nSaved to: " << fs::absolute(output).string()
                  << "\nRMS alone does not establish real-world accuracy.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Calibration failed: " << e.what() << '\n';
        return 1;
    }
}
