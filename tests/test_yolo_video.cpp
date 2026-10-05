#include "Perception/YOLOinfer.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// 用右相机视频检查公开的双帧接口；左帧为占位图，不用于左相机效果评估。
int main(int argc, char** argv)
{
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " model.onnx right_video [device=CPU] [max_frames=0] [stride=1] [output_dir=build/yolo_video]\n";
        return 1;
    }
    try {
        const std::string device = argc > 3 ? argv[3] : "CPU";
        const int limit = argc > 4 ? std::stoi(argv[4]) : 0;
        const int stride = argc > 5 ? std::stoi(argv[5]) : 1;
        const std::filesystem::path outputDir = argc > 6 ? argv[6] : "build/yolo_video";
        if (limit < 0 || stride < 1) {
            throw std::invalid_argument("max_frames must be >= 0 and stride must be >= 1.");
        }
        cv::VideoCapture video(argv[2]);
        if (!video.isOpened()) {
            throw std::runtime_error("Cannot open right-camera video.");
        }
        std::filesystem::create_directories(outputDir);
        std::ofstream csv(outputDir / "detections.csv");
        if (!csv) {
            throw std::runtime_error("Cannot create detections.csv.");
        }
        csv << "frame_id,timestamp_ms,target_count,target_rank,x,y,width,height,confidence,pair_pipeline_ms\n";

        const auto initStart = std::chrono::steady_clock::now();
        YOLOInference detector(argv[1], device);
        const double initMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - initStart).count();
        std::cout << "Initialization and warmup: " << initMs << " ms\n"
                  << "Right video: " << video.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                  << video.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n"
                  << "Left input: blank placeholder; pair_pipeline_ms includes both inputs.\n"
                  << std::flush;

        CameraFrame left;
        left.camera_side = CameraSide::Left;
        left.image = cv::Mat::zeros(3648, 5472, CV_8UC3);
        cv::Mat frame;
        std::uint64_t frameId = 0;
        int processed = 0;
        int detected = 0;
        int totalTargets = 0;
        double totalMs = 0.0;
        while ((limit == 0 || processed < limit) && video.read(frame)) {
            const auto currentId = frameId++;
            if (currentId % stride != 0) {
                continue;
            }
            CameraFrame right{frame, currentId,
                static_cast<std::int64_t>(std::llround(video.get(cv::CAP_PROP_POS_MSEC)))};
            right.camera_side = CameraSide::Right;
            left.frame_id = currentId;
            left.timestamp_ms = right.timestamp_ms;
            const auto start = std::chrono::steady_clock::now();
            const auto result = detector.RunYOLOInfer(left, right);
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            totalMs += elapsed;
            ++processed;

            const bool hasTarget = !result.right.empty();
            detected += hasTarget;
            totalTargets += static_cast<int>(result.right.size());
            if (result.right.empty()) {
                csv << currentId << ',' << right.timestamp_ms << ",0,0,0,0,0,0,0," << elapsed << '\n';
            }
            // 每个目标保存一行，rank仅表示本帧置信度顺序，不表示基地/前哨站身份。
            for (std::size_t i = 0; i < result.right.size(); ++i) {
                const auto& target = result.right[i];
                const auto& roi = target.roi;
                if ((roi & cv::Rect(0, 0, frame.cols, frame.rows)) != roi || roi.empty()
                    || target.frame_id != currentId || target.timestamp_ms != right.timestamp_ms) {
                    throw std::runtime_error("ROI bounds or frame metadata are incorrect.");
                }
                csv << currentId << ',' << right.timestamp_ms << ',' << result.right.size() << ','
                    << i + 1 << ',' << roi.x << ',' << roi.y << ',' << roi.width << ',' << roi.height
                    << ',' << target.conf << ',' << elapsed << '\n';
            }

            // 输出首个检测及每100个处理帧的预览，下游分别裁剪并保存两个ROI。
            if ((hasTarget && detected == 1) || processed % 100 == 0) {
                cv::Mat preview = frame.clone();
                for (std::size_t i = 0; i < result.right.size(); ++i) {
                    const auto& target = result.right[i];
                    const auto color = i == 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
                    cv::rectangle(preview, target.roi, color, 2);
                    const std::string rank = std::to_string(i + 1);
                    cv::putText(preview, "target=" + rank + " conf=" + std::to_string(target.conf),
                                cv::Point(20, 70 + static_cast<int>(i) * 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
                    const auto roiPath = outputDir / ("roi_" + std::to_string(currentId)
                                                       + "_target_" + rank + ".png");
                    if (!cv::imwrite(roiPath.string(), frame(target.roi))) {
                        throw std::runtime_error("Cannot save ROI preview.");
                    }
                }
                cv::putText(preview, "frame=" + std::to_string(currentId)
                            + " targets=" + std::to_string(result.right.size()), cv::Point(20, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                if (!cv::imwrite((outputDir / ("frame_" + std::to_string(currentId) + ".jpg")).string(),
                                 preview)) {
                    throw std::runtime_error("Cannot save annotated frame.");
                }
            }
            if (processed % 100 == 0) {
                std::cout << "Processed " << processed << ", detected " << detected << '\n' << std::flush;
            }
        }
        if (processed == 0) {
            throw std::runtime_error("No video frames were read.");
        }
        std::cout << "Processed=" << processed << " detected=" << detected
                  << " total_targets=" << totalTargets
                  << " mean_pair_pipeline_ms=" << totalMs / processed << "\n"
                  << "CSV: " << (outputDir / "detections.csv") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

