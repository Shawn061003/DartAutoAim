#pragma once

#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <string>
#include <vector>

// 实物板：6x6 tag36h11，黑色外框88mm，间隙26.4mm，黑边宽2个编码单元。
// 正看板面，ID每行从右向左递增，各行从上向下递增。
// 板坐标原点在整块标记阵列左上角，X向右、Y向下，Z=0。
namespace aprilgrid {
constexpr int kColumns=6, kRows=6, kTags=36, kPointCount=4*kTags, kBorderBits=2;
constexpr float kTagMm=88.F, kGapMm=26.4F;

inline cv::Ptr<cv::aruco::Dictionary> dictionary() {
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
}

// 按ID升序，每个标记依次为解码方向的左上、右上、右下、左下。
inline std::vector<cv::Point3f> objectPoints() {
    std::vector<cv::Point3f> points;
    points.reserve(kPointCount);
    for(int id=0;id<kTags;++id) {
        const float x=(kColumns-1-id%kColumns)*(kTagMm+kGapMm);
        const float y=(id/kColumns)*(kTagMm+kGapMm);
        points.insert(points.end(),{{x,y,0},{x+kTagMm,y,0},
                                   {x+kTagMm,y+kTagMm,0},{x,y+kTagMm,0}});
    }
    return points;
}

// 保留现有完整板采样约定：36个ID均唯一可见时才返回144个对应角点。
inline bool detect(const cv::Mat& image,std::vector<cv::Point2f>& points) {
    points.clear();
    if(image.empty()) return false;
    cv::Mat gray;
    if(image.channels()==1) gray=image;
    else cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
    auto parameters=cv::aruco::DetectorParameters::create();
    parameters->markerBorderBits=kBorderBits;
    // 默认周长比例会剔除5472像素全幅中约40像素宽的远处标记。
    parameters->minMarkerPerimeterRate=.005;
    parameters->cornerRefinementMethod=cv::aruco::CORNER_REFINE_SUBPIX;
    parameters->cornerRefinementWinSize=3;
    parameters->cornerRefinementMaxIterations=50;
    parameters->cornerRefinementMinAccuracy=.001;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    cv::aruco::detectMarkers(gray,dictionary(),corners,ids,parameters);
    std::array<int,kTags> locations;
    locations.fill(-1);
    for(std::size_t i=0;i<ids.size();++i) {
        const int id=ids[i];
        if(id<0 || id>=kTags) continue;
        if(locations[id]>=0) return false;
        locations[id]=static_cast<int>(i);
    }
    for(int id=0;id<kTags;++id) if(locations[id]<0) return false;
    points.reserve(kPointCount);
    for(int id=0;id<kTags;++id)
        points.insert(points.end(),corners[locations[id]].begin(),corners[locations[id]].end());
    return true;
}

inline void draw(cv::Mat& image,const std::vector<cv::Point2f>& points) {
    if(points.size()!=kPointCount) return;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    for(int id=0;id<kTags;++id) {
        corners.emplace_back(points.begin()+4*id,points.begin()+4*id+4);
        ids.push_back(id);
    }
    cv::aruco::drawDetectedMarkers(image,corners,ids);
}

inline void writeSpecification(cv::FileStorage& file) {
    file<<"board_type"<<"aprilgrid"<<"tag_family"<<"tag36h11"
        <<"board_columns"<<kColumns<<"board_rows"<<kRows
        <<"tag_size_mm"<<kTagMm<<"tag_gap_mm"<<kGapMm<<"tag_spacing_ratio"<<.3
        <<"marker_border_bits"<<kBorderBits<<"start_id"<<0<<"end_id"<<35
        <<"id_layout"<<"rows_top_to_bottom_ids_right_to_left"
        <<"point_order"<<"id_ascending_then_decoded_TL_TR_BR_BL";
}

inline bool matchesSpecification(const cv::FileStorage& file) {
    return static_cast<std::string>(file["board_type"])=="aprilgrid"
        && static_cast<std::string>(file["tag_family"])=="tag36h11"
        && static_cast<int>(file["board_columns"])==kColumns
        && static_cast<int>(file["board_rows"])==kRows
        && std::abs(static_cast<double>(file["tag_size_mm"])-kTagMm)<1e-5
        && std::abs(static_cast<double>(file["tag_gap_mm"])-kGapMm)<1e-5
        && static_cast<int>(file["marker_border_bits"])==kBorderBits
        && static_cast<std::string>(file["id_layout"])=="rows_top_to_bottom_ids_right_to_left"
        && static_cast<std::string>(file["point_order"])=="id_ascending_then_decoded_TL_TR_BR_BL";
}
} // namespace aprilgrid
