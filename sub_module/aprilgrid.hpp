#pragma once

#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <algorithm>
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

struct DetectionInfo {
    std::vector<int> ids,duplicates;
    std::vector<std::vector<cv::Point2f>> corners; // 与ids对应，始终使用原图坐标。
    double scale=1.;
    int attempts=0;
    bool complete() const {return ids.size()==kTags && duplicates.empty();}
    std::vector<int> missing() const {
        std::vector<int> result;
        for(int id=0;id<kTags;++id)
            if(std::find(ids.begin(),ids.end(),id)==ids.end()) result.push_back(id);
        return result;
    }
    void translate(cv::Point2f offset) {
        for(auto& marker:corners) for(auto& point:marker) point+=offset;
    }
};

inline std::string idText(const std::vector<int>& ids,std::size_t first=0,std::size_t count=kTags) {
    if(ids.empty()) return "-";
    std::string result;
    for(std::size_t i=first;i<std::min(ids.size(),first+count);++i)
        result+=(result.empty()?"":",")+std::to_string(ids[i]);
    return result;
}

inline DetectionInfo fromPoints(const std::vector<cv::Point2f>& points) {
    DetectionInfo result;
    if(points.size()!=kPointCount) return result;
    for(int id=0;id<kTags;++id) {
        result.ids.push_back(id);
        result.corners.emplace_back(points.begin()+4*id,points.begin()+4*id+4);
    }
    return result;
}

inline std::vector<cv::Point2f> fullPoints(const DetectionInfo& info) {
    std::vector<cv::Point2f> points;
    if(!info.complete() || info.corners.size()!=kTags) return points;
    for(int id=0;id<kTags;++id) {
        if(info.ids[id]!=id || info.corners[id].size()!=4) return {};
        points.insert(points.end(),info.corners[id].begin(),info.corners[id].end());
    }
    return points;
}

// 完整观测沿用原有points字段；只额外保存手动修过的部分标记，兼容旧标注。
inline void writeManualMarkers(cv::FileStorage& file,const std::string& key,const DetectionInfo& info) {
    if(info.ids.empty()) return;
    file<<key<<"{"<<"ids"<<info.ids<<"duplicates"<<info.duplicates<<"corners"<<"[";
    for(const auto& marker:info.corners) file<<marker;
    file<<"]"<<"}";
}
inline DetectionInfo readManualMarkers(const cv::FileNode& node,cv::Size size) {
    DetectionInfo info;
    if(node.empty()) return info;
    node["ids"]>>info.ids;
    if(!node["duplicates"].empty()) node["duplicates"]>>info.duplicates;
    for(int id:info.duplicates) if(id<0 || id>=kTags) return {};
    for(const auto& item:node["corners"]) {std::vector<cv::Point2f> marker;item>>marker;info.corners.push_back(std::move(marker));}
    if(info.ids.size()!=info.corners.size() || info.ids.size()>kTags) return {};
    int previous=-1;
    for(std::size_t i=0;i<info.ids.size();++i) {
        if(info.ids[i]<=previous || info.ids[i]>=kTags || info.corners[i].size()!=4) return {};
        previous=info.ids[i];
        for(const auto& point:info.corners[i])
            if(!std::isfinite(point.x) || !std::isfinite(point.y) || point.x<0 || point.y<0 || point.x>=size.width || point.y>=size.height) return {};
    }
    return info;
}

// 多尺度只提供检测初值；每次均回原生灰度图精修，不将插值像素作为标定观测。
// 部分结果仅供显示，仍须36个ID唯一完整时才返回可标定的144个角点。
inline bool detect(const cv::Mat& image,std::vector<cv::Point2f>& points,DetectionInfo* info=nullptr) {
    points.clear();
    if(info) *info={};
    if(image.empty()) return false;
    cv::Mat gray;
    if(image.channels()==1) gray=image;
    else cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
    if(gray.cols<11 || gray.rows<11) return false; // 3像素精修窗口需要足够的原生图像边界。
    auto parameters=cv::aruco::DetectorParameters::create();
    parameters->markerBorderBits=kBorderBits;
    // 默认周长比例会剔除5472像素全幅中约40像素宽的远处标记。
    parameters->minMarkerPerimeterRate=.005;
    parameters->cornerRefinementMethod=cv::aruco::CORNER_REFINE_NONE;
    DetectionInfo best;
    int attempts=0;
    for(const double factor:{1.,1.5,2.,.75,.5}) {
        // 限制插值图的内存；原图始终尝试，过大的放大倍率直接跳过。
        if(factor>1. && (gray.total()*factor*factor>25000000. ||
                         std::max(gray.cols,gray.rows)*factor>8192.)) continue;
        if(std::min(gray.cols,gray.rows)*factor<8.) continue;
        cv::Mat search;
        if(factor==1.) search=gray;
        else cv::resize(gray,search,{},factor,factor,factor<1.?cv::INTER_AREA:cv::INTER_LINEAR);
        ++attempts;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        cv::aruco::detectMarkers(search,dictionary(),corners,ids,parameters);
        DetectionInfo current;current.scale=factor;
        std::array<int,kTags> locations;locations.fill(-1);
        for(std::size_t i=0;i<ids.size();++i) {
            const int id=ids[i];
            if(id<0 || id>=kTags) continue;
            if(locations[id]>=0) current.duplicates.push_back(id);
            else locations[id]=static_cast<int>(i);
        }
        const double sx=static_cast<double>(search.cols)/gray.cols;
        const double sy=static_cast<double>(search.rows)/gray.rows;
        for(int id=0;id<kTags;++id) if(locations[id]>=0) {
            auto marker=corners[locations[id]];
            for(auto& point:marker) {
                // resize使用像素中心映射，X/Y分别用实际尺寸比例，兼容奇数ROI。
                point.x=static_cast<float>((point.x+.5)/sx-.5);
                point.y=static_cast<float>((point.y+.5)/sy-.5);
            }
            if(!std::all_of(marker.begin(),marker.end(),[&](const cv::Point2f& p) {
                return std::isfinite(p.x) && std::isfinite(p.y) && p.x>=0 && p.y>=0 && p.x<gray.cols && p.y<gray.rows;
            })) continue;
            cv::cornerSubPix(gray,marker,{3,3},{-1,-1},
                {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,50,.001});
            const bool valid=std::all_of(marker.begin(),marker.end(),[&](const cv::Point2f& p) {
                return std::isfinite(p.x) && std::isfinite(p.y) && p.x>=0 && p.y>=0 && p.x<gray.cols && p.y<gray.rows;
            });
            if(valid) {current.ids.push_back(id);current.corners.push_back(std::move(marker));}
        }
        if(current.ids.size()>best.ids.size() ||
           (current.ids.size()==best.ids.size() && current.duplicates.size()<best.duplicates.size())) best=std::move(current);
        if(best.complete()) break;
    }
    best.attempts=attempts;
    if(info) *info=best;
    if(!best.complete()) return false;
    points.reserve(kPointCount);
    for(const auto& marker:best.corners) points.insert(points.end(),marker.begin(),marker.end());
    return true;
}

inline void draw(cv::Mat& image,const DetectionInfo& info) {
    if(!info.ids.empty()) cv::aruco::drawDetectedMarkers(image,info.corners,info.ids);
}

inline void draw(cv::Mat& image,const std::vector<cv::Point2f>& points) {
    draw(image,fromPoints(points));
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
