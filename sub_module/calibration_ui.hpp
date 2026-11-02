#pragma once
#include "aprilgrid.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/freetype.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <stdexcept>

namespace calibration_ui {
// 拖动时仅更新显示；松开并通过四边形检查后，由调用方提交原生坐标。
class CornerEditor {
    int marker_=-1,corner_=-1,side_=0,undoSide_=-1;
    bool dragging_=false;
    cv::Point2d grabOffset_;
    aprilgrid::DetectionInfo before_,undo_;
public:
    bool enabled=false;
    bool dragging() const {return dragging_;}
    int side() const {return side_;}
    int undoSide() const {return undoSide_;}
    void reset() {*this=CornerEditor{};}
    template<class View> bool begin(const aprilgrid::DetectionInfo& info,const View& view,cv::Point2d screen,int side=0) {
        double nearest=10.;int marker=-1,corner=-1;
        for(std::size_t i=0;i<info.corners.size();++i) for(int j=0;j<4;++j) {
            const double distance=cv::norm(view.project(info.corners[i][j])-screen);
            if(distance<nearest) {nearest=distance;marker=static_cast<int>(i);corner=j;}
        }
        if(marker<0) return false;
        marker_=marker;corner_=corner;side_=side;before_=info;
        grabOffset_=cv::Point2d(info.corners[marker][corner])-view.unproject(screen);
        dragging_=true;return true;
    }
    template<class View> void move(aprilgrid::DetectionInfo& info,const View& view,cv::Point2d screen) {
        if(!dragging_) return;
        auto point=view.unproject(screen)+grabOffset_;
        info.corners[marker_][corner_]={static_cast<float>(std::clamp(point.x,0.,double(view.native.width-1))),
                                      static_cast<float>(std::clamp(point.y,0.,double(view.native.height-1)))};
    }
    void cancel(aprilgrid::DetectionInfo& info) {if(dragging_) info=before_;dragging_=false;}
    bool finish(aprilgrid::DetectionInfo& info) {
        if(!dragging_) return false;
        dragging_=false;
        const auto& marker=info.corners[marker_];const auto& original=before_.corners[marker_];
        if(!cv::isContourConvex(marker) || std::abs(cv::contourArea(marker))<1. ||
           cv::contourArea(marker,true)*cv::contourArea(original,true)<=0.) {info=before_;return false;}
        if(cv::norm(marker[corner_]-original[corner_])<1e-5) return false;
        undo_=before_;undoSide_=side_;return true;
    }
    bool undo(aprilgrid::DetectionInfo& info) {
        if(undoSide_<0) return false;
        info=std::move(undo_);undoSide_=-1;marker_=corner_=-1;return true;
    }
    void draw(cv::Mat& pane,const aprilgrid::DetectionInfo& display,int side=0) const {
        if(!enabled) return;
        for(std::size_t i=0;i<display.corners.size();++i) for(int j=0;j<4;++j) {
            const auto point=display.corners[i][j];
            const bool active=side==side_ && static_cast<int>(i)==marker_ && j==corner_;
            cv::circle(pane,point,active?6:3,active?cv::Scalar(0,165,255):cv::Scalar(0,255,255),1,cv::LINE_AA);
            if(active) {cv::line(pane,point-cv::Point2f(9,0),point+cv::Point2f(9,0),{0,165,255},1,cv::LINE_AA);
                         cv::line(pane,point-cv::Point2f(0,9),point+cv::Point2f(0,9),{0,165,255},1,cv::LINE_AA);}
        }
    }
};
inline bool gtkBackend() {
    if(const auto* backend=std::getenv("OPENCV_UI_BACKEND")) {
        const std::string name=backend;if(name.rfind("GTK",0)==0) return true;
    }
    const auto info=cv::getBuildInformation();const auto begin=info.find("GUI:");
    if(begin==std::string::npos) return false;
    return info.substr(begin,info.find('\n',begin)-begin).find("GTK")!=std::string::npos;
}
// OpenCV 4.5.4 GTK: wheel direction is reversed, vertical smooth events may be
// MOUSEHWHEEL, and scroll callbacks do not populate pointer coordinates.
struct PointerInput {
    bool gtk=gtkBackend(),seen=false;cv::Point2d last;
    static bool isWheel(int event) {return event==cv::EVENT_MOUSEWHEEL || event==cv::EVENT_MOUSEHWHEEL;}
    cv::Point2d point(int event,int x,int y) {
        if(isWheel(event) && gtk && seen) return last;
        if(!isWheel(event)) {last={static_cast<double>(x),static_cast<double>(y)};seen=true;}
        return {static_cast<double>(x),static_cast<double>(y)};
    }
    int direction(int flags) const {
        const int delta=cv::getMouseWheelDelta(flags);
        return (delta>0?1:delta<0?-1:0)*(gtk?-1:1);
    }
};
inline std::string fontPath() {
    if(const auto* path=std::getenv("DART_CALIBRATION_FONT")) {
        if(std::filesystem::is_regular_file(path)) return path;
        throw std::runtime_error("DART_CALIBRATION_FONT 指定的字体文件不存在");
    }
    for(const auto* path:{"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc","/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc","/mnt/c/Windows/Fonts/simsun.ttc"})
        if(std::filesystem::is_regular_file(path)) return path;
    throw std::runtime_error("缺少中文字体：请安装 fonts-noto-cjk，或用 DART_CALIBRATION_FONT 指定中文字体文件");
}
inline cv::Ptr<cv::freetype::FreeType2> font(bool latin=false) {
    thread_local auto chinese=[] {auto f=cv::freetype::createFreeType2();f->loadFontData(fontPath(),0);return f;}();
    // DroidSansFallback omits Latin glyphs and symbols such as +/-; use a Latin face too.
    thread_local auto ascii=[] {
        auto f=cv::freetype::createFreeType2();std::string path=fontPath();
        for(const auto* candidate:{"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"})
            if(std::filesystem::is_regular_file(candidate)) {path=candidate;break;}
        f->loadFontData(path,0);return f;
    }();
    return latin?ascii:chinese;
}
template<class F> inline void runs(const std::string& value,F callback) {
    const auto next=[&](std::size_t offset) {
        const auto lead=static_cast<unsigned char>(value[offset]);std::size_t length=1;unsigned cp=lead;
        if(lead>=0xf0) {length=4;cp=lead&7;}else if(lead>=0xe0) {length=3;cp=lead&15;}else if(lead>=0xc0) {length=2;cp=lead&31;}
        for(std::size_t i=1;i<length && offset+i<value.size();++i) cp=(cp<<6)|(static_cast<unsigned char>(value[offset+i])&63);
        return std::make_pair(std::min(value.size(),offset+length),cp<0x2e80);
    };
    std::size_t start=0;
    while(start<value.size()) {
        auto first=next(start);std::size_t end=first.first;
        while(end<value.size()) {const auto character=next(end);if(character.second!=first.second) break;end=character.first;}
        callback(value.substr(start,end-start),first.second);start=end;
    }
}
inline int textWidth(const std::string& value,int height) {
    int width=0,baseline=0;runs(value,[&](const std::string& run,bool latin){width+=font(latin)->getTextSize(run,height,-1,&baseline).width;});return width;
}
inline std::string fitText(std::string value,int height,int width) {
    // Discard a partial UTF-8 tail if a caller supplied a byte-truncated message.
    if(!value.empty()) {
        std::size_t start=value.size()-1;
        while(start>0 && (static_cast<unsigned char>(value[start])&0xc0)==0x80) --start;
        const auto lead=static_cast<unsigned char>(value[start]);
        const std::size_t length=lead<0x80?1:lead<0xe0?2:lead<0xf0?3:4;
        if(value.size()-start<length) value.resize(start);
    }
    if(textWidth(value,height)<=width) return value;
    while(!value.empty()) {
        std::size_t start=value.size()-1;
        while(start>0 && (static_cast<unsigned char>(value[start])&0xc0)==0x80) --start;
        value.resize(start);
        if(textWidth(value+"…",height)<=width) return value+"…";
    }
    return {};
}
inline void text(cv::Mat& canvas,const std::string& value,cv::Point origin,int height,cv::Scalar color) {
    const int width=canvas.cols-origin.x-6;if(width<=0 || origin.y<height || origin.y>=canvas.rows) return;
    const auto clipped=fitText(value,height,width);int baseline=0;
    runs(clipped,[&](const std::string& run,bool latin) {
        auto f=font(latin);f->putText(canvas,run,origin,height,color,-1,cv::LINE_AA,true);
        origin.x+=f->getTextSize(run,height,-1,&baseline).width;
    });
}
inline std::string sideName(const std::string& side) {return side=="left"?"左相机":"右相机";}
inline std::string modelName(const std::string& model) {
    return model=="auto"?"自动":model=="full"?"完整五项":model=="narrow"?"窄视场":"仅 k1";
}
} // namespace calibration_ui
