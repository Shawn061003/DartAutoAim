#pragma once
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
