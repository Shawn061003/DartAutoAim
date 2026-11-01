#pragma once
#include "../sub_module/aprilgrid.hpp"

// 渲染与实物相同的ID方向和双层黑边，返回按ID排列的理想像素角点。
inline std::vector<cv::Point2f> renderAprilGrid(cv::Mat& image,cv::Point origin,int tagPixels) {
    std::vector<cv::Point2f> expected;
    const int gap=cvRound(tagPixels*.3);
    for(int id=0;id<36;++id) {
        const int x=origin.x+(5-id%6)*(tagPixels+gap),y=origin.y+(id/6)*(tagPixels+gap);
        cv::Mat marker,bgr;
        cv::aruco::drawMarker(aprilgrid::dictionary(),id,tagPixels,marker,2);
        cv::cvtColor(marker,bgr,cv::COLOR_GRAY2BGR);
        bgr.copyTo(image(cv::Rect(x,y,tagPixels,tagPixels)));
        const float x0=x-.5f,y0=y-.5f,x1=x+tagPixels-.5f,y1=y+tagPixels-.5f;
        expected.insert(expected.end(),{{x0,y0},{x1,y0},{x1,y1},{x0,y1}});
    }
    return expected;
}
