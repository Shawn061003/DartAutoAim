#include "Computation/StereoscopicMeasurement.h"
#include "Computation/LocationTransform.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
CameraParams camera() {
    CameraParams p;
    p.width = 1280; p.height = 960;
    p.cameraMatrix = (cv::Mat_<double>(3,3) << 1000,0,640, 0,1000,480, 0,0,1);
    p.distCoeffs = cv::Mat::zeros(5,1,CV_64F);
    return p;
}
StereoscopicMeasurement estimator() {
    return StereoscopicMeasurement(camera(),camera(),cv::Mat::eye(3,3,CV_64F),
        (cv::Mat_<double>(3,1) << -500,0,0));
}
// 简单无畸变双目，BASE=24.4m、OUTPOST=15.2m，纵向分离以避免交叉配对。
GuideLightDetect::StereoDetectResult observation(int frame, cv::Point2f leftShift = {},
    cv::Point2f rightShift = {}, bool withBase = true) {
    GuideLightDetect::StereoDetectResult result;
    for (auto side : {CameraSide::Left,CameraSide::Right}) {
        auto& targets = side == CameraSide::Left ? result.left : result.right;
        auto add = [&](cv::Point3d p, cv::Point2f shift) {
            if (side == CameraSide::Left) p.x += 500;
            GuideLightDetect::GuideLightDetectResult d;
            d.status = GuideLightDetect::DetectStatus::SUCCESS;
            d.CenterPoint = cv::Point2f(640+1000*p.x/p.z,480+1000*p.y/p.z)+shift;
            d.frame_id = frame;
            d.timestamp_ms = frame*10+(side==CameraSide::Right ? 1:0);
            d.camera_side = side;
            targets.push_back(d);
        };
        if (withBase) add({0,0,24400},side==CameraSide::Left?leftShift:rightShift);
        add({2000,3000,15200},{});
        // 每次改变输入排序；稳定记录必须跟随BASE分类，不跟随检测下标。
        if ((frame+(side==CameraSide::Left?0:1))%2) std::reverse(targets.begin(),targets.end());
    }
    return result;
}
std::vector<StereoscopicMeasurement::StereoscopicMeasurementResult> observe(
    StereoscopicMeasurement& m,const GuideLightDetect::StereoDetectResult& detections) {
    auto results=m.RunMeasurement(detections);
    m.UpdateStability(detections,results);
    return results;
}
bool baseStable(StereoscopicMeasurement& m,int frame,cv::Point2f l={},cv::Point2f r={}) {
    const auto results=observe(m,observation(frame,l,r));
    require(results.size()==2,"Synthetic pair lost BASE/OUTPOST");
    const auto base=std::find_if(results.begin(),results.end(),[](const auto& r){return r.TargetName==TargetClasses::BASE;});
    const auto outpost=std::find_if(results.begin(),results.end(),[](const auto& r){return r.TargetName==TargetClasses::OUTPOST;});
    require(base!=results.end() && outpost!=results.end() && outpost->stable,"OUTPOST must be stable immediately");
    const LocationTransform transform(cv::Mat::eye(3,3,CV_64F),cv::Mat::zeros(3,1,CV_64F));
    const auto transformed=transform.RunTransform(results);
    require(transformed[0].stable==results[0].stable && transformed[1].stable==results[1].stable,
        "Coordinate transform dropped stable tag");
    return base->stable;
}
}
int main() {
    try {
        auto m=estimator();
        // 未满5次不判稳，小抖动可接受；第6次仍保留最近5次。
        for(int i=1;i<=6;++i) {
            const cv::Point2f jitter(i%2?0.f:.5f,i%3*.1f);
            require(baseStable(m,i,jitter,jitter)==(i>=5),"Five-sample warmup/sliding window failed");
        }
        m.ResetStability();
        // 每次只移动0.6px，但窗口首尾相差2.4px，不能按相邻帧判稳。
        for(int i=1;i<=7;++i) {
            const cv::Point2f shift((i-1)*.6f,0);
            require(!baseStable(m,10+i,shift,shift),"Cumulative motion was accepted");
        }
        m.ResetStability();
        for(int i=1;i<=4;++i) require(!baseStable(m,30+i),"Premature stability");
        require(!baseStable(m,35,{2,0},{2,0}),"Exactly 2px must not satisfy strict threshold");
        m.ResetStability();
        for(int i=1;i<=4;++i) baseStable(m,40+i);
        require(!baseStable(m,45,{1.5,1.5},{1.5,1.5}),"Diagonal Euclidean motion was accepted");
        for(bool moveLeft : {false,true}) {
            m.ResetStability();
            for(int i=1;i<=4;++i) baseStable(m,50+i);
            require(!baseStable(m,55,moveLeft?cv::Point2f(1.5,1.5):cv::Point2f(),
                moveLeft?cv::Point2f():cv::Point2f(1.5,1.5)),"Both cameras must satisfy threshold");
        }
        m.ResetStability();
        baseStable(m,60);
        // 旧位置须滑出队列后才能恢复稳定，不能无限保留最早观测。
        for(int i=1;i<=5;++i)
            require(baseStable(m,60+i,{3,0},{3,0})==(i==5),"Old sample did not expire at capacity 5");
        // 重复帧不能充当新观测；时间倒退同样清空。
        require(!baseStable(m,65,{3,0},{3,0}),"Duplicate timestamps advanced stability");
        for(int i=1;i<=5;++i)
            require(baseStable(m,65+i,{3,0},{3,0})==(i==5),"Duplicate frame did not invalidate history");
        require(!baseStable(m,69,{3,0},{3,0}),"Out-of-order timestamps retained stable history");
        // BASE丢失时，OUTPOST仍可用，但不能把之前的BASE历史带到重新出现之后。
        const auto outpostOnly=observe(m,observation(80,{},{},false));
        require(outpostOnly.size()==1 && outpostOnly[0].TargetName==TargetClasses::OUTPOST && outpostOnly[0].stable,
            "Missing BASE blocked stationary OUTPOST");
        for(int i=1;i<=5;++i) require(baseStable(m,80+i)==(i==5),"Missing BASE did not clear history");
        require(observe(m,{}).empty(),"Missing stereo input produced a measurement");
        require(!baseStable(m,90),"Empty input did not reset stable history");
        m.ResetStability();
        require(!baseStable(m,91),"Explicit reset failed");
        // 单独测距只输出几何结果，无效观测可供显示而不会先进入稳定性队列。
        m.ResetStability();
        for(int i=1;i<=4;++i) require(!baseStable(m,100+i),"Premature stability");
        const auto geometryOnly=m.RunMeasurement(observation(105,{4,0},{4,0}));
        require(geometryOnly.size()==2 && !geometryOnly[0].stable && geometryOnly[1].stable,
            "Geometry-only measurement changed stability tags");
        require(baseStable(m,106),"Geometry-only measurement polluted observation history");
        auto separate=estimator();
        require(!baseStable(separate,100),"Instances shared stability state");
        std::cout << "BASE stereo stability, rolling window, resets and tag propagation: PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';return 1;
    }
}