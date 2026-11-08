#include "Computation/LocationTransform.h"
#include "Computation/YawSolver.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(cv::Point3d actual, cv::Point3d expected, const char* message) {
    check(cv::norm(actual - expected) < 1e-8, message);
}
}
int main() {
    try {
        cv::Mat r = cv::Mat::eye(3,3,CV_64F), t = cv::Mat::zeros(3,1,CV_64F);
        const LocationTransform identity(r,t);
        using Measurement = StereoscopicMeasurement::StereoscopicMeasurementResult;
        Measurement near; near.RealLocation = {-160,16300,570}; near.TargetName = TargetClasses::OUTPOST;
        Measurement far; far.RealLocation = {-240,24460,640}; far.TargetName = TargetClasses::BASE;
        // Golden values in the current muzzle frame, in mm.
        auto result = identity.RunTransform({near},false);
        check(result.size()==1 && result[0].TargetName==near.TargetName,"Near result metadata lost");
        close(result[0].RealLocation,{-149.96897952609032,16258.902819878471,570},"Near model, units or frame incorrect");
        result = identity.RunTransform({far},true);
        close(result[0].RealLocation,{312.408,25558.232,640},"Far model, units or frame incorrect");
        const auto yaw = YawSolver{}.RunSolve(result,true);
        check(yaw && std::abs(*yaw-std::atan2(312.408,25558.232))<1e-12,
              "A fixed calibration yaw leaked into the downstream relative yaw");
        result=identity.RunTransform({near,far},true);
        check(result.size()==2 && result[0].TargetName==TargetClasses::OUTPOST && result[1].TargetName==TargetClasses::BASE,
              "Batch order, count or labels changed");
        check(identity.RunTransform({},true).empty() && identity.RunTransform({},false).empty(),"Empty batch created stale results");
        // R/T precedes compensation; Z follows the 3D transform, not an XY model.
        r=(cv::Mat_<double>(3,3)<<1,0,0,0,0,1,0,-1,0);
        t=(cv::Mat_<double>(3,1)<<240,-18.83,231.45);
        const LocationTransform transform(r,t);
        r.setTo(0); t.setTo(0); // Constructor must retain independent extrinsics.
        Measurement camera; camera.RealLocation={500,-800,24000}; camera.TargetName=TargetClasses::BASE;
        result=transform.RunTransform({camera},true);
        close(result[0].RealLocation,{1364.242,25200.138,1031.45},"R/T order or Z preservation incorrect");
        result=transform.RunTransform({camera},false);
        close(result[0].RealLocation,{715.41674174904892,24578.760089513053,1031.45},"Aim mode did not select the other model");
        close(camera.RealLocation,{500,-800,24000},"Input observation mutated");
        close(transform.RunTransform({camera},true)[0].RealLocation,
              {1364.242,25200.138,1031.45},"Repeated calls accumulated compensation");
        std::cout<<"Location transform compensation checks passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
