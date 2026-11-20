#pragma once
#include "stereo_core.hpp"
#include "Acquisition/AcquireFrame.h"
#include <iostream>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <sstream>
#include "../calibration_dataset.hpp"

namespace stereo_workflow {
using namespace stereo_calibration;
using namespace stereo_live;
struct Settings {
    enum class Mode { Collect,Calibration };
    Mode mode=Mode::Calibration;
    fs::path config=fs::path(DART_PROJECT_ROOT)/"config/dart.yaml";
    fs::path input=fs::path(DART_PROJECT_ROOT)/"sub_module/StereoCalibration/Input";
    fs::path output;
    int pairMs=-1,minPairs=10,maxPairs=30;
    double interval=1.;
    bool fixedIntrinsics=false;
    bool help=false;
};
inline Settings ParseSettings(int argc,char** argv) {
    cv::CommandLineParser parser(argc,argv,
        "{help h | | Show help without opening cameras}"
        "{collect | | Collect receive-time paired native BMPs; no target detection}"
        "{calibration | | Offline paired-image ROI annotation and bounded stereo calibration}"
        "{fixed-intrinsics | | Use the left and right K/D from dart.yaml to solve stereo extrinsics}"
        "{config | | dart.yaml path}"
        "{input | | Dataset root containing left/ and right/; default: StereoCalibration/Input}"
        "{output | | NEW calibration result directory}"
        "{pair-ms | -1 | Collection receive-time tolerance; default: dart.yaml}"
        "{interval | 1.0 | Seconds between automatic saved BMP pairs}"
        "{min-pairs | 10 | Minimum pairs for quality checks}"
        "{max-pairs | 30 | Maximum selected calibration pairs}");
    Settings s;
    if (parser.has("help")) {
        parser.printMessage();
        std::cout<<"./build/stereo_calibration --collect [--pair-ms=20 --interval=1]\n"
                 <<"./build/stereo_calibration --calibration [--fixed-intrinsics] [--input=/path/to/Input]\n"
                 <<"Board: 6x6 tag36h11, 88 mm tags, 26.4 mm gaps, borderBits=2.\n"
                 <<"Bounds in RIGHT optical frame (right/down/forward): |Tx|=450..570, |Ty|<=60, |Tz|<=160 mm.\n"
                 <<"Collection: A=auto/pause, SPACE=save pair, ESC/Q=quit.\n"
                 <<"Calibration: LMB=ROI, wheel=zoom, RMB click=zoom in, Shift+RMB=zoom out, RMB drag=pan, SPACE=accept+next, N=skip, B=back, P=next,\n"
                 <<"R=clear ROIs, D=remove from fit, U=undo, S=save, 0=reset zoom, 1/2=L zoom, 3/4=R zoom, [/]=list scroll, ESC/Q=quit.\n";
        s.help=true;return s;
    }
    const std::vector<std::string> flags{"collect","calibration","fixed-intrinsics"};
    const std::vector<std::string> options{"config","input","output","pair-ms","interval","min-pairs","max-pairs"};
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];const auto equal=arg.find('=');
        Require(arg.rfind("--",0)==0,"Use --collect / --calibration and --name=value options");
        const auto key=arg.substr(2,equal==std::string::npos?equal:equal-2);
        if (std::find(flags.begin(),flags.end(),key)!=flags.end()) {
            Require(equal==std::string::npos,"Use --"+key+" without a value");
        } else {
            Require(std::find(options.begin(),options.end(),key)!=options.end(),"Unknown option: "+key);
            Require(equal!=std::string::npos,"Use --name=value options");
        }
    }
    Require(parser.has("collect")!=parser.has("calibration"),"Choose exactly one of --collect or --calibration");
    s.mode=parser.has("collect")?Settings::Mode::Collect:Settings::Mode::Calibration;
    s.fixedIntrinsics=parser.has("fixed-intrinsics");
    Require(!s.fixedIntrinsics || s.mode==Settings::Mode::Calibration,"--fixed-intrinsics requires --calibration");
    if(parser.has("config")) s.config=parser.get<std::string>("config");
    if(parser.has("input")) s.input=parser.get<std::string>("input");
    s.output=parser.get<std::string>("output");
    s.pairMs=parser.get<int>("pair-ms");s.interval=parser.get<double>("interval");
    s.minPairs=parser.get<int>("min-pairs");s.maxPairs=parser.get<int>("max-pairs");
    Require(parser.check(),"Invalid command-line value");
    Require(!s.input.empty() && !s.config.empty() && std::isfinite(s.interval) && s.interval>=.1 && s.interval<=3600 &&
        s.pairMs>=-1 && s.pairMs<=1000 && s.minPairs>=3 && s.maxPairs>=s.minPairs && s.maxPairs<=100,"Invalid paths/time/sample limits");
    s.input=fs::absolute(s.input);s.config=fs::absolute(s.config);
    if(s.output.empty()) {
        const auto now=std::time(nullptr);std::tm local{};localtime_r(&now,&local);
        std::ostringstream name;name<<"calibration_"<<std::put_time(&local,"%Y%m%d_%H%M%S")<<'_'<<Clock::now().time_since_epoch().count()%100000;
        s.output=s.input.parent_path()/"Results"/name.str();
    }
    s.output=fs::absolute(s.output);return s;
}

inline std::string PairKey(std::string stem,bool left) {
    std::string lower=stem;std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return std::tolower(c);});
    const std::string prefix=left?"left":"right";
    if(lower.rfind(prefix,0)==0) {stem=stem.substr(prefix.size());while(!stem.empty() && (stem.front()=='_' || stem.front()=='-')) stem.erase(stem.begin());}
    Require(!stem.empty(),"Image name has no pair ID");
    if(std::all_of(stem.begin(),stem.end(),[](unsigned char c){return std::isdigit(c);})) {
        const auto first=stem.find_first_not_of('0');return first==std::string::npos?"0":stem.substr(first);
    }
    return stem;
}
inline bool KeyLess(const std::string& a,const std::string& b) {
    const auto digits=[](const std::string& s){return std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isdigit(c);});};
    if(digits(a) && digits(b) && a.size()!=b.size()) return a.size()<b.size();
    return a<b;
}
struct Record {
    enum class State { Pending,Accepted,Skipped };
    std::string key;
    LivePair sample;
    cv::Rect leftRoi,rightRoi;
    State state=State::Pending;
    aprilgrid::DetectionInfo leftManualMarkers,rightManualMarkers;
};
inline std::map<std::string,fs::path> IndexSide(const fs::path& path,bool left,bool optional=false) {
    if(optional && !fs::exists(path)) return {};
    Require(fs::is_directory(path),"Missing input directory: "+path.string());
    std::map<std::string,fs::path> result;
    for(const auto& image:calibration_dataset::images(path)) {
        const auto key=PairKey(image.stem().string(),left);
        Require(result.emplace(key,image).second,"Duplicate pair ID "+key+" in "+path.string());
    }
    return result;
}
struct Dataset {std::vector<Record> records;int pairMs=-1;};
inline std::map<std::string,fs::path> StoredSide(const fs::path& input,bool left) {
    const auto side=left?"left":"right";
    auto files=IndexSide(input/side,left);
    for(const auto& [key,path]:IndexSide(input/"selected"/side,left,true))
        Require(files.emplace(key,path).second,"Duplicate pair ID "+key+" across input and selected images");
    return files;
}
inline Dataset ReadDataset(const fs::path& input,bool allowEmpty=false) {
    const auto left=StoredSide(input,true),right=StoredSide(input,false);
    Dataset result;
    for(const auto& [key,path]:left) {
        const auto found=right.find(key);
        if(found==right.end()) {std::cerr<<"Ignoring unpaired LEFT image: "<<path<<'\n';continue;}
        Record r;r.key=key;r.sample.timestampsKnown=false;r.sample.pair.leftPath=path;r.sample.pair.rightPath=found->second;r.sample.pair.group=0;
        result.records.push_back(std::move(r));
    }
    for(const auto& [key,path]:right) if(!left.count(key)) std::cerr<<"Ignoring unpaired RIGHT image: "<<path<<'\n';
    std::sort(result.records.begin(),result.records.end(),[](const Record& a,const Record& b){return KeyLess(a.key,b.key);});
    Require(allowEmpty || !result.records.empty(),"No paired BMP images in "+input.string());
    std::map<std::string,LivePair> metadata;
    const auto manifest=input/"dataset.yaml";
    if(fs::exists(manifest)) {
        cv::FileStorage f(manifest.string(),cv::FileStorage::READ);Require(f.isOpened(),"Cannot read dataset.yaml");
        Require(static_cast<int>(f["schema_version"])==1,"Unsupported dataset manifest");
        result.pairMs=static_cast<int>(f["max_host_timestamp_diff_ms"]);
        Require(result.pairMs>=0 && result.pairMs<=1000,"Invalid collection manifest time tolerance");
        for(const auto& n:f["pairs"]) {
            if(!n["frame_timestamps_available"].empty() && static_cast<int>(n["frame_timestamps_available"])==0) continue;
            LivePair p;p.leftId=std::stoull(static_cast<std::string>(n["left_frame_id"]));p.rightId=std::stoull(static_cast<std::string>(n["right_frame_id"]));
            p.leftTime=std::stoll(static_cast<std::string>(n["left_receive_timestamp_ms"]));p.rightTime=std::stoll(static_cast<std::string>(n["right_receive_timestamp_ms"]));
            Require(TimestampDifference(p.leftTime,p.rightTime)<=static_cast<std::uint64_t>(result.pairMs),"Manifest contains an out-of-tolerance pair");
            Require(metadata.emplace(static_cast<std::string>(n["id"]),p).second,"Duplicate manifest pair ID");
        }
    }
    for(std::size_t i=0;i<result.records.size();++i) {
        auto& r=result.records[i];r.sample.pair.number=static_cast<int>(i+1);
        const auto it=metadata.find(r.key);
        if(it!=metadata.end()) {
            r.sample.leftId=it->second.leftId;r.sample.rightId=it->second.rightId;r.sample.leftTime=it->second.leftTime;r.sample.rightTime=it->second.rightTime;r.sample.timestampsKnown=true;
        }
    }
    return result;
}

class PairWriter {
    fs::path root_;
    int tolerance_;
    std::vector<LivePair> saved_;
    std::vector<std::string> ids_;
    std::uint64_t next_=1;
    int manifestTolerance_=0;
    cv::Size leftSize_,rightSize_;
    void reload() {
        const auto data=ReadDataset(root_,true);
        saved_.clear();ids_.clear();std::vector<std::string> keys;
        for(const auto& record:data.records) {saved_.push_back(record.sample);ids_.push_back(record.key);}
        for(bool left:{true,false}) for(const auto& [key,path]:StoredSide(root_,left)) keys.push_back(key);
        if(fs::exists(root_/"dataset.yaml")) {
            cv::FileStorage previous((root_/"dataset.yaml").string(),cv::FileStorage::READ);
            for(const auto& node:previous["pairs"]) keys.push_back(static_cast<std::string>(node["id"]));
        }
        next_=calibration_dataset::nextNumber(keys);
        manifestTolerance_=std::max(tolerance_,data.pairMs);
        leftSize_={};rightSize_={};
        if(!saved_.empty()) {
            const auto left=cv::imread(saved_.front().pair.leftPath.string(),cv::IMREAD_UNCHANGED);
            const auto right=cv::imread(saved_.front().pair.rightPath.string(),cv::IMREAD_UNCHANGED);
            Require(!left.empty() && !right.empty(),"Cannot read existing capture dimensions");
            leftSize_=left.size();rightSize_=right.size();
        }
    }
    void manifest() const {
        cv::FileStorage f((root_/"dataset.pending.yaml").string(),cv::FileStorage::WRITE);
        Require(f.isOpened(),"Cannot write collection manifest");
        f<<"schema_version"<<1<<"max_host_timestamp_diff_ms"<<manifestTolerance_
         <<"timestamp_semantics"<<"host steady_clock receive time; NOT synchronized exposure"<<"pairs"<<"[";
        for(std::size_t i=0;i<saved_.size();++i) {
            const auto& p=saved_[i];f<<"{"<<"id"<<ids_[i]<<"frame_timestamps_available"<<static_cast<int>(p.timestampsKnown)
                <<"left_image"<<fs::relative(p.pair.leftPath,root_).string()<<"right_image"<<fs::relative(p.pair.rightPath,root_).string()
                <<"left_frame_id"<<std::to_string(p.leftId)<<"right_frame_id"<<std::to_string(p.rightId)
                <<"left_receive_timestamp_ms"<<std::to_string(p.leftTime)<<"right_receive_timestamp_ms"<<std::to_string(p.rightTime)
                <<"host_timestamp_diff_ms"<<static_cast<int>(TimestampDifference(p.leftTime,p.rightTime))<<"}";
        }
        f<<"]";f.release();fs::rename(root_/"dataset.pending.yaml",root_/"dataset.yaml");
    }
public:
    PairWriter(fs::path root,int tolerance):root_(std::move(root)),tolerance_(tolerance) {
        Require(tolerance>=0 && tolerance<=1000,"Invalid pairing tolerance");
        calibration_dataset::Lock lock(root_);
        fs::create_directories(root_/"left");fs::create_directories(root_/"right");reload();
    }
    std::size_t count() const {return saved_.size();}
    void append(const StereoFrame& pair) {
        calibration_dataset::Lock lock(root_);reload();
        Require(!pair.left.image.empty() && !pair.right.image.empty(),"Cannot save an incomplete pair");
        Require(TimestampDifference(pair.left.timestamp_ms,pair.right.timestamp_ms)<=static_cast<std::uint64_t>(tolerance_),"Pair exceeds receive-time tolerance");
        Require((leftSize_.empty() || leftSize_==pair.left.image.size()) &&
                (rightSize_.empty() || rightSize_==pair.right.image.size()),"New camera dimensions differ from existing capture images");
        const auto digits=std::to_string(next_);const auto id=std::string(digits.size()<6?6-digits.size():0,'0')+digits;
        const fs::path l=root_/"left"/(id+".bmp"),r=root_/"right"/(id+".bmp");
        const fs::path lp=root_/"left"/(id+".pending.bmp"),rp=root_/"right"/(id+".pending.bmp");
        Require(!fs::exists(l) && !fs::exists(r),"Refusing to overwrite collected image pair");
        bool committed=false;
        try {
            Require(cv::imwrite(lp.string(),pair.left.image) && cv::imwrite(rp.string(),pair.right.image),"Cannot write paired BMPs");
            fs::rename(lp,l);fs::rename(rp,r);
            LivePair p;p.pair.leftPath=l;p.pair.rightPath=r;p.leftId=pair.left.frame_id;p.rightId=pair.right.frame_id;
            p.leftTime=pair.left.timestamp_ms;p.rightTime=pair.right.timestamp_ms;saved_.push_back(p);ids_.push_back(std::to_string(next_));committed=true;
            manifest();
        } catch(...) {
            if(committed) {saved_.pop_back();ids_.pop_back();}
            for(const auto& path:{lp,rp,l,r}) {std::error_code ignored;fs::remove(path,ignored);}
            throw;
        }
    }
};

inline std::string FileStamp(const fs::path& p) {
    return std::to_string(fs::file_size(p))+":"+std::to_string(fs::last_write_time(p).time_since_epoch().count());
}
inline void SaveAnnotations(const fs::path& path,const std::vector<Record>& records) {
    const auto pending=path.parent_path()/(path.stem().string()+".pending.yaml");
    cv::FileStorage f(pending.string(),cv::FileStorage::WRITE);Require(f.isOpened(),"Cannot save annotations");
    aprilgrid::writeSpecification(f);
    f<<"schema_version"<<2<<"records"<<"[";
    for(const auto& r:records) {
        f<<"{"<<"id"<<r.key<<"state"<<static_cast<int>(r.state)<<"left_file"<<r.sample.pair.leftPath.filename().string()
         <<"right_file"<<r.sample.pair.rightPath.filename().string()<<"left_stamp"<<FileStamp(r.sample.pair.leftPath)<<"right_stamp"<<FileStamp(r.sample.pair.rightPath)
         <<"left_roi"<<r.leftRoi<<"right_roi"<<r.rightRoi<<"left_points"<<r.sample.pair.left<<"right_points"<<r.sample.pair.right
         <<"corner_ids"<<r.sample.pair.pointIds
         <<"right_reversed"<<static_cast<int>(r.sample.pair.rightReversed);
        aprilgrid::writeManualMarkers(f,"left_manual_markers",r.leftManualMarkers);
        aprilgrid::writeManualMarkers(f,"right_manual_markers",r.rightManualMarkers);f<<"}";
    }
    f<<"]";f.release();fs::rename(pending,path);
}
inline bool ValidCorners(const std::vector<cv::Point2f>& points,cv::Size size,const std::vector<int>& ids={}) {
    if(!aprilgrid::validIds(ids,points.size())) return false;
    return std::all_of(points.begin(),points.end(),[&](const cv::Point2f& p){return std::isfinite(p.x) && std::isfinite(p.y) && p.x>=0 && p.y>=0 && p.x<size.width && p.y<size.height;});
}
inline void LoadAnnotations(const fs::path& path,std::vector<Record>& records,cv::Size left,cv::Size right) {
    if(!fs::exists(path)) return;
    cv::FileStorage f(path.string(),cv::FileStorage::READ);Require(f.isOpened(),"Cannot read annotations");
    Require(static_cast<int>(f["schema_version"])==2 && aprilgrid::matchesSpecification(f),
        "Annotation AprilGrid specification mismatch; reannotate old checkerboard datasets");
    std::map<std::string,Record*> byKey;for(auto& r:records) byKey[r.key]=&r;
    for(const auto& n:f["records"]) {
        const auto it=byKey.find(static_cast<std::string>(n["id"]));if(it==byKey.end()) continue;
        auto& r=*it->second;
        if(static_cast<std::string>(n["left_file"])!=r.sample.pair.leftPath.filename().string() ||
           static_cast<std::string>(n["right_file"])!=r.sample.pair.rightPath.filename().string() ||
           static_cast<std::string>(n["left_stamp"])!=FileStamp(r.sample.pair.leftPath) || static_cast<std::string>(n["right_stamp"])!=FileStamp(r.sample.pair.rightPath)) {
            std::cerr<<"Ignoring stale annotation for pair "<<r.key<<'\n';continue;
        }
        cv::Rect lr,rr;n["left_roi"]>>lr;n["right_roi"]>>rr;
        if((lr&cv::Rect({},left))!=lr || (rr&cv::Rect({},right))!=rr) continue;
        std::vector<cv::Point2f> lp,rp;n["left_points"]>>lp;n["right_points"]>>rp;
        std::vector<int> ids;if(!n["corner_ids"].empty()) n["corner_ids"]>>ids;
        if((!lp.empty() && !ValidCorners(lp,left,ids)) || (!rp.empty() && !ValidCorners(rp,right,ids))) continue;
        const int state=static_cast<int>(n["state"]);
        if(state<0 || state>2 || (state==static_cast<int>(Record::State::Accepted) &&
           (lp.size()!=rp.size() || !ValidCorners(lp,left,ids) || !ValidCorners(rp,right,ids)))) continue;
        r.leftRoi=lr;r.rightRoi=rr;r.sample.pair.left=std::move(lp);r.sample.pair.right=std::move(rp);
        r.sample.pair.rightReversed=static_cast<int>(n["right_reversed"])!=0;r.state=static_cast<Record::State>(state);
        r.sample.pair.pointIds=std::move(ids);
        r.leftManualMarkers=aprilgrid::readManualMarkers(n["left_manual_markers"],left);
        r.rightManualMarkers=aprilgrid::readManualMarkers(n["right_manual_markers"],right);
    }
}
inline std::vector<LivePair> Selected(const std::vector<Record>& records) {
    std::vector<LivePair> result;for(const auto& r:records) if(r.state==Record::State::Accepted) result.push_back(r.sample);return result;
}

// Mouse coordinates and annotations always map through the native image, including zoom/pan/letterboxing.
struct Viewport {
    cv::Rect area;cv::Size native;double zoom=1.;cv::Point2d center;
    void reset(cv::Size size) {native=size;zoom=1.;center={size.width/2.,size.height/2.};}
    double scale() const {return native.empty()?1.:std::min(static_cast<double>(area.width)/native.width,static_cast<double>(area.height)/native.height)*zoom;}
    cv::Point2d project(cv::Point2d p) const {return (p-center)*scale()+cv::Point2d(area.x+area.width/2.,area.y+area.height/2.);}
    cv::Point2d unproject(cv::Point2d p) const {return (p-cv::Point2d(area.x+area.width/2.,area.y+area.height/2.))*(1./scale())+center;}
    void clamp() {
        const double hx=area.width/(2*scale()),hy=area.height/(2*scale());
        center.x=2*hx>=native.width?native.width/2.:std::clamp(center.x,hx,native.width-hx);
        center.y=2*hy>=native.height?native.height/2.:std::clamp(center.y,hy,native.height-hy);
    }
    void magnify(cv::Point2d cursor,double factor) {
        if(native.empty()) return;
        const auto point=unproject(cursor);zoom=std::clamp(zoom*factor,1.,32.);
        center=point-(cursor-cv::Point2d(area.x+area.width/2.,area.y+area.height/2.))*(1./scale());clamp();
    }
    cv::Point2d clipped(cv::Point2d p) const {return {std::clamp(p.x,0.,static_cast<double>(native.width)),std::clamp(p.y,0.,static_cast<double>(native.height))};}
    cv::Rect roi(cv::Point2d a,cv::Point2d b) const {
        a=clipped(unproject(a));b=clipped(unproject(b));
        return cv::Rect(cv::Point(static_cast<int>(std::floor(std::min(a.x,b.x))),static_cast<int>(std::floor(std::min(a.y,b.y)))),
                        cv::Point(static_cast<int>(std::ceil(std::max(a.x,b.x))),static_cast<int>(std::ceil(std::max(a.y,b.y)))))&cv::Rect({},native);
    }
};
inline bool DetectRoi(const cv::Mat& raw,cv::Rect roi,std::vector<cv::Point2f>& points,aprilgrid::DetectionInfo* info=nullptr) {
    points.clear();if(info) *info={};
    if(raw.empty() || roi.width<24 || roi.height<24 || (roi&cv::Rect({},raw.size()))!=roi) return false;
    aprilgrid::DetectionInfo detected;
    const bool found=aprilgrid::detect(raw(roi),points,&detected,true);
    const cv::Point2f offset(static_cast<float>(roi.x),static_cast<float>(roi.y));
    detected.translate(offset);if(info) *info=detected;
    if(!found) return false;
    for(auto& p:points) p+=offset;
    if(!ValidCorners(points,raw.size(),aprilgrid::pointIds(detected))) {points.clear();return false;}return true;
}
struct Detection {
    std::size_t index=0;std::uint64_t epoch=0;
    std::vector<cv::Point2f> left,right;
    std::vector<int> pointIds;
    aprilgrid::DetectionInfo leftMarkers,rightMarkers;
    bool leftFound=false,rightFound=false,reversed=false;
    double milliseconds=0;
};
inline Detection DetectPair(cv::Mat left,cv::Mat right,cv::Rect lr,cv::Rect rr,std::size_t index,std::uint64_t epoch) {
    const auto started=Clock::now();Detection result;result.index=index;result.epoch=epoch;
    result.leftFound=DetectRoi(left,lr,result.left,&result.leftMarkers);result.rightFound=DetectRoi(right,rr,result.right,&result.rightMarkers);
    aprilgrid::correspond(result.leftMarkers,result.rightMarkers,result.left,result.right,result.pointIds);
    result.leftFound=ValidCorners(result.left,left.size(),result.pointIds);
    result.rightFound=ValidCorners(result.right,right.size(),result.pointIds);
    result.milliseconds=std::chrono::duration<double,std::milli>(Clock::now()-started).count();return result;
}
} // namespace stereo_workflow
