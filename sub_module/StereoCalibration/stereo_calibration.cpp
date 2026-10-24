// Split capture/annotation workflow. All calibration points remain in native sensor coordinates.
#include "workflow.hpp"
#include "../calibration_ui.hpp"
#include "Acquisition/AcquireFrame.h"
#include "Acquisition/Quit.h"
#include "common/DartConfig.h"
#include <atomic>
#include <mutex>
#include <thread>

namespace stereo_workflow {
constexpr const char* kReviewWindow="双目标定 - 配对图片与 ROI";
constexpr const char* kCollectWindow="双目标定 - 采集配对 BMP";
const cv::Scalar kWhite(225,225,225),kGreen(75,225,90),kYellow(70,215,255);
void Label(cv::Mat& canvas,const std::string& value,int x,int y,cv::Scalar color=kWhite,double size=.52) {
    calibration_ui::text(canvas,value,{x,y},std::max(14,static_cast<int>(std::lround(size*34))),color);
}
void DrawView(cv::Mat& canvas,const cv::Mat& raw,const Viewport& view,cv::Rect roi,const std::vector<cv::Point2f>& corners) {
    auto pane=canvas(view.area);pane.setTo(cv::Scalar(18,18,18));if(raw.empty()) return;
    const double s=view.scale();const cv::Mat affine=(cv::Mat_<double>(2,3)<<s,0,view.area.width/2.-s*view.center.x,0,s,view.area.height/2.-s*view.center.y);
    cv::warpAffine(raw,pane,affine,pane.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(18,18,18));
    const cv::Point2d offset(view.area.x,view.area.y);
    if(!roi.empty()) cv::rectangle(pane,view.project(roi.tl())-offset,view.project(roi.br())-offset,kYellow,1,cv::LINE_AA);
    if(!corners.empty()) {
        std::vector<cv::Point2f> display;
        for(const auto& p:corners) display.emplace_back(view.project(p)-offset);
        cv::drawChessboardCorners(pane,{kColumns,kRows},display,corners.size()==kColumns*kRows);
        for(const auto i:{std::size_t(0),corners.size()-1}) cv::putText(pane,std::to_string(i),display[i]+cv::Point2f(5,-5),cv::FONT_HERSHEY_SIMPLEX,.5,kYellow,1,cv::LINE_AA);
    }
}
struct Button {cv::Rect box;std::string caption;int key;};
const std::vector<Button>& Buttons() {
    static const std::vector<Button> result{
        {{15,835,175,42},"接受并下一组",' '},{{200,835,175,42},"跳过并下一组 [N]",'n'},
        {{385,835,145,42},"上一组 [B]",'b'},{{540,835,145,42},"下一组 [P]",'p'},
        {{695,835,180,42},"撤销接受 [U]",'u'},{{885,835,180,42},"保存结果 [S]",'s'},
        {{15,895,210,42},"清除两侧 ROI [R]",'r'},{{235,895,210,42},"移出标定 [D]",'d'},
        {{455,895,220,42},"反转右侧角点顺序 [F]",'f'},{{685,895,190,42},"重置缩放 [0]",'0'},
        {{885,895,180,42},"退出 [Q]",'q'},
        {{15,716,85,28},"左 + [1]",'1'},{{110,716,85,28},"左 - [2]",'2'},
        {{685,716,85,28},"右 + [3]",'3'},{{780,716,85,28},"右 - [4]",'4'},
        {{1370,933,190,32},"列表上滚 [",'['},{{1570,933,190,32},"列表下滚 ]",']'}};
    return result;
}

class LatestCollectionPair {
    CameraInput input_;
    std::atomic<bool> stop_{false};std::thread worker_;std::mutex mutex_;
    std::optional<StereoFrame> latest_;std::exception_ptr error_;
public:
    LatestCollectionPair(const DartCongfig& config,int tolerance):input_(config.leftCamera,config.rightCamera,config.frameQueue.capacity) {
        input_.StartCamera();
        worker_=std::thread([this,tolerance] {
            try {while(!stop_.load()) {
                auto pair=input_.frameQueue().GetFrame(tolerance);
                if(pair) {std::lock_guard<std::mutex> lock(mutex_);latest_=std::move(pair);}
                else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }} catch(...) {std::lock_guard<std::mutex> lock(mutex_);error_=std::current_exception();}
        });
    }
    void stop() {stop_.store(true);if(worker_.joinable()) worker_.join();input_.StopCamera();}
    ~LatestCollectionPair(){stop();}
    std::optional<StereoFrame> take() {
        std::lock_guard<std::mutex> lock(mutex_);if(error_) std::rethrow_exception(error_);
        auto result=std::move(latest_);latest_.reset();return result;
    }
};
cv::Mat CollectionDashboard(const std::optional<StereoFrame>& latest,std::size_t count,bool automatic,int pairMs,double interval,const std::string& status) {
    cv::Mat canvas(850,1600,CV_8UC3,cv::Scalar(25,25,25));
    if(latest) {
        Viewport lv,rv;lv.area={10,50,630,650};rv.area={650,50,630,650};lv.reset(latest->left.image.size());rv.reset(latest->right.image.size());
        DrawView(canvas,latest->left.image,lv,{},{});DrawView(canvas,latest->right.image,rv,{},{});
        Label(canvas,"左相机 #"+std::to_string(latest->left.frame_id),15,30);Label(canvas,"右相机 #"+std::to_string(latest->right.frame_id),655,30);
        Label(canvas,"接收时差："+std::to_string(TimestampDifference(latest->left.timestamp_ms,latest->right.timestamp_ms))+" ms",1300,190);
    }
    Label(canvas,"配对 BMP 采集",1300,45,kGreen);
    Label(canvas,"已保存："+std::to_string(count),1300,90);
    Label(canvas,automatic?"自动保存":"已暂停",1300,125,kYellow);
    Label(canvas,"间隔："+Number(interval,1)+" 秒",1300,155);
    Label(canvas,"配对上限："+std::to_string(pairMs)+" ms",1300,225);
    Label(canvas,"采集时不检测棋盘",1300,270);
    Label(canvas,"接收配对 ≠ 曝光同步",1300,305,kYellow);
    Label(canvas,status.substr(0,145),15,735,kYellow);
    Label(canvas,"A：自动/暂停 | 空格：保存下一组 | Q/ESC：退出；退出时等待图片保存完成",15,785);
    return canvas;
}
int Collect(Settings s) {
    cv::setNumThreads(4);
    const DartCongfig config(s.config.string());if(s.pairMs<0) s.pairMs=config.frameQueue.maxTimestampDiffMs;
    Require(s.pairMs>=0 && s.pairMs<=1000,"Invalid receive-time tolerance");
    CameraStopSignals signals;LatestCollectionPair camera(config,s.pairMs);PairWriter writer(s.input,s.pairMs);
    std::future<std::size_t> saving;std::optional<StereoFrame> latest;
    std::size_t count=0;bool automatic=true,request=false;
    auto lastPairAt=Clock::now(),lastSavedAt=Clock::now()-std::chrono::hours(1);
    std::string status="等待左右相机配对图像";
    std::cout<<"Collect native paired BMPs into "<<s.input<<"; receive tolerance="<<s.pairMs<<" ms, interval="<<s.interval<<" s\n"
             <<"No chessboard detector runs during collection. Keep the board stationary and sharp.\n"
             <<"Timestamps are host receive times, NOT synchronized exposure. A=pause/auto, SPACE=save now, Q=quit.\n";
    cv::namedWindow(kCollectWindow,cv::WINDOW_NORMAL);cv::resizeWindow(kCollectWindow,1600,850);
    try {
        while(!signals.stopRequested()) {
            if(saving.valid() && saving.wait_for(std::chrono::seconds(0))==std::future_status::ready) {count=saving.get();status="已保存图片组 "+std::to_string(count);}
            if(auto pair=camera.take()) {
                lastPairAt=Clock::now();latest=std::move(pair);
                const bool due=automatic && std::chrono::duration<double>(lastPairAt-lastSavedAt).count()>=s.interval;
                if(!saving.valid() && (request || due)) {
                    const auto raw=*latest;lastSavedAt=lastPairAt;request=false;
                    saving=std::async(std::launch::async,[&writer,raw] {writer.append(raw);return writer.count();});
                    status="后台保存原始 BMP 图片组";
                }
            } else if(Clock::now()-lastPairAt>std::chrono::seconds(10))
                throw std::runtime_error("No paired frames for 10 s; verify both cameras and try --pair-ms=20 (host receive tolerance)");
            const auto canvas=CollectionDashboard(latest,count,automatic,s.pairMs,s.interval,status);
            cv::imshow(kCollectWindow,canvas);const int key=cv::waitKey(5)&0xff;
            if(key==27 || key=='q' || cv::getWindowProperty(kCollectWindow,cv::WND_PROP_VISIBLE)==0) break;
            if(key=='a') automatic=!automatic;
            if(key==' ') request=true;
        }
        camera.stop();if(saving.valid()) count=saving.get();cv::destroyWindow(kCollectWindow);
        std::cout<<"Saved "<<count<<" pairs; next run: ./build/stereo_calibration --calibration --input="<<s.input<<'\n';return 0;
    } catch(...) {camera.stop();cv::destroyWindow(kCollectWindow);throw;}
}

class Review {
    Settings settings_;Camera left_,right_;Dataset dataset_;
    std::size_t index_=0;cv::Mat rawLeft_,rawRight_;
    Viewport views_[2];Detection detected_;
    std::future<Detection> detection_;bool needDetection_=false;
    std::future<std::pair<std::uint64_t,Solution>> fitting_;
    std::future<std::uint64_t> saving_;
    Solution solved_;
    std::uint64_t epoch_=0,revision_=0,solvedRevision_=0,savedRevision_=std::numeric_limits<std::uint64_t>::max();
    bool needFit_=false,quit_=false;
    std::vector<std::size_t> undo_;
    int listOffset_=0,drag_=-1,pan_=-1;cv::Point2d start_,cursor_,lastPan_,rightStart_;
    bool rightMoved_=false;int activeSide_=0;calibration_ui::PointerInput pointer_;
    std::string status_="分别框选左右图片中的完整标定板，并保留白色边缘";
    int count(Record::State state) const {return static_cast<int>(std::count_if(dataset_.records.begin(),dataset_.records.end(),[&](const Record& r){return r.state==state;}));}
    Record& current(){return dataset_.records[index_];}
    const Record& current() const {return dataset_.records[index_];}
    void persist(){SaveAnnotations(settings_.input/"annotations.yaml",dataset_.records);}
    void changed() {++revision_;solved_={};needFit_=true;persist();}
    void invalidateCurrent() {
        auto& r=current();const bool included=r.state==Record::State::Accepted;
        r.state=Record::State::Pending;r.sample.pair.left.clear();r.sample.pair.right.clear();
        if(included) changed();
        detected_={};++epoch_;needDetection_=!rawLeft_.empty() && !rawRight_.empty();persist();
    }
    Options reportOptions() const {
        Options o;o.config=settings_.config;o.output=settings_.output;o.leftIntrinsics=settings_.leftIntrinsics;o.rightIntrinsics=settings_.rightIntrinsics;
        o.captureMode="offline_manual_roi_native_full_frames";o.minPairs=settings_.minPairs;o.maxPairs=settings_.maxPairs;o.holdSeconds=0;o.pairMs=dataset_.pairMs;return o;
    }
    void load(std::size_t index) {
        index_=std::min(index,dataset_.records.size()-1);++epoch_;needDetection_=false;drag_=pan_=-1;detected_={};
        rawLeft_.release();rawRight_.release();
        try {
            const auto& r=current();rawLeft_=cv::imread(r.sample.pair.leftPath.string(),cv::IMREAD_COLOR|cv::IMREAD_IGNORE_ORIENTATION);
            rawRight_=cv::imread(r.sample.pair.rightPath.string(),cv::IMREAD_COLOR|cv::IMREAD_IGNORE_ORIENTATION);
            Require(!rawLeft_.empty() && !rawRight_.empty(),"当前 BMP 图片组无法读取，请跳过");
            Require(rawLeft_.size()==left_.referenceSize && rawRight_.size()==right_.referenceSize,"图片尺寸与内参标定尺寸不一致，请跳过或选择匹配的内参");
            views_[0].reset(rawLeft_.size());views_[1].reset(rawRight_.size());
            detected_.left=r.sample.pair.left;detected_.right=r.sample.pair.right;detected_.reversed=r.sample.pair.rightReversed;
            detected_.leftFound=ValidCorners(detected_.left,rawLeft_.size());detected_.rightFound=ValidCorners(detected_.right,rawRight_.size());
            needDetection_=r.state!=Record::State::Accepted && (!r.leftRoi.empty() || !r.rightRoi.empty()) && (!detected_.leftFound || !detected_.rightFound);
            status_=r.state==Record::State::Accepted?"已纳入标定；重新框选后需再次接受此组图片":"分别框选两侧 ROI 并检查角点编号；空格接受，N 跳过";
        } catch(const std::exception& e) {rawLeft_.release();rawRight_.release();views_[0].reset({});views_[1].reset({});status_=e.what();}
    }
    void advance() {if(index_+1<dataset_.records.size()) load(index_+1);else status_="已到最后一组；可检查右侧列表或按 S 保存";}
    void startFit() {
        if(!needFit_ || fitting_.valid()) return;
        needFit_=false;const auto snapshot=Selected(dataset_.records);const auto revision=revision_;
        if(snapshot.size()<3) {solved_={};solvedRevision_=revision_;return;}
        const auto left=left_,right=right_;const int minimum=settings_.minPairs;
        fitting_=std::async(std::launch::async,[snapshot,revision,left,right,minimum] {return std::make_pair(revision,Solve(snapshot,left,right,minimum));});
    }
    void finishFit() {
        auto result=fitting_.get();if(result.first!=revision_) return;
        solved_=std::move(result.second);solvedRevision_=revision_;
        std::cout<<"Fit pairs:";for(const auto& p:solved_.pairs) std::cout<<' '<<p.pair.number;
        std::cout<<"; RMS="<<solved_.rms<<" epi P95="<<solved_.epiP95<<" T_mm="<<solved_.T.t()
                 <<"; free T_mm="<<solved_.translationConstraint.initialT.t()<<" free RMS="<<solved_.translationConstraint.initialRms<<'\n';
    }
    void save() {
        persist();
        if(saving_.valid()) {status_="结果仍在保存，请稍候";return;}
        if(solved_.R.empty() || solvedRevision_!=revision_) {status_="标注已保存；至少接受 3 组图片并完成求解后才能保存结果";return;}
        const auto snapshot=solved_;const auto options=reportOptions();const auto left=left_,right=right_;const auto rev=revision_;
        saving_=std::async(std::launch::async,[snapshot,options,left,right,rev] {Save(options,left,right,snapshot);return rev;});
        status_="后台保存有约束和无约束标定结果";
    }
public:
    Review(Settings settings,Camera left,Camera right):settings_(std::move(settings)),left_(std::move(left)),right_(std::move(right)),dataset_(ReadDataset(settings_.input)) {
        Require(!fs::exists(settings_.output),"Output already exists; choose a NEW --output directory");fs::create_directories(settings_.output);
        views_[0].area={12,64,660,650};views_[1].area={684,64,660,650};
        LoadAnnotations(settings_.input/"annotations.yaml",dataset_.records,left_.referenceSize,right_.referenceSize);
        Require(count(Record::State::Accepted)<=settings_.maxPairs,"已保存的选择超过 --max-pairs，请增大数量上限");
        const auto pending=std::find_if(dataset_.records.begin(),dataset_.records.end(),[](const Record& r){return r.state==Record::State::Pending;});
        load(pending==dataset_.records.end()?0:static_cast<std::size_t>(pending-dataset_.records.begin()));
        ++revision_;needFit_=true;
    }
    bool quitting() const {return quit_;}
    const std::vector<Record>& records() const {return dataset_.records;}
    const Viewport& viewport(int side) const {return views_[side];}
    int listOffset() const {return listOffset_;}
    const Solution& solution() const {return solved_;}
    std::size_t currentIndex() const {return index_;}
    void update() {
        if(saving_.valid() && saving_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {savedRevision_=saving_.get();status_=savedRevision_==revision_?"已保存两种标定结果及参考图":"已保存上一版结果；当前选择已改变";}
            catch(const std::exception& e){status_=std::string("结果保存失败：")+e.what();}
        }
        if(fitting_.valid() && fitting_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {finishFit();}catch(const std::exception& e){status_=std::string("求解失败：")+e.what();std::cerr<<status_<<'\n';}
        }
        startFit();
        if(detection_.valid() && detection_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {
                auto result=detection_.get();
                if(result.index==index_ && result.epoch==epoch_) {
                    detected_=std::move(result);auto& r=current();r.sample.pair.left=detected_.left;r.sample.pair.right=detected_.right;r.sample.pair.rightReversed=detected_.reversed;
                    status_=detected_.leftFound && detected_.rightFound?"两侧均已找到棋盘：核对角点 0/87 是否对应；空格接受，F 反转右侧顺序":"一侧 ROI 内未找到完整棋盘；请框全棋盘并保留白边，或按 N 跳过";persist();
                }
            }catch(const std::exception& e){status_=std::string("ROI 检测失败：")+e.what();}
        }
        if(needDetection_ && !detection_.valid()) {
            needDetection_=false;const auto left=rawLeft_,right=rawRight_;const auto lr=current().leftRoi,rr=current().rightRoi;const auto index=index_;const auto epoch=epoch_;
            detection_=std::async(std::launch::async,[left,right,lr,rr,index,epoch] {return DetectPair(left,right,lr,rr,index,epoch);});
        }
    }
    void key(int key) {
        auto& r=current();
        if(key==27 || key=='q') {quit_=true;return;}
        if(key=='b') {if(index_>0) load(index_-1);return;}
        if(key=='p') {advance();return;}
        if(key=='0') {views_[0].reset(rawLeft_.size());views_[1].reset(rawRight_.size());return;}
        if(key=='s') {save();return;}
        if(key=='[' || key==']') {listOffset_=std::clamp(listOffset_+(key=='['?-3:3),0,std::max(0,count(Record::State::Accepted)-18));return;}
        if(key=='1' || key=='2' || key=='3' || key=='4' || key=='+' || key=='=' || key=='-') {
            if(key>='1' && key<='4') activeSide_=key<='2'?0:1;
            auto& view=views_[activeSide_];view.magnify(view.project(view.center),(key=='2' || key=='4' || key=='-')?1./1.3:1.3);return;
        }
        if(key=='u') {
            while(!undo_.empty()) {const auto i=undo_.back();undo_.pop_back();if(dataset_.records[i].state==Record::State::Accepted) {dataset_.records[i].state=Record::State::Pending;changed();load(i);break;}}
            return;
        }
        if(key=='n') {++epoch_;needDetection_=false;const bool included=r.state==Record::State::Accepted;r.state=Record::State::Skipped;if(included) changed();else persist();advance();return;}
        if(key=='d') {if(r.state==Record::State::Accepted) {r.state=Record::State::Pending;changed();status_="已移出当前图片组，保留标注";}return;}
        if(key=='r') {r.leftRoi={};r.rightRoi={};invalidateCurrent();needDetection_=false;status_="两侧 ROI 已清除，请重新框选";return;}
        if(key=='f' && detected_.rightFound && !detection_.valid() && !needDetection_) {
            std::reverse(detected_.right.begin(),detected_.right.end());detected_.reversed=!detected_.reversed;
            r.sample.pair.right=detected_.right;r.sample.pair.rightReversed=detected_.reversed;
            if(r.state==Record::State::Accepted) changed();else persist();status_="已反转右侧角点顺序，请核对同编号是否为同一个实际角点";return;
        }
        if(key==' ') {
            if(detection_.valid() || needDetection_ || !detected_.leftFound || !detected_.rightFound || r.leftRoi.empty() || r.rightRoi.empty()) {status_="两侧 ROI 均须检测到完整的 88 个角点才能接受";return;}
            if(r.state!=Record::State::Accepted) {
                if(count(Record::State::Accepted)>=settings_.maxPairs) {status_="已达图片组数上限，请先移除已纳入的一组";return;}
                r.state=Record::State::Accepted;undo_.push_back(index_);changed();
            }
            advance();
        }
    }
    void mouse(int event,int x,int y,int flags) {
        const auto point=pointer_.point(event,x,y);const cv::Point pixel(cvRound(point.x),cvRound(point.y));
        int side=-1;for(int i=0;i<2;++i) if(views_[i].area.contains(pixel) && !views_[i].native.empty()) side=i;
        if(side>=0) activeSide_=side;
        if(calibration_ui::PointerInput::isWheel(event)) {
            const int direction=pointer_.direction(flags);if(direction==0) return;
            if(side>=0) views_[side].magnify(point,direction>0?1.3:1./1.3);
            else if(point.x>=1360) key(direction>0?'[':']');
            return;
        }
        if(event==cv::EVENT_LBUTTONDOWN) {
            for(const auto& b:Buttons()) if(b.box.contains({x,y})) {key(b.key);return;}
            if(x>=1360 && y>=438 && y<924) {
                const int row=(y-438)/27+std::min(listOffset_,std::max(0,count(Record::State::Accepted)-18));int n=0;
                for(std::size_t i=0;i<dataset_.records.size();++i) if(dataset_.records[i].state==Record::State::Accepted) {if(n++==row) {load(i);return;}}
            }
            if(side>=0) {const auto p=views_[side].unproject(point);if(p.x>=0 && p.y>=0 && p.x<views_[side].native.width && p.y<views_[side].native.height) {drag_=side;start_=cursor_=point;}}
        }
        if(event==cv::EVENT_RBUTTONDOWN && side>=0) {pan_=side;rightMoved_=false;rightStart_=lastPan_=point;}
        if(event==cv::EVENT_MOUSEMOVE) {
            if(drag_>=0) cursor_=point;
            if(pan_>=0) {rightMoved_=rightMoved_ || cv::norm(point-rightStart_)>4.;if(rightMoved_) {auto& view=views_[pan_];view.center-=(point-lastPan_)*(1./view.scale());view.clamp();lastPan_=point;}}
        }
        if(event==cv::EVENT_RBUTTONUP && pan_>=0) {
            if(!rightMoved_ && side==pan_) views_[pan_].magnify(point,(flags&cv::EVENT_FLAG_SHIFTKEY)?1./1.3:1.3);
            pan_=-1;
        }
        if(event==cv::EVENT_LBUTTONUP && drag_>=0) {
            const int chosen=drag_;drag_=-1;const auto roi=views_[chosen].roi(start_,point);
            if(roi.width<24 || roi.height<24) {status_="ROI 太小，请放大后框选完整棋盘";return;}
            if(chosen==0) current().leftRoi=roi;else current().rightRoi=roi;
            invalidateCurrent();status_="正在后台检测所选 ROI 内的原始分辨率角点";
        }
    }
    static void Mouse(int event,int x,int y,int flags,void* self) {
        try {static_cast<Review*>(self)->mouse(event,x,y,flags);}catch(const std::exception& e){static_cast<Review*>(self)->status_=e.what();}
    }
    cv::Mat render() const {
        cv::Mat canvas(1000,1800,CV_8UC3,cv::Scalar(25,25,25));const auto& r=current();
        Label(canvas,"图片组 "+r.key+"  ["+std::to_string(index_+1)+" / "+std::to_string(dataset_.records.size())+"]",15,28,kGreen,.65);
        Label(canvas,std::string("状态：")+(r.state==Record::State::Accepted?"已接受":r.state==Record::State::Skipped?"已跳过":"待处理"),360,28,kYellow);
        Label(canvas,"左相机 "+std::to_string(left_.referenceSize.width)+"x"+std::to_string(left_.referenceSize.height),15,52);
        Label(canvas,"右相机 "+std::to_string(right_.referenceSize.width)+"x"+std::to_string(right_.referenceSize.height),685,52);
        DrawView(canvas,rawLeft_,views_[0],r.leftRoi,detected_.left);DrawView(canvas,rawRight_,views_[1],r.rightRoi,detected_.right);
        if(drag_>=0) {
            const auto box=views_[drag_].roi(start_,cursor_);const auto& v=views_[drag_];auto pane=canvas(v.area);const cv::Point2d offset(v.area.x,v.area.y);
            cv::rectangle(pane,v.project(box.tl())-offset,v.project(box.br())-offset,kGreen,2);
        }
        Label(canvas,status_.substr(0,145),15,770,kYellow);
        Label(canvas,std::string("ROI 检测：")+(detection_.valid()?"处理中":"就绪")+"；上次 "+Number(detected_.milliseconds,0)+" ms；左/右："+
            (detected_.leftFound?"已找到":"未找到")+" / "+(detected_.rightFound?"已找到":"未找到"),15,798);
        Label(canvas,"左键框选 ROI；滚轮缩放；右键单击放大、拖动平移；Shift+右键缩小；接受前核对角点 0 / 87",15,825);
        for(const auto& b:Buttons()) {cv::rectangle(canvas,b.box,{65,65,65},cv::FILLED);Label(canvas,b.caption,b.box.x+8,b.box.y+26,kWhite,.47);}
        Label(canvas,"标注自动保存到 Input/annotations.yaml；BMP 原图和角点均保留原始坐标",15,972,kGreen,.5);
        cv::line(canvas,{1355,0},{1355,999},{75,75,75});const int x=1370;
        Label(canvas,"当前标定数据",x,30,kGreen,.6);
        Label(canvas,"已纳入 "+std::to_string(count(Record::State::Accepted))+" / "+std::to_string(settings_.maxPairs)+"；已跳过 "+std::to_string(count(Record::State::Skipped)),x,59);
        Label(canvas,"数据版本 "+std::to_string(revision_)+(fitting_.valid() || needFit_?"：计算中":"：就绪"),x,87,kYellow);
        if(!solved_.R.empty()) {
            Label(canvas,"RMS: "+Number(solved_.rms)+" px",x,117);
            Label(canvas,"极线误差 P95："+Number(solved_.epiP95)+" px",x,145);
            Label(canvas,"最大极线误差："+Number(solved_.epiMax)+" px",x,173);
            Label(canvas,"基线长度："+Number(cv::norm(solved_.T),2)+" mm",x,201);
            Label(canvas,"T mm: "+Number(solved_.T.at<double>(0),1)+", "+Number(solved_.T.at<double>(1),1)+", "+Number(solved_.T.at<double>(2),1),x,229);
            for(int row=0;row<3;++row) Label(canvas,(row==0?"R: ":"   ")+Number(solved_.R.at<double>(row,0),5)+" "+Number(solved_.R.at<double>(row,1),5)+" "+Number(solved_.R.at<double>(row,2),5),x,257+row*27,kWhite,.46);
            Label(canvas,"无约束 RMS："+Number(solved_.translationConstraint.initialRms)+" px",x,339);
            Label(canvas,solved_.translationConstraint.atLimit?"T：接近约束边界":"T：位于约束范围内",x,366,solved_.translationConstraint.atLimit?kYellow:kGreen);
            Label(canvas,solved_.passed?"质量检查通过（测距尚未验证）":"质量检查未通过：数量、误差或收敛状态",x,393,solved_.passed?kGreen:kYellow,.43);
        } else Label(canvas,count(Record::State::Accepted)<3?"接受至少 3 组图片后显示 RMS":"等待当前数据的求解结果",x,120,kYellow);
        Label(canvas,"已纳入组号 | 左/右 RMS | 极线 P95",x,432,kGreen,.46);
        const int selected=count(Record::State::Accepted);const int offset=std::min(listOffset_,std::max(0,selected-18));
        int n=0,shown=0;
        for(const auto& item:dataset_.records) if(item.state==Record::State::Accepted) {
            if(n++<offset) continue;
            if(shown>=18) break;
            std::string line="#"+item.key;
            const auto it=std::find_if(solved_.pairs.begin(),solved_.pairs.end(),[&](const LivePair& p){return p.pair.number==item.sample.pair.number;});
            if(it!=solved_.pairs.end()) {const int i=static_cast<int>(it-solved_.pairs.begin());line+=" | "+Number(solved_.errors.at<double>(i,0),2)+"/"+Number(solved_.errors.at<double>(i,1),2)+" | "+Number(solved_.perPair[i].p95,2);}
            else line+=" | 等待求解";
            Label(canvas,line.substr(0,49),x,459+27*shown,item.key==r.key?kYellow:kWhite,.47);++shown;
        }
        Label(canvas,"点击列表检查图片；滚轮或按钮翻动列表",x,978,kWhite,.44);return canvas;
    }
    int finish() {
        persist();if(detection_.valid()) {try{detection_.get();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}}
        for(;;) {if(fitting_.valid()) finishFit();startFit();if(!fitting_.valid()) break;}
        if(saving_.valid()) savedRevision_=saving_.get();
        if(!solved_.R.empty() && solvedRevision_==revision_ && savedRevision_!=revision_) Save(reportOptions(),left_,right_,solved_);
        std::cout<<"Annotations: "<<settings_.input/"annotations.yaml"<<"; reports: "<<settings_.output<<'\n';
        return solved_.passed?0:2;
    }
};
int Calibrate(const Settings& settings) {
    const DartCongfig config(settings.config.string());const auto left=Intrinsics(config.leftCamera,settings.leftIntrinsics),right=Intrinsics(config.rightCamera,settings.rightIntrinsics);
    cv::setNumThreads(4);CameraStopSignals signals;Review app(settings,left,right);
    std::cout<<"Offline ROI calibration. No cameras are opened. Input="<<settings.input<<" output="<<settings.output<<'\n';
    cv::namedWindow(kReviewWindow,cv::WINDOW_NORMAL);cv::resizeWindow(kReviewWindow,1620,900);cv::setMouseCallback(kReviewWindow,Review::Mouse,&app);
    try {
        while(!signals.stopRequested() && !app.quitting()) {
            app.update();cv::imshow(kReviewWindow,app.render());const int key=cv::waitKey(10)&0xff;
            if(cv::getWindowProperty(kReviewWindow,cv::WND_PROP_VISIBLE)==0) break;
            if(key!=255) app.key(key);
        }
        cv::setMouseCallback(kReviewWindow,nullptr);cv::destroyWindow(kReviewWindow);return app.finish();
    }catch(...) {cv::setMouseCallback(kReviewWindow,nullptr);cv::destroyWindow(kReviewWindow);throw;}
}
} // namespace stereo_workflow
#ifndef DART_STEREO_WORKFLOW_NO_MAIN
int main(int argc,char** argv) {
    try {const auto s=stereo_workflow::ParseSettings(argc,argv);if(s.help) return 0;
        return s.mode==stereo_workflow::Settings::Mode::Collect?stereo_workflow::Collect(s):stereo_workflow::Calibrate(s);}
    catch(const std::exception& e){std::cerr<<"Stereo calibration failed: "<<e.what()<<'\n';return 1;}
}
#endif
