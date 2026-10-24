#pragma once
#include <cctype>
#include <map>
#include <limits>

// Included after the monocular solver; capture never runs a chessboard detector.
namespace livecalib {
inline void ensure(bool ok,const std::string& message) {if(!ok) throw std::runtime_error(message);}
constexpr const char* collectWindow="单目标定 - 采集 BMP";
constexpr const char* reviewWindow="单目标定 - 离线 ROI";
inline std::string fileStamp(const fs::path& p) {
    return std::to_string(fs::file_size(p))+":"+std::to_string(fs::last_write_time(p).time_since_epoch().count());
}
inline bool validPoints(const std::vector<cv::Point2f>& points,cv::Size size,cv::Size board) {
    return points.size()==static_cast<std::size_t>(board.area()) && std::all_of(points.begin(),points.end(),[&](const cv::Point2f& p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && p.x>=0 && p.y>=0 && p.x<size.width && p.y<size.height;
    });
}
struct MonoRecord {
    enum class State { Pending,Accepted,Skipped };
    fs::path path;std::string key;Sample sample;cv::Rect roi;State state=State::Pending;
};
struct MonoDataset {std::vector<MonoRecord> records;cv::Size size;};
inline MonoDataset readDataset(const Options& o) {
    ensure(fs::is_directory(o.input),"Input directory does not exist: "+o.input.string());
    MonoDataset result;
    for(const auto& entry:fs::directory_iterator(o.input)) {
        auto ext=entry.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return std::tolower(c);});
        if(!entry.is_regular_file() || ext!=".bmp" || entry.path().stem().string().find(".pending")!=std::string::npos) continue;
        MonoRecord r;r.path=entry.path();r.key=entry.path().stem().string();r.sample.image=entry.path().filename().string();result.records.push_back(std::move(r));
    }
    std::sort(result.records.begin(),result.records.end(),[](const MonoRecord& a,const MonoRecord& b){return a.path.filename()<b.path.filename();});
    ensure(!result.records.empty(),"No BMP images in "+o.input.string());
    for(const auto& r:result.records) {
        const auto image=cv::imread(r.path.string(),cv::IMREAD_COLOR|cv::IMREAD_IGNORE_ORIENTATION);
        if(!image.empty()) {result.size=image.size();break;}
    }
    ensure(!result.size.empty(),"No decodable BMP image in Input");
    const auto manifest=o.input/"dataset.yaml";
    if(fs::exists(manifest)) {
        cv::FileStorage f(manifest.string(),cv::FileStorage::READ);ensure(f.isOpened(),"Cannot read collection manifest");
        ensure(static_cast<std::string>(f["camera_side"])==o.side,"Collection camera side differs from --side");
        ensure(cv::Size(static_cast<int>(f["image_width"]),static_cast<int>(f["image_height"]))==result.size,"Manifest/native image dimensions differ");
        std::map<std::string,std::string> ids;
        for(const auto& n:f["frames"]) ids[static_cast<std::string>(n["file"])]=static_cast<std::string>(n["frame_id"]);
        for(auto& r:result.records) {const auto it=ids.find(r.sample.image);if(it!=ids.end()) r.sample.frameId=std::stoull(it->second);}
    }
    return result;
}
inline std::vector<Sample> selected(const std::vector<MonoRecord>& records) {
    std::vector<Sample> samples;for(const auto& r:records) if(r.state==MonoRecord::State::Accepted) samples.push_back(r.sample);return samples;
}
inline void saveAnnotations(const Options& o,const MonoDataset& data) {
    const auto pending=o.input/"annotations.pending.yaml";
    cv::FileStorage f(pending.string(),cv::FileStorage::WRITE);ensure(f.isOpened(),"Cannot save annotations");
    f<<"schema_version"<<1<<"camera_side"<<o.side<<"columns"<<o.board.width<<"rows"<<o.board.height<<"square_mm"<<o.squareMm
     <<"image_width"<<data.size.width<<"image_height"<<data.size.height<<"records"<<"[";
    for(const auto& r:data.records) f<<"{"<<"file"<<r.sample.image<<"stamp"<<fileStamp(r.path)<<"state"<<static_cast<int>(r.state)<<"roi"<<r.roi<<"points"<<r.sample.points<<"}";
    f<<"]";f.release();fs::rename(pending,o.input/"annotations.yaml");
}
inline void loadAnnotations(const Options& o,MonoDataset& data) {
    if(!fs::exists(o.input/"annotations.yaml")) return;
    cv::FileStorage f((o.input/"annotations.yaml").string(),cv::FileStorage::READ);ensure(f.isOpened(),"Cannot read annotations");
    ensure(static_cast<int>(f["schema_version"])==1 && static_cast<std::string>(f["camera_side"])==o.side &&
        static_cast<int>(f["columns"])==o.board.width && static_cast<int>(f["rows"])==o.board.height &&
        std::abs(static_cast<double>(f["square_mm"])-o.squareMm)<1e-6 &&
        cv::Size(static_cast<int>(f["image_width"]),static_cast<int>(f["image_height"]))==data.size,"Annotation camera/board/dimensions mismatch");
    std::map<std::string,MonoRecord*> files;for(auto& r:data.records) files[r.sample.image]=&r;
    for(const auto& n:f["records"]) {
        const auto it=files.find(static_cast<std::string>(n["file"]));if(it==files.end()) continue;
        auto& r=*it->second;if(static_cast<std::string>(n["stamp"])!=fileStamp(r.path)) {std::cerr<<"Ignoring stale annotation: "<<r.path<<'\n';continue;}
        cv::Rect roi;std::vector<cv::Point2f> points;n["roi"]>>roi;n["points"]>>points;
        const int state=static_cast<int>(n["state"]);
        if(state<0 || state>2 || (roi&cv::Rect({},data.size))!=roi || (!points.empty() && !validPoints(points,data.size,o.board))) continue;
        if(state==1 && (roi.empty() || !validPoints(points,data.size,o.board))) continue;
        r.roi=roi;r.sample.points=std::move(points);r.state=static_cast<MonoRecord::State>(state);
    }
}

class MonoWriter {
    Options options_;cv::Size size_;std::vector<CameraFrame> metadata_;
    void manifest() const {
        cv::FileStorage f((options_.input/"dataset.pending.yaml").string(),cv::FileStorage::WRITE);ensure(f.isOpened(),"Cannot write collection manifest");
        f<<"schema_version"<<1<<"camera_side"<<options_.side<<"image_width"<<size_.width<<"image_height"<<size_.height
         <<"config_source"<<fs::absolute(options_.config).string()<<"timestamp_convention"<<"host_receive_ms"<<"frames"<<"[";
        for(std::size_t i=0;i<metadata_.size();++i) f<<"{"<<"file"<<cv::format("%06d.bmp",static_cast<int>(i+1))
            <<"frame_id"<<std::to_string(metadata_[i].frame_id)<<"timestamp_ms"<<std::to_string(metadata_[i].timestamp_ms)<<"}";
        f<<"]";f.release();fs::rename(options_.input/"dataset.pending.yaml",options_.input/"dataset.yaml");
    }
public:
    explicit MonoWriter(Options o):options_(std::move(o)) {
        ensure(!fs::exists(options_.input) || (fs::is_directory(options_.input) && fs::is_empty(options_.input)),"Collection requires a NEW or empty --input directory; existing images are never overwritten");
        fs::create_directories(options_.input);
    }
    std::size_t count() const {return metadata_.size();}
    std::size_t append(const CameraFrame& frame) {
        ensure(!frame.image.empty(),"Cannot save an empty camera image");
        if(size_.empty()) size_=frame.image.size();
        ensure(frame.image.size()==size_,"Camera frame dimensions changed during collection");
        const auto stem=cv::format("%06d",static_cast<int>(metadata_.size()+1));
        const auto temporary=options_.input/(stem+".pending.bmp"),target=options_.input/(stem+".bmp");
        ensure(!fs::exists(target),"Refusing to overwrite "+target.string());
        ensure(cv::imwrite(temporary.string(),frame.image),"Cannot save BMP");fs::rename(temporary,target);
        CameraFrame meta=frame;meta.image.release();metadata_.push_back(std::move(meta));
        try {manifest();}catch(...) {metadata_.pop_back();fs::remove(target);throw;}
        return count();
    }
};

inline int collect(const Options& o) {
    cv::setNumThreads(4);MonoWriter writer(o);
    const DartCongfig config(o.config.string());const auto& params=o.side=="left"?config.leftCamera:config.rightCamera;
    CameraStopSignals signals;LatestCamera camera(params,o.side=="left"?CameraSide::Left:CameraSide::Right);
    std::future<std::size_t> saving;std::optional<CameraFrame> latest;
    std::size_t count=0;bool automatic=o.automatic,request=false;
    auto received=Clock::now(),lastSaved=received-std::chrono::hours(1);std::string status="等待相机图像";
    std::cout<<"Collect "<<o.side<<" native BMPs into "<<o.input<<"; no chessboard detection or video encoding.\n";
    cv::namedWindow(collectWindow,cv::WINDOW_NORMAL);cv::resizeWindow(collectWindow,1280,850);
    try {
        while(!signals.stopRequested()) {
            if(saving.valid() && saving.wait_for(std::chrono::seconds(0))==std::future_status::ready) {count=saving.get();status="已保存图片 "+std::to_string(count);}
            if(auto frame=camera.take()) {
                latest=std::move(frame);received=Clock::now();
                if(!saving.valid() && (request || (automatic && std::chrono::duration<double>(received-lastSaved).count()>=o.interval))) {
                    const auto snapshot=*latest;lastSaved=received;request=false;
                    saving=std::async(std::launch::async,[&writer,snapshot]{return writer.append(snapshot);});status="后台保存原始 BMP";
                }
            } else if(Clock::now()-received>std::chrono::seconds(10)) throw std::runtime_error("No camera frames for 10 s; check SDK/network/exposure");
            cv::Mat canvas(850,1280,CV_8UC3,cv::Scalar(25,25,25));
            if(latest) {
                cv::Mat preview;const double scale=std::min(1000./latest->image.cols,680./latest->image.rows);
                cv::resize(latest->image,preview,{},scale,scale,cv::INTER_AREA);preview.copyTo(canvas(cv::Rect(10,40,preview.cols,preview.rows)));
                text(canvas,"实时画面 "+calibration_ui::sideName(o.side)+" #"+std::to_string(latest->frame_id),10,25,{80,255,80});
                text(canvas,std::to_string(latest->image.cols)+" x "+std::to_string(latest->image.rows),1020,170);
            }
            text(canvas,"已保存："+std::to_string(count),1020,60);text(canvas,automatic?"自动保存":"已暂停",1020,95);
            text(canvas,cv::format("间隔：%.1f 秒",o.interval),1020,130);
            text(canvas,status.substr(0,140),10,755);text(canvas,"A：自动/暂停 | 空格：保存下一帧 | Q/ESC：退出",10,805);
            cv::imshow(collectWindow,canvas);const int key=cv::waitKey(5)&0xff;
            if(key==27 || key=='q' || cv::getWindowProperty(collectWindow,cv::WND_PROP_VISIBLE)==0) break;
            if(key=='a') automatic=!automatic;
            if(key==' ') request=true;
        }
        if(saving.valid()) count=saving.get();
        cv::destroyWindow(collectWindow);
        std::cout<<"Saved "<<count<<" images. Next: ./build/demarcate --calibration --side="<<o.side<<" --input="<<o.input<<'\n';return 0;
    }catch(...) {cv::destroyWindow(collectWindow);throw;}
}

// The display transform never changes the native coordinates used by the solver.
struct MonoViewport {
    cv::Rect area{10,50,1080,560};cv::Size native;double zoom=1.;cv::Point2d center;
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
        const auto p=unproject(cursor);zoom=std::clamp(zoom*factor,1.,32.);
        center=p-(cursor-cv::Point2d(area.x+area.width/2.,area.y+area.height/2.))*(1./scale());clamp();
    }
    cv::Rect roi(cv::Point2d a,cv::Point2d b) const {
        a=unproject(a);b=unproject(b);
        return cv::Rect(cv::Point(static_cast<int>(std::floor(std::min(a.x,b.x))),static_cast<int>(std::floor(std::min(a.y,b.y)))),
            cv::Point(static_cast<int>(std::ceil(std::max(a.x,b.x))),static_cast<int>(std::ceil(std::max(a.y,b.y)))))&cv::Rect({},native);
    }
};
inline bool detectRoi(const cv::Mat& image,cv::Rect roi,cv::Size board,std::vector<cv::Point2f>& points) {
    points.clear();if(image.empty() || roi.width<24 || roi.height<24 || (roi&cv::Rect({},image.size()))!=roi) return false;
    cv::Mat gray;cv::cvtColor(image(roi),gray,cv::COLOR_BGR2GRAY);
    bool found=cv::findChessboardCornersSB(gray,board,points,cv::CALIB_CB_NORMALIZE_IMAGE);
    if(!found) {points.clear();found=cv::findChessboardCorners(gray,board,points,cv::CALIB_CB_ADAPTIVE_THRESH|cv::CALIB_CB_NORMALIZE_IMAGE|cv::CALIB_CB_FAST_CHECK);}
    if(!found || points.size()!=static_cast<std::size_t>(board.area())) {points.clear();return false;}
    double spacing=1e9;
    for(int y=0;y<board.height;++y) for(int x=0;x<board.width;++x) {
        const int i=y*board.width+x;
        if(x+1<board.width) spacing=std::min(spacing,cv::norm(points[i+1]-points[i]));
        if(y+1<board.height) spacing=std::min(spacing,cv::norm(points[i+board.width]-points[i]));
    }
    const int half=std::clamp(static_cast<int>(std::floor(spacing*.35)),1,9);
    cv::cornerSubPix(gray,points,{half,half},{-1,-1},{cv::TermCriteria::COUNT|cv::TermCriteria::EPS,50,.001});
    for(auto& p:points) p+=cv::Point2f(static_cast<float>(roi.x),static_cast<float>(roi.y));
    if(!validPoints(points,image.size(),board)) {points.clear();return false;}return true;
}
struct MonoDetection {std::size_t index=0;std::uint64_t epoch=0;std::vector<cv::Point2f> points;double ms=0;bool found=false;};
struct MonoButton {cv::Rect box;const char* caption;int key;};
inline const std::vector<MonoButton>& monoButtons() {
    static const std::vector<MonoButton> buttons{
        {{435,700,200,42},"接受并下一张 [空格]",' '},{{650,700,200,42},"跳过并下一张 [N]",'n'},
        {{865,700,220,42},"保存结果 [S]",'s'},{{435,760,200,42},"上一张 [B]",'b'},
        {{650,760,200,42},"下一张 [P]",'p'},{{865,760,220,42},"移出标定 [D]",'d'},
        {{435,820,200,42},"清除 ROI [R]",'r'},{{650,820,200,42},"撤销接受 [U]",'u'},
        {{865,820,220,42},"重置缩放 [0]",'0'},
        {{860,8,105,32},"放大 [+]",'+'},{{975,8,105,32},"缩小 [-]",'-'},
        {{1115,925,210,38},"列表上滚 [",'['},{{1340,925,210,38},"列表下滚 ]",']'}};
    return buttons;
}
class MonoReview {
    Options o_;MonoDataset data_;std::size_t index_=0;cv::Mat raw_;MonoViewport view_;
    MonoDetection detected_;std::future<MonoDetection> detection_;bool needDetection_=false;
    std::future<std::pair<std::uint64_t,Solution>> fitting_;std::future<std::uint64_t> saving_;
    Solution solved_;bool needFit_=true,quit_=false;
    std::uint64_t epoch_=0,revision_=1,solvedRevision_=0,savedRevision_=0;
    std::vector<std::size_t> undo_;int listOffset_=0;bool drag_=false,pan_=false,rightMoved_=false;cv::Point2d start_,cursor_,lastPan_,rightStart_;
    calibration_ui::PointerInput pointer_;
    std::string status_="框选完整标定板，并保留白色边缘";
    MonoRecord& current() {return data_.records[index_];}
    const MonoRecord& current() const {return data_.records[index_];}
    void persist() const {saveAnnotations(o_,data_);}
    void changed() {++revision_;solved_={};needFit_=true;persist();}
    void load(std::size_t index) {
        index_=std::min(index,data_.records.size()-1);++epoch_;detected_={};needDetection_=false;drag_=pan_=false;
        raw_=cv::imread(current().path.string(),cv::IMREAD_COLOR|cv::IMREAD_IGNORE_ORIENTATION);
        if(raw_.empty() || raw_.size()!=data_.size) {raw_.release();view_.reset({});status_="图片无法读取或原始尺寸不一致，按 N 跳过";return;}
        view_.reset(raw_.size());detected_.points=current().sample.points;detected_.found=validPoints(detected_.points,data_.size,o_.board);
        needDetection_=current().state!=MonoRecord::State::Accepted && !current().roi.empty() && !detected_.found;
        status_=current().state==MonoRecord::State::Accepted?"已纳入标定；重新框选后需再次接受":"框选 ROI 并检查角点；空格接受，N 跳过";
    }
    void advance() {if(index_+1<data_.records.size()) load(index_+1);else status_="已到最后一张；可检查右侧列表或按 S 保存";}
    void startFit() {
        if(!needFit_ || fitting_.valid()) return;
        needFit_=false;const auto samples=selected(data_.records);const auto rev=revision_;
        if(samples.size()<3) {solved_={};solvedRevision_=rev;return;}
        const auto options=o_;const auto size=data_.size;
        fitting_=std::async(std::launch::async,[options,size,samples,rev]{return std::make_pair(rev,solve(options,size,samples,modelFlags(options.model)));});
    }
    void finishFit() {
        auto completed=fitting_.get();if(completed.first!=revision_) return;
        solved_=std::move(completed.second);solvedRevision_=revision_;
        std::cout<<"Monocular fit: "<<solved_.samples.size()<<" views, RMS="<<solved_.data->totalAvgErr<<" px\n";
    }
    void save() {
        persist();
        if(saving_.valid()) {status_="结果仍在保存，请稍候";return;}
        if(solvedRevision_!=revision_ || !solved_.data || solved_.data->cameraMatrix.empty()) {status_="标注已保存；至少接受 3 张图片并完成求解后才能保存内参";return;}
        const auto solution=solved_;const auto options=o_;const auto revision=revision_;const auto screen=render();
        saving_=std::async(std::launch::async,[solution,options,revision,screen] {
            saveReport(options,solution);ensure(cv::imwrite((options.output/"review.png").string(),screen),"Cannot save review screenshot");return revision;
        });status_="后台保存内参结果及界面截图";
    }
public:
    explicit MonoReview(Options o):o_(std::move(o)),data_(readDataset(o_)) {
        loadAnnotations(o_,data_);ensure(selected(data_.records).size()<=static_cast<std::size_t>(o_.maxViews),"Saved selection exceeds --max-views; increase the limit");
        ensure(!fs::exists(o_.output),"Output already exists; choose a new --output directory");fs::create_directories(o_.output);
        const auto it=std::find_if(data_.records.begin(),data_.records.end(),[](const MonoRecord& r){return r.state==MonoRecord::State::Pending;});
        load(it==data_.records.end()?0:static_cast<std::size_t>(it-data_.records.begin()));
    }
    bool quitting() const {return quit_;}
    const MonoDataset& dataset() const {return data_;}
    const Solution& solution() const {return solved_;}
    std::size_t currentIndex() const {return index_;}
    const MonoViewport& viewport() const {return view_;}
    int listOffset() const {return listOffset_;}
    bool detectorBusy() const {return detection_.valid() || needDetection_;}
    void update() {
        if(saving_.valid() && saving_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {savedRevision_=saving_.get();status_=savedRevision_==revision_?"已保存内参结果及界面截图":"已保存上一版结果；当前选择已改变";}
            catch(const std::exception& e) {status_=std::string("保存失败：")+e.what();}
        }
        if(fitting_.valid() && fitting_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {finishFit();}catch(const std::exception& e){status_=std::string("求解失败：")+e.what();std::cerr<<status_<<'\n';}
        }
        startFit();
        if(detection_.valid() && detection_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {
                auto result=detection_.get();
                if(result.index==index_ && result.epoch==epoch_) {
                    detected_=std::move(result);current().sample.points=detected_.points;persist();
                    status_=detected_.found?"已找到完整棋盘；检查角点位置后按空格接受":"ROI 内未找到完整棋盘；请框全角点并保留白边，或按 N 跳过";
                }
            }catch(const std::exception& e){status_=std::string("ROI 检测失败：")+e.what();}
        }
        if(needDetection_ && !detection_.valid()) {
            needDetection_=false;const auto raw=raw_;const auto roi=current().roi;const auto board=o_.board;const auto index=index_;const auto epoch=epoch_;
            detection_=std::async(std::launch::async,[raw,roi,board,index,epoch] {
                const auto started=Clock::now();MonoDetection d;d.index=index;d.epoch=epoch;d.found=detectRoi(raw,roi,board,d.points);
                d.ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();return d;
            });
        }
    }
    void key(int key) {
        auto& r=current();
        if(key==27 || key=='q') {quit_=true;return;}
        if(key=='b') {if(index_>0) load(index_-1);return;}
        if(key=='p') {advance();return;}
        if(key=='0') {view_.reset(raw_.size());return;}
        if(key=='s') {save();return;}
        if(key=='+' || key=='=' || key=='-') {view_.magnify(view_.project(view_.center),key=='-'?1./1.3:1.3);return;}
        if(key=='[' || key==']') {listOffset_=std::clamp(listOffset_+(key=='['?-3:3),0,std::max(0,static_cast<int>(selected(data_.records).size())-12));return;}
        if(key=='u') {
            while(!undo_.empty()) {const auto i=undo_.back();undo_.pop_back();if(data_.records[i].state==MonoRecord::State::Accepted) {data_.records[i].state=MonoRecord::State::Pending;changed();load(i);break;}}return;
        }
        if(key=='n') {++epoch_;needDetection_=false;const bool included=r.state==MonoRecord::State::Accepted;r.state=MonoRecord::State::Skipped;if(included) changed();else persist();advance();return;}
        if(key=='d') {if(r.state==MonoRecord::State::Accepted) {r.state=MonoRecord::State::Pending;changed();status_="已移出标定，保留 ROI 和角点";}return;}
        if(key=='r') {setRoi({});status_="已清除 ROI，请重新框选";return;}
        if(key==' ') {
            if(raw_.empty() || detectorBusy() || !detected_.found || r.roi.empty()) {status_="当前 ROI 必须检测到完整棋盘才能接受";return;}
            if(r.state!=MonoRecord::State::Accepted) {
                if(selected(data_.records).size()>=static_cast<std::size_t>(o_.maxViews)) {status_="已达图片数量上限，请移除一张或增大 --max-views";return;}
                r.state=MonoRecord::State::Accepted;undo_.push_back(index_);changed();
            }
            advance();
        }
    }
    void setRoi(cv::Rect roi) {
        ensure((roi&cv::Rect({},data_.size))==roi,"ROI outside native image");
        auto& r=current();const bool included=r.state==MonoRecord::State::Accepted;r.state=MonoRecord::State::Pending;r.roi=roi;r.sample.points.clear();
        detected_={};++epoch_;needDetection_=!raw_.empty() && !roi.empty();
        if(included) changed();else persist();
    }
    void mouse(int event,int x,int y,int flags) {
        const auto point=pointer_.point(event,x,y);const cv::Point pixel(cvRound(point.x),cvRound(point.y));
        const bool inside=view_.area.contains(pixel) && !raw_.empty();
        if(calibration_ui::PointerInput::isWheel(event)) {
            const int direction=pointer_.direction(flags);if(direction==0) return;
            if(inside) view_.magnify(point,direction>0?1.3:1./1.3);
            else if(point.x>=1100) key(direction>0?'[':']');
            return;
        }
        if(event==cv::EVENT_LBUTTONDOWN) {
            for(const auto& b:monoButtons()) if(b.box.contains({x,y})) {key(b.key);return;}
            if(x>=1100 && y>=535 && y<955) {
                const int offset=std::min(listOffset_,std::max(0,static_cast<int>(selected(data_.records).size())-12));
                const int row=(y-535)/30+offset;int n=0;
                for(std::size_t i=0;i<data_.records.size();++i) if(data_.records[i].state==MonoRecord::State::Accepted && n++==row) {load(i);return;}
            }
            const auto native=view_.unproject(point);
            if(inside && native.x>=0 && native.y>=0 && native.x<view_.native.width && native.y<view_.native.height) {drag_=true;start_=cursor_=point;}
        }
        if(event==cv::EVENT_RBUTTONDOWN && inside) {pan_=true;rightMoved_=false;rightStart_=lastPan_=point;}
        if(event==cv::EVENT_MOUSEMOVE) {
            if(drag_) cursor_=point;
            if(pan_) {
                rightMoved_=rightMoved_ || cv::norm(point-rightStart_)>4.;
                if(rightMoved_) {view_.center-=(point-lastPan_)*(1./view_.scale());view_.clamp();lastPan_=point;}
            }
        }
        if(event==cv::EVENT_RBUTTONUP && pan_) {
            if(!rightMoved_ && inside) view_.magnify(point,(flags&cv::EVENT_FLAG_SHIFTKEY)?1./1.3:1.3);
            pan_=false;
        }
        if(event==cv::EVENT_LBUTTONUP && drag_) {
            drag_=false;const auto roi=view_.roi(start_,point);
            if(roi.width<24 || roi.height<24) {status_="ROI 太小，请框选完整棋盘并保留白边";return;}
            setRoi(roi);status_="正在后台检测 ROI 内原始分辨率角点";
        }
    }
    static void Mouse(int e,int x,int y,int flags,void* self) {
        auto& app=*static_cast<MonoReview*>(self);try {app.mouse(e,x,y,flags);}catch(const std::exception& ex){app.status_=ex.what();}
    }
    cv::Mat render() const {
        cv::Mat canvas(1000,1600,CV_8UC3,cv::Scalar(25,25,25));const auto& r=current();
        text(canvas,calibration_ui::sideName(o_.side)+" 图片 "+r.key+" ["+std::to_string(index_+1)+" / "+std::to_string(data_.records.size())+"]",15,25,{80,255,80});
        text(canvas,std::string("状态：")+(r.state==MonoRecord::State::Accepted?"已接受":r.state==MonoRecord::State::Skipped?"已跳过":"待处理"),650,25);
        if(!raw_.empty()) {
            auto pane=canvas(view_.area);const double s=view_.scale();const cv::Mat a=(cv::Mat_<double>(2,3)<<s,0,view_.area.width/2.-s*view_.center.x,0,s,view_.area.height/2.-s*view_.center.y);
            cv::warpAffine(raw_,pane,a,pane.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(18,18,18));
            const cv::Point2d offset(view_.area.x,view_.area.y);
            const auto box=drag_?view_.roi(start_,cursor_):r.roi;
            if(!box.empty()) cv::rectangle(pane,view_.project(box.tl())-offset,view_.project(box.br())-offset,{70,215,255},1,cv::LINE_AA);
            if(detected_.found) {
                std::vector<cv::Point2f> points;for(const auto& p:detected_.points) points.emplace_back(view_.project(p)-offset);
                cv::drawChessboardCorners(pane,o_.board,points,true);
            }
        }
        text(canvas,status_.substr(0,140),15,635,{70,215,255});
        drawCoverage(canvas,data_.size,selected(data_.records));
        text(canvas,"绿色网格：已纳入图片的角点覆盖区域",10,900);
        text(canvas,"左键框选 ROI；滚轮缩放；右键单击放大、拖动平移",435,670);
        for(const auto& b:monoButtons()) {cv::rectangle(canvas,b.box,{65,65,65},cv::FILLED);text(canvas,b.caption,b.box.x+8,b.box.y+27);}
        text(canvas,std::string("ROI 检测：")+(detectorBusy()?"处理中":"就绪")+cv::format(" | 上次 %.0f ms",detected_.ms),435,900);
        text(canvas,"标注自动保存；Q/ESC 退出并保存当前有效标定结果",15,955);
        text(canvas,"按原始像素求解；请变化棋盘位置、大小及倾角；测距精度尚未验证",15,985);
        cv::line(canvas,{1100,0},{1100,999},{75,75,75});int y=30;
        auto line=[&](const std::string& t){text(canvas,t,1115,y);y+=28;};
        line("当前标定数据");line("相机："+calibration_ui::sideName(o_.side)+" | 模型："+calibration_ui::modelName(o_.model));
        line("原始尺寸："+std::to_string(data_.size.width)+" x "+std::to_string(data_.size.height));
        line("已纳入："+std::to_string(selected(data_.records).size())+" / "+std::to_string(o_.maxViews));
        line(needFit_ || fitting_.valid()?"求解器：计算中":"求解器：就绪");
        if(solved_.data && !solved_.data->cameraMatrix.empty()) {
            const auto& d=*solved_.data;double worst=0;cv::minMaxLoc(d.perViewErrors,nullptr,&worst);
            line(cv::format("RMS：%.4f px | 最差：%.4f",d.totalAvgErr,worst));
            line(cv::format("fx: %.2f | fy: %.2f",d.cameraMatrix.at<double>(0,0),d.cameraMatrix.at<double>(1,1)));
            line(cv::format("cx: %.2f | cy: %.2f",d.cameraMatrix.at<double>(0,2),d.cameraMatrix.at<double>(1,2)));
            line(cv::format("95%% 焦距区间：fx ±%.2f / fy ±%.2f",1.96*d.stdDeviations.at<double>(0),1.96*d.stdDeviations.at<double>(1)));
            line(std::string("图片数量：")+(solved_.enough?"通过":"需要更多数据"));line(std::string("RMS: ")+(solved_.rmsOK?"通过":"偏高"));
            line(std::string("置信度：")+(solved_.confidenceOK?"通过":"偏弱"));line(std::string("覆盖情况：")+(solved_.coverageOK?"通过":"需要更多数据"));
            line("求解标志："+std::to_string(solved_.flags));
        }else line("接受至少 3 张图片后显示 RMS");
        text(canvas,"已纳入图片 | 单张 RMS (px)",1115,525,{80,255,80});
        const int total=static_cast<int>(selected(data_.records).size());const int offset=std::min(listOffset_,std::max(0,total-12));int n=0,shown=0;
        for(const auto& item:data_.records) if(item.state==MonoRecord::State::Accepted) {
            if(n++<offset) continue;
            if(shown>=12) break;
            std::string entry=item.key;
            const auto it=std::find_if(solved_.samples.begin(),solved_.samples.end(),[&](const Sample& s){return s.image==item.sample.image;});
            if(it!=solved_.samples.end()) entry+=cv::format(" | %.4f",solved_.data->perViewErrors.at<double>(static_cast<int>(it-solved_.samples.begin())));
            else entry+=" | 等待求解";
            text(canvas,entry.substr(0,55),1115,555+shown*30,item.key==r.key?cv::Scalar(70,215,255):cv::Scalar(230,230,230));++shown;
        }
        text(canvas,"点击列表检查图片；滚轮或按钮翻动列表",1115,985);return canvas;
    }
    int finish() {
        persist();if(detection_.valid()) {try{detection_.get();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';}}
        for(;;) {if(fitting_.valid()) finishFit();startFit();if(!fitting_.valid()) break;}
        if(saving_.valid()) savedRevision_=saving_.get();
        if(solvedRevision_==revision_ && savedRevision_!=revision_ && solved_.data && !solved_.data->cameraMatrix.empty()) {
            saveReport(o_,solved_);ensure(cv::imwrite((o_.output/"review.png").string(),render()),"Cannot save review screenshot");
        }
        std::cout<<"Annotations: "<<o_.input/"annotations.yaml"<<"; results: "<<o_.output<<'\n';return 0;
    }
};
inline int calibrate(const Options& o) {
    cv::setNumThreads(4);CameraStopSignals signals;MonoReview app(o);
    std::cout<<"Offline monocular ROI calibration; no camera opened. Input="<<o.input<<" output="<<o.output<<'\n';
    cv::namedWindow(reviewWindow,cv::WINDOW_NORMAL);cv::resizeWindow(reviewWindow,1440,900);cv::setMouseCallback(reviewWindow,MonoReview::Mouse,&app);
    try {
        while(!signals.stopRequested() && !app.quitting()) {
            app.update();cv::imshow(reviewWindow,app.render());const int key=cv::waitKey(10)&0xff;
            if(cv::getWindowProperty(reviewWindow,cv::WND_PROP_VISIBLE)==0) break;
            if(key!=255) app.key(key);
        }
        cv::setMouseCallback(reviewWindow,nullptr);cv::destroyWindow(reviewWindow);return app.finish();
    }catch(...) {cv::setMouseCallback(reviewWindow,nullptr);cv::destroyWindow(reviewWindow);throw;}
}
} // namespace livecalib
