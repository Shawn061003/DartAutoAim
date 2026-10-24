// Headless workflow regression: real BMP I/O, native ROI coordinates, annotation epochs and selected-fit identity.
#define DART_STEREO_WORKFLOW_NO_MAIN
#include "../sub_module/StereoCalibration/stereo_calibration.cpp"

namespace {
using namespace stereo_workflow;
void Check(bool value,const std::string& message) {if(!value) throw std::runtime_error(message);}
template<class F> void Reject(F action,const std::string& message) {
    bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}Check(rejected,message);
}
std::vector<cv::Point2f> Board(cv::Mat& image,cv::Point origin,int cell) {
    for(int y=0;y<=kRows;++y) for(int x=0;x<=kColumns;++x)
        cv::rectangle(image,{origin.x+x*cell,origin.y+y*cell,cell,cell},(x+y)%2?cv::Scalar(250,250,250):cv::Scalar(10,10,10),cv::FILLED);
    std::vector<cv::Point2f> points;for(int y=0;y<kRows;++y) for(int x=0;x<kColumns;++x) points.emplace_back(origin.x+(x+1)*cell-.5f,origin.y+(y+1)*cell-.5f);
    return points;
}
double Error(const std::vector<cv::Point2f>& a,const std::vector<cv::Point2f>& b) {
    if(a.size()!=b.size() || a.empty()) return 1e9;
    const double direct=cv::norm(a,b,cv::NORM_L2SQR);auto reversed=b;std::reverse(reversed.begin(),reversed.end());
    return std::sqrt(std::min(direct,cv::norm(a,reversed,cv::NORM_L2SQR))/a.size());
}
void Poll(Review& review,const std::function<bool()>& complete,int seconds=15) {
    const auto until=Clock::now()+std::chrono::seconds(seconds);
    do{review.update();if(complete()) return;std::this_thread::sleep_for(std::chrono::milliseconds(1));}while(Clock::now()<until);
    throw std::runtime_error("Timed out waiting for headless workflow update");
}
void Drag(Review& review,const Viewport& view,cv::Rect roi) {
    const cv::Point a=view.project(roi.tl()),b=view.project(roi.br());
    review.mouse(cv::EVENT_LBUTTONDOWN,a.x,a.y,0);review.mouse(cv::EVENT_MOUSEMOVE,b.x,b.y,cv::EVENT_FLAG_LBUTTON);review.mouse(cv::EVENT_LBUTTONUP,b.x,b.y,0);
}
}
int main() {
    try {
        cv::setNumThreads(4);
        const auto root=fs::temp_directory_path()/("dart_stereo_workflow_"+std::to_string(Clock::now().time_since_epoch().count()));fs::create_directories(root);
        const auto parse=[](std::vector<std::string> args) {
            std::vector<char*> argv;for(auto& arg:args) argv.push_back(arg.data());return ParseSettings(static_cast<int>(argv.size()),argv.data());
        };
        Check(parse({"stereo_calibration","--collect"}).mode==Settings::Mode::Collect &&
              parse({"stereo_calibration","--calibration"}).mode==Settings::Mode::Calibration,"Requested two-mode CLI did not parse");
        Reject([&]{parse({"stereo_calibration","--collect","--calibration"});},"Ambiguous collection/calibration mode accepted");
        Reject([&]{parse({"stereo_calibration","--calibration","--unknown=1"});},"Unknown workflow argument accepted");
        Reject([&]{parse({"stereo_calibration","--collect","--interval=nan"});},"Nonfinite collection interval accepted");
        {PairWriter empty(root/"empty_capture",5);}
        {PairWriter retry(root/"empty_capture",20);Check(retry.count()==0,"Empty failed collection could not be retried");}
        // Lossless collection, explicit timing gate and refusal to overwrite a prior session.
        PairWriter writer(root/"capture",5);StereoFrame pair;
        pair.left.image=cv::Mat(960,1280,CV_8UC3,cv::Scalar(210,210,210));pair.right.image=cv::Mat(720,960,CV_8UC3,cv::Scalar(210,210,210));
        const auto expectedLeft=Board(pair.left.image,{300,240},22),expectedRight=Board(pair.right.image,{160,180},30);
        pair.left.frame_id=123;pair.right.frame_id=456;pair.left.timestamp_ms=10000;pair.right.timestamp_ms=10004;
        Check(cv::imwrite((root/"collection-dashboard.png").string(),CollectionDashboard(pair,0,true,5,1.,"后台保存原始 BMP 图片组")),"Cannot render live collection preview");
        writer.append(pair);Check(writer.count()==1,"Collection lost a saved pair");
        Check(cv::norm(cv::imread((root/"capture/left/000001.bmp").string()),pair.left.image)==0 &&
              cv::norm(cv::imread((root/"capture/right/000001.bmp").string()),pair.right.image)==0,"BMP collection changed native pixels");
        auto loaded=ReadDataset(root/"capture");Check(loaded.records.size()==1 && loaded.pairMs==5 && loaded.records[0].sample.timestampsKnown &&
              loaded.records[0].sample.leftId==123 && loaded.records[0].sample.rightId==456 && loaded.records[0].sample.rightTime==10004,"Manifest lost source frame/time identity");
        pair.right.timestamp_ms=10006;Reject([&]{writer.append(pair);},"Out-of-tolerance collection accepted");
        Check(writer.count()==1 && !fs::exists(root/"capture/left/000002.bmp"),"Rejected pair left half a saved sample");
        Reject([&]{PairWriter overwrite(root/"capture",5);},"Existing capture session overwritten");
        Check(PairKey("left000010",true)=="10" && PairKey("right_0010",false)=="10" && KeyLess("2","10"),"Pair ID normalization/sorting failed");
        fs::copy_file(root/"capture/left/000001.bmp",root/"capture/left/left1.bmp");
        Reject([&]{ReadDataset(root/"capture");},"Ambiguous duplicate pair IDs accepted");fs::remove(root/"capture/left/left1.bmp");
        // Native ROI-only detection and exact offset recovery on both resolutions.
        const cv::Rect lr(280,220,310,240),rr(140,160,400,310);
        auto detection=DetectPair(pair.left.image,pair.right.image,lr,rr,7,99);
        Check(detection.index==7 && detection.epoch==99 && detection.leftFound && detection.rightFound &&
              Error(detection.left,expectedLeft)<.2 && Error(detection.right,expectedRight)<.2,"ROI detection lost native coordinates or result generation");
        std::vector<cv::Point2f> missing{{1,2}};
        Check(!DetectRoi(pair.left.image,{0,0,100,100},missing) && missing.empty(),"Missing ROI reused old corners");
        Check(!DetectRoi(pair.left.image,{-5,0,100,100},missing),"Out-of-image ROI accepted");
        Viewport viewport;viewport.area={12,64,660,650};viewport.reset({5472,3648});const cv::Point2d native(3250.25,2130.75);
        Check(cv::norm(viewport.unproject(viewport.project(native))-native)<1e-9,"Letterboxed native mapping failed");
        const auto cursor=viewport.project(native);viewport.magnify(cursor,4.);
        Check(cv::norm(viewport.unproject(cursor)-native)<1e-8,"Cursor-centered zoom changed native coordinates");
        const auto recovered=viewport.roi(viewport.project({3100,2000}),viewport.project({3400,2300}));
        Check(std::abs(recovered.x-3100)<=1 && std::abs(recovered.y-2000)<=1 && std::abs(recovered.width-300)<=2,"Zoomed ROI recovery changed native geometry");
        // Resume annotations, reject modified source images, and preserve original image bytes.
        auto& annotated=loaded.records[0];annotated.leftRoi=lr;annotated.rightRoi=rr;annotated.sample.pair.left=detection.left;annotated.sample.pair.right=detection.right;annotated.state=Record::State::Accepted;
        const auto annotations=root/"capture/annotations.yaml";SaveAnnotations(annotations,loaded.records);
        auto resumed=ReadDataset(root/"capture");LoadAnnotations(annotations,resumed.records,pair.left.image.size(),pair.right.image.size());
        Check(Selected(resumed.records).size()==1 && resumed.records[0].sample.pair.left==detection.left,"Accepted ROI annotations did not resume");
        const auto source=root/"capture/left/000001.bmp";const auto stamp=fs::last_write_time(source);fs::last_write_time(source,stamp+std::chrono::seconds(1));
        auto replaced=ReadDataset(root/"capture");LoadAnnotations(annotations,replaced.records,pair.left.image.size(),pair.right.image.size());
        Check(Selected(replaced.records).empty(),"Modified source image reused stale accepted corners");fs::last_write_time(source,stamp);
        // Create an offline dataset with known projected observations for an actual incremental constrained fit.
        const fs::path input=root/"Input";fs::create_directories(input/"left");fs::create_directory(input/"right");
        for(int i=1;i<=4;++i) {fs::copy_file(root/"capture/left/000001.bmp",input/"left"/(std::to_string(i)+".bmp"));fs::copy_file(root/"capture/right/000001.bmp",input/"right"/(std::to_string(i)+".bmp"));}
        Camera left{(cv::Mat_<double>(3,3)<<1600,0,640,0,1600,480,0,0,1),cv::Mat::zeros(5,1,CV_64F),{1280,960}};
        Camera right{(cv::Mat_<double>(3,3)<<2000,0,480,0,2000,360,0,0,1),cv::Mat::zeros(5,1,CV_64F),{960,720}};
        auto data=ReadDataset(input);cv::Mat rotation;cv::Rodrigues(cv::Vec3d(.004,.02,-.003),rotation);const cv::Mat translation=(cv::Mat_<double>(3,1)<<-510,2,12);
        std::vector<cv::Point3f> object;for(int y=0;y<kRows;++y) for(int x=0;x<kColumns;++x) object.emplace_back(x*kSquareMm,y*kSquareMm,0);
        for(int i=0;i<3;++i) {
            auto& r=data.records[i];const cv::Vec3d rv(-.15+.14*i,.12-.12*i,.015*i);const cv::Mat tv=(cv::Mat_<double>(3,1)<<-30+100*i,-100+70*i,2500+200*i);
            cv::Mat br,rrv;cv::Rodrigues(rv,br);cv::Rodrigues(rotation*br,rrv);const cv::Mat rtv=rotation*tv+translation;
            cv::projectPoints(object,rv,tv,left.K,left.D,r.sample.pair.left);cv::projectPoints(object,rrv,rtv,right.K,right.D,r.sample.pair.right);
            r.leftRoi=lr;r.rightRoi=rr;r.state=Record::State::Accepted;
        }
        SaveAnnotations(input/"annotations.yaml",data.records);Settings settings;settings.input=input;settings.output=root/"Result";settings.minPairs=3;
        Review review(settings,left,right);Poll(review,[&]{return !review.solution().R.empty();});
        Check(review.solution().pairs.size()==3 && cv::norm(review.solution().T-translation)<.1 && review.solution().rms<.01,
              "Restored fit failed to recover known bounded extrinsics");
        Check(review.currentIndex()==3,"Review did not resume at the pending pair");
        // Exercise both actual review input paths, including broken GTK wheel event coordinates.
        const auto point=review.viewport(1).project({480,360});
        review.mouse(cv::EVENT_RBUTTONDOWN,cvRound(point.x),cvRound(point.y),0);
        review.mouse(cv::EVENT_RBUTTONUP,cvRound(point.x),cvRound(point.y),0);
        Check(review.viewport(1).zoom>1 && review.viewport(0).zoom==1,"Right-click zoom chose the wrong stereo pane");
        review.mouse(cv::EVENT_RBUTTONDOWN,cvRound(point.x),cvRound(point.y),cv::EVENT_FLAG_SHIFTKEY);
        review.mouse(cv::EVENT_RBUTTONUP,cvRound(point.x),cvRound(point.y),cv::EVENT_FLAG_SHIFTKEY);
        Check(std::abs(review.viewport(1).zoom-1)<1e-9,"Shift+right click did not zoom out");
        review.mouse(cv::EVENT_MOUSEMOVE,cvRound(point.x),cvRound(point.y),0);
        review.mouse(cv::EVENT_MOUSEHWHEEL,0,0,calibration_ui::gtkBackend()?-65536:120*65536);
        Check(review.viewport(1).zoom>1 && review.viewport(0).zoom==1,"GTK wheel position was not restored");
        review.key('3');const auto panCenter=review.viewport(1).center;const auto panZoom=review.viewport(1).zoom;
        review.mouse(cv::EVENT_RBUTTONDOWN,1000,350,0);review.mouse(cv::EVENT_MOUSEMOVE,1040,370,cv::EVENT_FLAG_RBUTTON);review.mouse(cv::EVENT_RBUTTONUP,1040,370,0);
        Check(cv::norm(review.viewport(1).center-panCenter)>1 && review.viewport(1).zoom==panZoom,"Right drag did not pan independently of zoom");
        review.key('0');

        Check(cv::imwrite((root/"review-dashboard.png").string(),review.render()),"Cannot render paired-ROI review dashboard");
        Viewport lv,rv;lv.area={12,64,660,650};rv.area={684,64,660,650};lv.reset(left.referenceSize);rv.reset(right.referenceSize);
        Drag(review,lv,lr);review.update();review.key('n'); // Skip while its detector is running; the old result must not enter another image.
        for(int i=0;i<20;++i) {review.update();std::this_thread::sleep_for(std::chrono::milliseconds(5));}
        Check(review.records()[3].state==Record::State::Skipped && review.solution().pairs.size()==3,"Skipped/in-flight ROI changed the fit dataset");
        Drag(review,lv,lr);Drag(review,rv,rr);
        Poll(review,[&]{return review.records()[3].sample.pair.left.size()==88 && review.records()[3].sample.pair.right.size()==88;});
        Check(cv::imwrite((root/"roi-dashboard.png").string(),review.render()),"Cannot render native ROI overlay");
        review.key(' ');Check(review.records()[3].state==Record::State::Accepted,"Complete manual ROIs could not be accepted");
        review.key('d');Check(review.records()[3].state==Record::State::Pending && review.solution().R.empty(),"Removed pair left stale fit metrics visible");
        Poll(review,[&]{return review.solution().pairs.size()==3;});
        review.key('s');const int result=review.finish();Check(result==0,"Valid offline ROI fit did not pass quality checks");
        cv::FileStorage report((settings.output/"stereo.yaml").string(),cv::FileStorage::READ);
        Check(static_cast<std::string>(report["capture_mode"])=="offline_manual_roi_native_full_frames" && report["pairs"].size()==3 &&
              static_cast<int>(report["pairs"][0]["frame_timestamps_available"])==0 && static_cast<int>(report["pairs"][0]["host_timestamp_diff_ms"])==-1,
              "Offline report misreported fit membership or unknown exposure/receive times");
        cv::FileStorage unrestricted((settings.output/"stereo_unconstrained.yaml").string(),cv::FileStorage::READ);
        Check(unrestricted.isOpened() && unrestricted["pairs"].size()==report["pairs"].size() &&
              static_cast<int>(unrestricted["translation_constraint"]["enabled"])==0 &&
              static_cast<std::string>(unrestricted["capture_mode"])=="offline_manual_roi_native_full_frames" &&
              static_cast<int>(unrestricted["pairs"][0]["frame_timestamps_available"])==0 && fs::exists(settings.output/"reference_unconstrained.png"),
              "Offline save did not include a matching complete unrestricted calibration report");
        Settings list=settings;list.input=root/"List";list.output=root/"ListResult";
        fs::create_directories(list.input/"left");fs::create_directory(list.input/"right");
        for(int i=0;i<22;++i) for(const auto* side:{"left","right"})
            Check(cv::imwrite((list.input/side/cv::format("%06d.bmp",i)).string(),cv::Mat(48,64,CV_8UC3,cv::Scalar(200,200,200))),"Cannot write scroll fixture");
        auto listData=ReadDataset(list.input);
        for(auto& r:listData.records) {
            r.state=Record::State::Accepted;r.leftRoi=r.rightRoi={0,0,64,48};
            for(int y=0;y<8;++y) for(int x=0;x<11;++x) r.sample.pair.left.emplace_back(4+4*x,4+4*y);
            r.sample.pair.right=r.sample.pair.left;
        }
        SaveAnnotations(list.input/"annotations.yaml",listData.records);
        auto tinyLeft=left,tinyRight=right;tinyLeft.referenceSize=tinyRight.referenceSize={64,48};Review listReview(list,tinyLeft,tinyRight);
        listReview.mouse(cv::EVENT_MOUSEMOVE,1450,600,0);
        listReview.mouse(cv::EVENT_MOUSEHWHEEL,0,0,calibration_ui::gtkBackend()?65536:-120*65536);
        Check(listReview.listOffset()==3,"Stereo list wheel did not scroll down");
        listReview.mouse(cv::EVENT_LBUTTONDOWN,1650,945,0);Check(listReview.listOffset()==4,"Stereo list button failed to clamp");
        listReview.mouse(cv::EVENT_LBUTTONDOWN,1450,450,0);Check(listReview.currentIndex()==4,"Scrolled stereo row hit-testing selected the wrong pair");
        listReview.key('[');Check(listReview.listOffset()==1,"Stereo list keyboard did not scroll up");
        std::cout<<"Stereo workflow passed. Artifacts: "<<root<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
