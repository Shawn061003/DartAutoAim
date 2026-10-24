#define DART_CALIBRATION_NO_MAIN
#include "../sub_module/Calibration/demarcate.cpp"

namespace {
void check(bool ok,const std::string& message) {if(!ok) throw std::runtime_error(message);}
void waitDetector(livecalib::MonoReview& app) {
    const auto deadline=livecalib::Clock::now()+std::chrono::seconds(20);
    do {app.update();if(!app.detectorBusy()) return;std::this_thread::sleep_for(std::chrono::milliseconds(2));}while(livecalib::Clock::now()<deadline);
    throw std::runtime_error("ROI detector timeout");
}
}
int main() {
    try {
        using namespace livecalib;cv::setNumThreads(2);
        const auto root=fs::temp_directory_path()/("dart_mono_workflow_"+std::to_string(Clock::now().time_since_epoch().count()));
        Options o;o.side="left";o.model="full";o.input=root/"Input";o.output=root/"Results";o.minViews=4;o.maxViews=12;
        const cv::Size size(2000,1400);const int ox=500,oy=400,cell=40;
        cv::Mat board(size,CV_8UC3,cv::Scalar(220,220,220));
        for(int y=0;y<9;++y) for(int x=0;x<12;++x) cv::rectangle(board,{ox+x*cell,oy+y*cell,cell,cell},(x+y)%2?cv::Scalar(255,255,255):cv::Scalar(0,0,0),cv::FILLED);
        const cv::Mat original=board.clone();const cv::Rect roi(ox-20,oy-20,520,400);
        MonoWriter writer(o);CameraFrame frame;frame.image=board;frame.frame_id=121;frame.timestamp_ms=456;
        check(writer.append(frame)==1,"Cannot collect native BMP");
        frame.image=cv::Mat(size,CV_8UC3,cv::Scalar(128,128,128));++frame.frame_id;writer.append(frame);
        const auto loaded=cv::imread((o.input/"000001.bmp").string());check(loaded.size()==size && cv::norm(loaded,original)==0,"BMP changed raw pixels");
        bool refused=false;try {MonoWriter duplicate(o);}catch(const std::runtime_error&){refused=true;}check(refused,"Existing data was not protected");
        const auto dataset=readDataset(o);check(dataset.records.size()==2 && dataset.records.front().sample.frameId==121,"Frame manifest identity lost");
        std::vector<cv::Point2f> points,expected;
        check(detectRoi(loaded,roi,o.board,points),"Native ROI board not found");
        for(int y=0;y<8;++y) for(int x=0;x<11;++x) expected.emplace_back(ox+(x+1)*cell-.5f,oy+(y+1)*cell-.5f);
        check(cornerMotion(points,expected)<.15 && cv::norm(board,original)==0,"ROI/native offset or subpixel coordinates incorrect");
        check(!detectRoi(frame.image,roi,o.board,points) && points.empty(),"Blank ROI retained stale corners");
        MonoViewport view;view.reset(size);const cv::Point2d native(700,600),screen=view.project(native);
        view.magnify(screen,2);check(cv::norm(view.unproject(screen)-native)<1e-8,"Zoom lost cursor anchor");
        const auto mapped=view.roi(view.project(roi.tl()),view.project(roi.br()));check(mapped==roi,"Zoomed ROI offset incorrect");
        const auto parse=[](std::vector<std::string> args) {
            std::vector<char*> argv;for(auto& arg:args) argv.push_back(arg.data());return parseOptions(static_cast<int>(argv.size()),argv.data());
        };
        check(parse({"demarcate","--collect","--side=right"}).mode==Options::Mode::Collect &&
              parse({"demarcate","--calibration","--side=left"}).input.filename()=="left","Two-mode CLI/default camera directory incorrect");
        bool ambiguous=false;try {parse({"demarcate","--collect","--calibration","--side=left"});}catch(const std::invalid_argument&){ambiguous=true;}
        check(ambiguous,"Ambiguous modes accepted");
        calibration_ui::PointerInput gtk;gtk.gtk=true;gtk.point(cv::EVENT_MOUSEMOVE,500,300);
        check(gtk.point(cv::EVENT_MOUSEHWHEEL,0,0)==cv::Point2d(500,300) && gtk.direction(-65536)==1 && gtk.direction(65536)==-1 && gtk.direction(0)==0,
              "GTK wheel coordinate/direction workaround failed");
        calibration_ui::PointerInput standard;standard.gtk=false;
        check(standard.direction(120*65536)==1 && standard.direction(-120*65536)==-1,"Standard wheel direction changed");
        MonoReview app(o);
        const auto clickPoint=app.viewport().project({800,650});
        app.mouse(cv::EVENT_RBUTTONDOWN,cvRound(clickPoint.x),cvRound(clickPoint.y),0);
        app.mouse(cv::EVENT_RBUTTONUP,cvRound(clickPoint.x),cvRound(clickPoint.y),0);
        check(app.viewport().zoom>1,"Right click did not zoom in");
        app.mouse(cv::EVENT_RBUTTONDOWN,cvRound(clickPoint.x),cvRound(clickPoint.y),cv::EVENT_FLAG_SHIFTKEY);
        app.mouse(cv::EVENT_RBUTTONUP,cvRound(clickPoint.x),cvRound(clickPoint.y),cv::EVENT_FLAG_SHIFTKEY);
        check(std::abs(app.viewport().zoom-1)<1e-9,"Shift+right click did not zoom out");
        app.mouse(cv::EVENT_MOUSEMOVE,cvRound(clickPoint.x),cvRound(clickPoint.y),0);
        app.mouse(cv::EVENT_MOUSEHWHEEL,0,0,calibration_ui::gtkBackend()?-65536:120*65536);
        check(app.viewport().zoom>1,"Wheel did not use the tracked image position");
        app.key('+');const auto beforePan=app.viewport().center;const auto beforeZoom=app.viewport().zoom;
        app.mouse(cv::EVENT_RBUTTONDOWN,550,350,0);app.mouse(cv::EVENT_MOUSEMOVE,590,370,cv::EVENT_FLAG_RBUTTON);app.mouse(cv::EVENT_RBUTTONUP,590,370,0);
        check(cv::norm(app.viewport().center-beforePan)>1 && app.viewport().zoom==beforeZoom,"Right drag failed to pan or incorrectly zoomed");
        app.key('0');

        app.setRoi(roi);app.update();app.key('n');waitDetector(app);
        check(app.currentIndex()==1 && app.dataset().records[1].sample.points.empty() && app.dataset().records[0].state==MonoRecord::State::Skipped,"Stale result crossed a skip/image change");
        app.key('b');app.key('+');
        const auto a=app.viewport().project(roi.tl()),b=app.viewport().project(roi.br());
        app.mouse(cv::EVENT_LBUTTONDOWN,cvRound(a.x),cvRound(a.y),0);app.mouse(cv::EVENT_MOUSEMOVE,cvRound(b.x),cvRound(b.y),cv::EVENT_FLAG_LBUTTON);app.mouse(cv::EVENT_LBUTTONUP,cvRound(b.x),cvRound(b.y),0);
        waitDetector(app);app.key(' ');app.key('b');
        check(selected(app.dataset().records).size()==1,"Complete ROI could not be accepted");
        const auto covered=app.render();const cv::Point2f first=app.dataset().records[0].sample.points.front();
        const cv::Point cellPoint(10+static_cast<int>(first.x*10/size.width)*40+20,675+static_cast<int>(first.y*10/size.height)*20+10);
        check(covered.at<cv::Vec3b>(cellPoint)==cv::Vec3b(55,140,55),"Included corners missing from bottom-left coverage grid");
        check(cv::imwrite((root/"review.png").string(),covered),"Cannot render Chinese review window");
        app.key('d');check(selected(app.dataset().records).empty() && app.render().at<cv::Vec3b>(cellPoint)==cv::Vec3b(40,40,40),"Removed image remained in coverage");
        app.key(' ');auto resumed=readDataset(o);loadAnnotations(o,resumed);
        check(resumed.records[0].state==MonoRecord::State::Accepted && cv::norm(resumed.records[0].roi.tl()-roi.tl())<=3 && cv::norm(resumed.records[0].roi.br()-roi.br())<=3,"Annotations did not resume");
        app.finish();
        auto changed=readDataset(o);fs::last_write_time(changed.records[0].path,fs::last_write_time(changed.records[0].path)+std::chrono::seconds(2));loadAnnotations(o,changed);
        check(changed.records[0].state==MonoRecord::State::Pending,"Changed image reused stale annotation");
        // End-to-end offline solve/export using nine known, independent board poses.
        Options fit=o;fit.input=root/"Synthetic";fit.output=root/"Fit";fs::create_directories(fit.input);
        const cv::Size fitSize(1280,960);const cv::Mat k=(cv::Mat_<double>(3,3)<<1050,0,640,0,1040,480,0,0,1);
        const auto object=objectTemplate(fit);
        for(int i=0;i<9;++i) check(cv::imwrite((fit.input/cv::format("%06d.bmp",i)).string(),cv::Mat(fitSize,CV_8UC3,cv::Scalar(200,200,200))),"Cannot write synthetic image");
        auto synthetic=readDataset(fit);
        for(int i=0;i<9;++i) {
            auto& r=synthetic.records[i];r.state=MonoRecord::State::Accepted;r.roi={0,0,fitSize.width,fitSize.height};
            cv::projectPoints(object,cv::Vec3d(-.22+.07*(i%4),.18-.09*(i%5),.03*(i%3)),cv::Vec3d(-240+110*(i%4),-170+120*(i/4),900+60*(i%3)),k,cv::Mat::zeros(5,1,CV_64F),r.sample.points);
        }
        saveAnnotations(fit,synthetic);MonoReview fitting(fit);fitting.update();fitting.finish();
        check(fitting.solution().samples.size()==9 && fitting.solution().data->totalAvgErr<.01,"Offline fit lost selection or has high RMS");
        cv::FileStorage f((fit.output/"intrinsics.yaml").string(),cv::FileStorage::READ);cv::Mat recovered;f["camera_matrix"]>>recovered;
        check(cv::norm(recovered-k)<1 && f["board_poses"].size()==9 && static_cast<std::string>(f["source_directory"])==fit.input.string(),"Offline export/known intrinsics recovery failed");
        check(fs::exists(fit.output/"review.png"),"Saved report lacks coverage screenshot");
        // More rows than fit on screen: wheel, buttons and click-to-review share the same offset.
        Options list=o;list.input=root/"List";list.output=root/"ListResult";list.maxViews=30;fs::create_directory(list.input);
        for(int i=0;i<20;++i) check(cv::imwrite((list.input/cv::format("%06d.bmp",i)).string(),cv::Mat(48,64,CV_8UC3,cv::Scalar(200,200,200))),"Cannot write list fixture");
        auto listData=readDataset(list);
        for(auto& r:listData.records) {
            r.state=MonoRecord::State::Accepted;r.roi={0,0,64,48};
            for(int y=0;y<8;++y) for(int x=0;x<11;++x) r.sample.points.emplace_back(4+4*x,4+4*y);
        }
        saveAnnotations(list,listData);MonoReview listApp(list);
        listApp.mouse(cv::EVENT_MOUSEMOVE,1200,600,0);
        listApp.mouse(cv::EVENT_MOUSEHWHEEL,0,0,calibration_ui::gtkBackend()?65536:-120*65536);
        check(listApp.listOffset()==3,"Monocular list wheel did not scroll down");
        listApp.key(']');listApp.mouse(cv::EVENT_LBUTTONDOWN,1450,940,0);
        check(listApp.listOffset()==8,"Monocular list button/keyboard did not clamp at the last page");
        listApp.mouse(cv::EVENT_LBUTTONDOWN,1200,555,0);check(listApp.currentIndex()==8,"Scrolled list click selected the wrong monocular image");
        listApp.key('[');check(listApp.listOffset()==5,"Monocular list did not scroll up");
        std::cout<<"Monocular workflow checks passed. Review image: "<<root/"review.png"<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
