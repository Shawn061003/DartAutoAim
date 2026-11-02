#define DART_CALIBRATION_NO_MAIN
#include "../sub_module/Calibration/demarcate.cpp"
#include "aprilgrid_fixture.hpp"

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
        const auto expected=renderAprilGrid(board,{ox,oy},cell);
        const cv::Mat original=board.clone();const cv::Rect roi(ox-20,oy-20,520,400);
        MonoWriter writer(o);CameraFrame frame;frame.image=board;frame.frame_id=121;frame.timestamp_ms=456;
        check(writer.append(frame)==1,"Cannot collect native BMP");
        frame.image=cv::Mat(size,CV_8UC3,cv::Scalar(128,128,128));++frame.frame_id;writer.append(frame);
        const auto loaded=cv::imread((o.input/"000001.bmp").string());check(loaded.size()==size && cv::norm(loaded,original)==0,"BMP changed raw pixels");
        bool refused=false;try {MonoWriter duplicate(o);}catch(const std::runtime_error&){refused=true;}check(refused,"Existing data was not protected");
        const auto dataset=readDataset(o);check(dataset.records.size()==2 && dataset.records.front().sample.frameId==121,"Frame manifest identity lost");
        std::vector<cv::Point2f> points;
        check(detectRoi(loaded,roi,o.board,points),"Native ROI board not found");
        check(cornerMotion(points,expected)<.15 && cv::norm(board,original)==0,"ROI/native offset or subpixel coordinates incorrect");
        const auto physical=aprilgrid::objectPoints();
        check(physical.size()==144 && cv::norm(physical[0]-cv::Point3f(572,0,0))<.001 &&
              cv::norm(physical[2]-cv::Point3f(660,88,0))<.001 &&
              cv::norm(physical[20]-cv::Point3f(0,0,0))<.001 &&
              cv::norm(physical[143]-cv::Point3f(0,660,0))<.001,"Physical ID layout or millimetre scale incorrect");
        cv::Mat rotated;cv::rotate(original,rotated,cv::ROTATE_180);
        check(aprilgrid::detect(rotated,points),"Rotated AprilGrid not detected");
        for(std::size_t i=0;i<points.size();++i)
            check(cv::norm(points[i]-cv::Point2f(size.width-1-expected[i].x,size.height-1-expected[i].y))<.2,
                  "Image rotation changed ID/corner correspondence");
        cv::Mat incomplete=original.clone();
        incomplete(cv::Rect(ox+5*(cell+cvRound(.3*cell)),oy,cell,cell)).setTo(cv::Scalar(220,220,220));
        aprilgrid::DetectionInfo markers;
        check(!detectRoi(incomplete,roi,o.board,points,&markers) && points.empty(),"Missing ID accepted as a full board");
        check(markers.ids.size()==35 && markers.missing()==std::vector<int>{0} && markers.attempts>1 && markers.duplicates.empty(),
              "Partial detection lost visible/missing IDs or skipped scale retries");
        for(std::size_t i=0;i<markers.ids.size();++i) for(int j=0;j<4;++j)
            check(cv::norm(markers.corners[i][j]-expected[4*markers.ids[i]+j])<.2,"Partial diagnostic corners lost native ROI offset");
        auto overlay=incomplete.clone();aprilgrid::draw(overlay,markers);
        check(cv::norm(overlay,incomplete)>0 && cv::norm(board,original)==0,"Partial marker overlay missing or changed original image");
        check(!detectRoi(frame.image,roi,o.board,points,&markers) && points.empty() && markers.ids.empty() && markers.missing().size()==36,
              "Blank ROI retained stale corners or diagnostic IDs");
        check(!detectRoi(frame.image,{-1,0,50,50},o.board,points,&markers) && markers.attempts==0,"Invalid ROI retained diagnostics");
        MonoViewport view;view.reset(size);const cv::Point2d native(700,600),screen=view.project(native);
        view.magnify(screen,2);check(cv::norm(view.unproject(screen)-native)<1e-8,"Zoom lost cursor anchor");
        const auto mapped=view.roi(view.project(roi.tl()),view.project(roi.br()));check(mapped==roi,"Zoomed ROI offset incorrect");
        const auto parse=[](std::vector<std::string> args) {
            std::vector<char*> argv;for(auto& arg:args) argv.push_back(arg.data());return parseOptions(static_cast<int>(argv.size()),argv.data());
        };
        check(parse({"calibration","--collect","--side=right"}).mode==Options::Mode::Collect &&
              parse({"calibration","--calibration","--side=left"}).input.filename()=="left","Two-mode CLI/default camera directory incorrect");
        bool ambiguous=false;try {parse({"calibration","--collect","--calibration","--side=left"});}catch(const std::invalid_argument&){ambiguous=true;}
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
        // All four handles retain their ID/corner order through a zoomed native-coordinate edit.
        const auto beforeEdit=app.dataset().records[0].sample.points;const auto beforeRoi=app.dataset().records[0].roi;
        app.key('e');
        for(int corner=0;corner<4;++corner) {
            const cv::Point hit=app.viewport().project(beforeEdit[corner]),drop=hit+cv::Point(3,2);
            const cv::Point2f nativeDrop=beforeEdit[corner]+cv::Point2f(3./app.viewport().scale(),2./app.viewport().scale());
            app.mouse(cv::EVENT_LBUTTONDOWN,hit.x,hit.y,0);app.mouse(cv::EVENT_MOUSEMOVE,drop.x,drop.y,cv::EVENT_FLAG_LBUTTON);
            check(app.dataset().records[0].sample.points==beforeEdit,"Drag preview changed committed calibration data");
            app.mouse(cv::EVENT_LBUTTONUP,drop.x,drop.y,0);
            const auto& edited=app.dataset().records[0];
            check(edited.state==MonoRecord::State::Pending && selected(app.dataset().records).empty() && edited.roi==beforeRoi && !app.detectorBusy(),
                  "Corner edit triggered ROI detection or retained acceptance");
            for(std::size_t i=0;i<beforeEdit.size();++i)
                check(cv::norm(edited.sample.points[i]-(i==static_cast<std::size_t>(corner)?nativeDrop:beforeEdit[i]))<1e-4,
                      "Corner drag changed another ID/corner or lost native coordinates");
            auto saved=readDataset(o);loadAnnotations(o,saved);
            check(cv::norm(saved.records[0].sample.points[corner]-nativeDrop)<1e-4,"Manual corner was not persisted");
            app.key('z');check(app.dataset().records[0].sample.points==beforeEdit,"Corner undo did not restore the last gesture");
        }
        const cv::Point hit=app.viewport().project(beforeEdit[0]),bad=app.viewport().project(beforeEdit[2]+cv::Point2f(4,4));
        app.mouse(cv::EVENT_LBUTTONDOWN,hit.x+2,hit.y+1,0);app.mouse(cv::EVENT_LBUTTONUP,hit.x+2,hit.y+1,0);
        check(app.dataset().records[0].sample.points==beforeEdit,"Clicking a handle without dragging moved the corner");
        app.mouse(cv::EVENT_LBUTTONDOWN,hit.x,hit.y,0);app.mouse(cv::EVENT_LBUTTONUP,bad.x,bad.y,0);
        check(app.dataset().records[0].sample.points==beforeEdit,"Crossed/flipped marker edit was accepted");
        app.mouse(cv::EVENT_LBUTTONDOWN,hit.x,hit.y,0);app.mouse(cv::EVENT_MOUSEMOVE,hit.x+3,hit.y+2,cv::EVENT_FLAG_LBUTTON);app.key(27);
        check(!app.quitting() && app.dataset().records[0].sample.points==beforeEdit,"Escape did not cancel a drag safely");
        check(cv::imwrite((root/"corner-editor.png").string(),app.render()),"Cannot render manual corner handles");
        app.mouse(cv::EVENT_LBUTTONDOWN,hit.x,hit.y,0);app.mouse(cv::EVENT_LBUTTONUP,hit.x+2,hit.y+1,0);
        const auto committed=app.dataset().records[0].sample.points;
        app.key(' ');app.key('b');check(selected(app.dataset().records)[0].points==committed,"Reaccepting ignored manual corners");
        app.key('d');check(selected(app.dataset().records).empty() && app.render().at<cv::Vec3b>(cellPoint)==cv::Vec3b(40,40,40),"Removed image remained in coverage");
        app.key(' ');auto resumed=readDataset(o);loadAnnotations(o,resumed);
        check(resumed.records[0].state==MonoRecord::State::Accepted && cv::norm(resumed.records[0].roi.tl()-roi.tl())<=3 && cv::norm(resumed.records[0].roi.br()-roi.br())<=3,"Annotations did not resume");
        app.finish();
        // Partial edits survive a reload but cannot become calibration observations.
        Options partial=o;partial.input=root/"Partial";partial.output=root/"PartialResult";fs::create_directories(partial.input);
        check(cv::imwrite((partial.input/"000001.bmp").string(),incomplete),"Cannot write partial edit fixture");
        MonoReview partialApp(partial);partialApp.setRoi(roi);waitDetector(partialApp);partialApp.key('e');
        const cv::Point partialHit=partialApp.viewport().project(expected[20]),partialDrop=partialHit+cv::Point(2,2);
        partialApp.mouse(cv::EVENT_LBUTTONDOWN,partialHit.x,partialHit.y,0);partialApp.mouse(cv::EVENT_LBUTTONUP,partialDrop.x,partialDrop.y,0);
        const auto& partialRecord=partialApp.dataset().records[0];
        check(partialRecord.manualMarkers.ids.size()==35 && partialRecord.manualMarkers.missing()==std::vector<int>{0} && partialRecord.sample.points.empty(),
              "Partial edit lost marker IDs or created full observations");
        const auto manual=partialRecord.manualMarkers;partialApp.key(' ');
        check(selected(partialApp.dataset().records).empty(),"Partial manual edit bypassed complete-board gate");partialApp.finish();
        partial.output=root/"PartialReload";MonoReview partialReload(partial);partialReload.update();
        check(!partialReload.detectorBusy() && partialReload.dataset().records[0].manualMarkers.corners==manual.corners,
              "Reload discarded or redetected manual partial corners");partialReload.finish();
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
            cv::projectPoints(object,cv::Vec3d(-.22+.07*(i%4),.18-.09*(i%5),.03*(i%3)),cv::Vec3d(-240+110*(i%4),-170+120*(i/4),1800+60*(i%3)),k,cv::Mat::zeros(5,1,CV_64F),r.sample.points);
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
            for(const auto& p:aprilgrid::objectPoints()) r.sample.points.emplace_back(4+.08f*p.x,4+.05f*p.y);
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
