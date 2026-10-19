// Headless regression checks: real chessboard detection, official filtering, solve and stereo-compatible export.
#define DART_CALIBRATION_NO_MAIN
#include "../sub_module/demarcate.cpp"

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<livecalib::Sample> projectedViews(const livecalib::Options& options, const cv::Mat& k) {
    const auto object = livecalib::objectTemplate(options);
    std::vector<livecalib::Sample> samples;
    for (int i=0;i<9;++i) {
        std::vector<cv::Point2f> points;
        const cv::Vec3d rotation(-0.22+0.07*(i%4),0.18-0.09*(i%5),0.03*(i%3));
        const cv::Vec3d translation(-240+110*(i%4),-170+120*(i/4),900+60*(i%3));
        cv::projectPoints(object,rotation,translation,k,cv::Mat::zeros(5,1,CV_64F),points);
        for (std::size_t j=0;j<points.size();++j) {
            points[j].x+=0.025*std::sin(static_cast<double>(i*13+j));
            points[j].y+=0.025*std::cos(static_cast<double>(i*11+j));
        }
        samples.push_back({points,"frames/view_"+std::to_string(i)+".png",static_cast<std::uint64_t>(i)});
    }
    return samples;
}
}

int main() {
    try {
        using namespace livecalib;
        Options options; options.side="left"; options.model="full"; options.minViews=4; options.maxViews=8;
        const cv::Size nativeSize(5472,3648);
        cv::Mat image(nativeSize,CV_8UC3,cv::Scalar(220,220,220));
        const int ox=1800,oy=1400,cell=80;
        for(int y=0;y<9;++y) for(int x=0;x<12;++x)
            cv::rectangle(image,{ox+x*cell,oy+y*cell,cell,cell},(x+y)%2?cv::Scalar(255,255,255):cv::Scalar(0,0,0),cv::FILLED);
        const cv::Mat original=image.clone();
        auto data=cv::makePtr<calib::calibrationData>(); data->imageSize=nativeSize;
        calib::captureParameters parameters; parameters.board=calib::Chessboard; parameters.boardSize=options.board;
        parameters.squareSize=options.squareMm; parameters.captureDelay=1; parameters.fps=1; parameters.templDst=295;
        BoardProcessor detector(data,parameters);
        require(detector.observe(image),"Native-resolution rendered chessboard not detected");
        std::vector<cv::Point2f> expected;
        for(int y=0;y<8;++y) for(int x=0;x<11;++x) expected.emplace_back(ox+(x+1)*cell-0.5f,oy+(y+1)*cell-0.5f);
        require(cornerMotion(expected,detector.points())<0.15,"Downsampled search failed to restore native subpixel coordinates");
        auto reversed=detector.points(); std::reverse(reversed.begin(),reversed.end());
        std::vector<Sample> one{{detector.points(),"pose.png",0}};
        require(!diverse(reversed,one,nativeSize),"Reversed corner ordering counted as a distinct pose");
        for(auto& p:reversed) p.x+=200;
        require(diverse(reversed,one,nativeSize),"A new sensor position was rejected as duplicate");
        require(detector.acceptableAngle() && data->imagePoints.empty(),"Initial pose gate corrupts observations");
        // Deliberately block the worker: polling and rejecting another submission must stay nonblocking.
        std::promise<void> entered, release;
        auto enteredFuture=entered.get_future(); auto releaseFuture=release.get_future().share();
        BackgroundDetector blocked([&](DetectionJob job) {
            entered.set_value(); releaseFuture.wait();
            DetectionResult result; result.frame=std::move(job.frame); result.epoch=job.epoch;
            return result;
        });
        DetectionJob blockedJob; blockedJob.frame.image=original; blockedJob.frame.frame_id=42; blockedJob.epoch=7;
        require(blocked.submit(blockedJob),"Cannot launch background detector");
        const bool started=enteredFuture.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        const auto beforePoll=Clock::now();
        const bool noResult=!blocked.takeReady().has_value();
        const bool noQueue=!blocked.submit(blockedJob);
        const double pollSeconds=std::chrono::duration<double>(Clock::now()-beforePoll).count();
        release.set_value(); // Always release before assertions/destruction, including on failure.
        require(started && noResult && noQueue && pollSeconds<0.25,"Background detection blocks UI polling or queues stale frames");
        std::optional<DetectionResult> completed;
        const auto deadline=Clock::now()+std::chrono::seconds(2);
        while (!(completed=blocked.takeReady()) && Clock::now()<deadline) std::this_thread::yield();
        require(completed && completed->frame.frame_id==42 && completed->epoch==7 && !blocked.busy(),
                "Background result lost frame identity or generation");
        require(cv::norm(completed->frame.image,original)==0,"Worker changed the original image");
        // Exercise the real worker path, including native corners and the isolated angle-check state.
        DetectionJob job; job.frame.image=original; job.frame.frame_id=123; job.capture=parameters;
        job.receivedAt=Clock::now(); job.epoch=8;
        BackgroundDetector realDetector;
        require(realDetector.submit(job),"Cannot launch native-resolution background detection");
        const auto realDeadline=Clock::now()+std::chrono::seconds(30);
        while (!(completed=realDetector.takeReady()) && Clock::now()<realDeadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(completed && completed->found && completed->angleOK && completed->frame.frame_id==123 &&
                completed->epoch==8 && completed->receivedAt==job.receivedAt,
                "Real background detector lost its frame or pose gate");
        require(cornerMotion(expected,completed->points)<0.15 && cv::norm(completed->frame.image,original)==0,
                "Background detection altered the source frame or native coordinates");
        const auto detectedBoard=std::move(*completed);
        BackgroundDetector failing([](DetectionJob) -> DetectionResult { throw std::runtime_error("test detector failure"); });
        require(failing.submit(job),"Cannot launch failing worker");
        bool propagated=false;
        const auto errorDeadline=Clock::now()+std::chrono::seconds(2);
        while (!propagated && Clock::now()<errorDeadline) {
            try { failing.takeReady(); }
            catch (const std::runtime_error& e) { propagated=std::string(e.what())=="test detector failure"; }
            std::this_thread::yield();
        }
        require(propagated && !failing.busy(),"Detector exception was lost or left the worker busy");
        const cv::Size size(1280,960);
        const cv::Mat k=(cv::Mat_<double>(3,3)<<1050,0,640,0,1040,480,0,0,1);
        auto samples=projectedViews(options,k);
        auto solution=solve(options,size,samples,modelFlags(options.model));
        require(solution.samples.size()==8,"Official filter did not retain exactly max-views");
        require(solution.data->imagePoints.size()==solution.samples.size(),"Filtered metadata misalignment");
        require(solution.data->totalAvgErr<0.06,"Projected-view calibration residual too large");
        require(std::abs(solution.data->cameraMatrix.at<double>(0,0)-1050)<15,"Known focal length not recovered");
        for(std::size_t i=0;i<solution.samples.size();++i) {
            require(solution.samples[i].points==solution.data->imagePoints[i],"Image filename points to wrong retained observations");
            std::vector<cv::Point2f> projected;
            cv::projectPoints(objectTemplate(options),solution.data->rvecs[i],solution.data->tvecs[i],
                solution.data->cameraMatrix,solution.data->distCoeffs,projected);
            double rms=std::sqrt(cv::norm(projected,solution.samples[i].points,cv::NORM_L2SQR)/projected.size());
            require(std::abs(rms-solution.data->perViewErrors.at<double>(static_cast<int>(i)))<1e-4,"Saved per-view errors do not match retained solution");
        }
        auto autoOptions=options; autoOptions.model="auto";
        auto autoSolution=solve(autoOptions,size,samples,0);
        require(autoSolution.data->totalAvgErr<0.1,"Official flag tuning did not converge");
        auto narrowOptions=options; narrowOptions.model="narrow";
        auto narrow=solve(narrowOptions,size,samples,modelFlags("narrow"));
        require(narrow.data->distCoeffs.at<double>(1)==0 && narrow.data->distCoeffs.at<double>(2)==0 &&
                narrow.data->distCoeffs.at<double>(3)==0 && narrow.data->distCoeffs.at<double>(4)==0,"Narrow model constraints changed");
        auto partial=samples; partial.resize(2);
        require(solve(options,size,partial,0).data->cameraMatrix.empty(),"Insufficient data yielded a saved calibration");
        options.output=fs::temp_directory_path()/("dart_live_calibration_test_"+std::to_string(Clock::now().time_since_epoch().count()));
        fs::create_directory(options.output); saveReport(options,solution);
        cv::FileStorage report((options.output/"intrinsics.yaml").string(),cv::FileStorage::READ);
        cv::Mat loadedK,loadedD; report["camera_matrix"]>>loadedK; report["distortion_coefficients"]>>loadedD;
        require(loadedK.rows==3 && loadedK.cols==3 && loadedD.rows==5 && loadedD.cols==1,"Export incompatible with stereo intrinsics input");
        require(static_cast<int>(report["image_width"])==1280 && static_cast<int>(report["image_height"])==960,"Export lost native dimensions");
        require(report["board_poses"].size()==8 && cv::norm(loadedK,solution.data->cameraMatrix)==0,"Export lost retained observations or parameters");
        cv::Mat preview(size,CV_8UC3,cv::Scalar(70,70,70));
        cv::drawChessboardCorners(preview,options.board,solution.samples.front().points,true);
        const auto screen=dashboard(preview,options,solution.samples,solution,false,true,false,"Updated; move board to a new pose",&detectedBoard,true,999);
        require(cv::imwrite((options.output/"dashboard.png").string(),screen),"Cannot render dashboard");
        std::cout<<"Dashboard: "<<options.output/"dashboard.png"<<'\n';
        // The official coverage grid must handle native sizes not divisible by 10.
        auto edge=cv::makePtr<calib::calibrationData>(); edge->imageSize={5472,3648};
        edge->imagePoints={{{5471,3647}}}; edge->totalAvgErr=0.1; edge->cameraMatrix=k;
        edge->distCoeffs=cv::Mat::zeros(5,1,CV_64F); edge->stdDeviations=cv::Mat::zeros(18,1,CV_64F);
        calib::calibController quality(edge,0,false,0); quality.updateState();
        edge->stdDeviations.at<double>(4)=std::numeric_limits<double>::quiet_NaN(); quality.updateState();
        require(!quality.getConfidenceIntrervalsState(),"Non-finite uncertainty was reported as confident");
        std::cout<<"Live calibration headless regression checks passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
