// No SDK devices or GUI windows: mixed-resolution geometry, pairing gate, output and live dashboard.
#define DART_STEREO_CALIBRATION_NO_MAIN
#include "../sub_module/stereo_calibrate.cpp"

namespace {
void Check(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
std::vector<stereo_live::LivePair> Synthetic(const Camera& left,const Camera& right,const cv::Mat& R,const cv::Mat& T,const fs::path& root) {
    std::vector<cv::Point3f> board;
    for(int y=0;y<kRows;++y) for(int x=0;x<kColumns;++x) board.emplace_back(x*kSquareMm,y*kSquareMm,0);
    std::vector<stereo_live::LivePair> samples;
    for(int i=0;i<12;++i) {
        const cv::Vec3d rv(.07*(i%4-1),.08*(i%3-1),.025*(i%5));
        cv::Mat rotation; cv::Rodrigues(rv,rotation);
        const cv::Mat translation=(cv::Mat_<double>(3,1)<<-300+35*i,-150+25*(i%4),14000+400*i);
        cv::Mat rightRotation; cv::Rodrigues(R*rotation,rightRotation);
        const cv::Mat rightTranslation=R*translation+T;
        stereo_live::LivePair sample;
        sample.pair.number=i+1; sample.pair.group=0;
        sample.pair.leftPath=root/"left"/("left"+std::to_string(i+1)+".png");
        sample.pair.rightPath=root/"right"/("right"+std::to_string(i+1)+".png");
        cv::projectPoints(board,rv,translation,left.K,left.D,sample.pair.left);
        cv::projectPoints(board,rightRotation,rightTranslation,right.K,right.D,sample.pair.right);
        for(std::size_t j=0;j<board.size();++j) {
            sample.pair.left[j].x+=0.015*std::sin(static_cast<double>(i*7+j));
            sample.pair.left[j].y+=0.015*std::cos(static_cast<double>(i*13+j));
            sample.pair.right[j].x+=0.015*std::cos(static_cast<double>(i*11+j));
            sample.pair.right[j].y+=0.015*std::sin(static_cast<double>(i*17+j));
        }
        const bool reverse=i%3==0;
        if(reverse) std::reverse(sample.pair.right.begin(),sample.pair.right.end());
        sample.pair.rightReversed=stereo_live::AlignRight(sample.pair.left,sample.pair.right);
        Check(sample.pair.rightReversed==reverse,"180-degree right ordering was not corrected");
        sample.leftId=100+i; sample.rightId=700+i;
        sample.leftTime=10000+i*1000; sample.rightTime=sample.leftTime+3;
        samples.push_back(std::move(sample));
    }
    return samples;
}
}


int main() {
    try {
        using namespace stereo_live;
        cv::setNumThreads(4);
        const auto root=fs::temp_directory_path()/("dart_live_stereo_test_"+std::to_string(Clock::now().time_since_epoch().count()));
        fs::create_directories(root/"left"); fs::create_directory(root/"right");
        Camera left{(cv::Mat_<double>(3,3)<<11100,0,2730,0,11080,1820,0,0,1),
                    (cv::Mat_<double>(5,1)<<.04,-.01,.0002,-.0001,.001),{5472,3648}};
        Camera right{(cv::Mat_<double>(3,3)<<16800,0,720,0,16720,540,0,0,1),
                    (cv::Mat_<double>(5,1)<<-.12,.03,-.0001,.0002,-.002),{1440,1080}};
        const cv::Mat originalK1=left.K.clone(),originalK2=right.K.clone(),originalD1=left.D.clone(),originalD2=right.D.clone();
        cv::Mat R; cv::Rodrigues(cv::Vec3d(.007,.026,-.004),R);
        const cv::Mat T=(cv::Mat_<double>(3,1)<<-490,-4,35);
        auto pairs=Synthetic(left,right,R,T,root);
        // Exercise the live detector at both native sizes; search scaling must not leak into output coordinates.
        StereoFrame rendered;
        rendered.left.image=cv::Mat(left.referenceSize,CV_8UC3,cv::Scalar(220,220,220));
        rendered.right.image=cv::Mat(right.referenceSize,CV_8UC3,cv::Scalar(220,220,220));
        auto drawBoard=[](cv::Mat& image,int ox,int oy,int cell) {
            std::vector<cv::Point2f> expected;
            for(int y=0;y<=kRows;++y) for(int x=0;x<=kColumns;++x)
                cv::rectangle(image,{ox+x*cell,oy+y*cell,cell,cell},(x+y)%2?cv::Scalar(255,255,255):cv::Scalar(0,0,0),cv::FILLED);
            for(int y=0;y<kRows;++y) for(int x=0;x<kColumns;++x)
                expected.emplace_back(ox+(x+1)*cell-.5f,oy+(y+1)*cell-.5f);
            return expected;
        };
        const auto expectedLeft=drawBoard(rendered.left.image,1900,1250,80);
        const auto expectedRight=drawBoard(rendered.right.image,220,200,45);
        const auto detected=Observe(std::move(rendered));
        Check(detected.leftFound && detected.rightFound && Motion(detected.left,expectedLeft)<.2 &&
              Motion(detected.right,expectedRight)<.2,"Live detection lost native sensor subpixel coordinates");
        // Distant left boards occupy only 3 pixels/cell after the old 1600-pixel overview.
        // Preserve native points and use a refinement window smaller than a neighbouring cell.
        // fx ~= 10414 px and 30 mm squares at 14 m give ~22.3 native pixels/cell.
        for (const int cell : {8,10,12,22}) {
            StereoFrame distant;
            distant.left.image=cv::Mat(left.referenceSize,CV_8UC3,cv::Scalar(210,210,210));
            distant.right=detected.frame.right;
            distant.left.frame_id=1000+cell; distant.right.frame_id=2000+cell;
            const auto expectedFar=drawBoard(distant.left.image,3580,2280,cell);
            cv::GaussianBlur(distant.left.image,distant.left.image,cell==22?cv::Size(5,5):cv::Size(3,3),cell==22?1.2:.65);
            const cv::Mat raw=distant.left.image.clone();
            const auto far=Observe(std::move(distant));
            Check(far.leftFound && far.rightFound && Motion(far.left,expectedFar)<.2 &&
                  Motion(far.right,expectedRight)<.2,cv::format("Distant cell=%d: found L=%d R=%d, native RMS L=%.3f R=%.3f",cell,far.leftFound,far.rightFound,Motion(far.left,expectedFar),Motion(far.right,expectedRight)));
            Check(far.frame.left.frame_id==static_cast<std::uint64_t>(1000+cell) &&
                  far.frame.right.frame_id==static_cast<std::uint64_t>(2000+cell) &&
                  cv::norm(far.frame.left.image,raw)==0,"Distant-board search changed the paired frame identity or raw image");
        }
        std::vector<cv::Point2f> missing{{1,2}};
        Check(!Detect(cv::Mat(left.referenceSize,CV_8UC3,cv::Scalar(128,128,128)),missing,true,3200) &&
              missing.empty(),"Missing distant board left stale corner coordinates");
        std::vector<cv::Point3f> constraintBoard;
        for (int y=0;y<kRows;++y) for (int x=0;x<kColumns;++x) constraintBoard.emplace_back(x*kSquareMm,y*kSquareMm,0);
        std::vector<std::vector<cv::Point2f>> jacobianLeft,jacobianRight;
        for (const auto& pair:pairs) { jacobianLeft.push_back(pair.pair.left); jacobianRight.push_back(pair.pair.right); }
        BoundedStereoProblem problem(constraintBoard,jacobianLeft,jacobianRight,{left.K,left.D,right.K,right.D},T.at<double>(0));
        cv::Mat parameters=problem.initialize(R,T),residual,jacobian;
        Check(problem.compute(parameters,residual,jacobian),"Cannot evaluate bounded stereo derivatives");
        // Central differences independently verify all global R/T and per-board pose Jacobian columns.
        for (int col=0;col<parameters.rows;++col) {
            const double h=1e-6*std::max(1.,std::abs(parameters.at<double>(col)));
            cv::Mat plus=parameters.clone(),minus=parameters.clone(),ep,em;
            plus.at<double>(col)+=h; minus.at<double>(col)-=h;
            Check(problem.compute(plus,ep,cv::noArray()) && problem.compute(minus,em,cv::noArray()),"Cannot evaluate finite difference");
            const cv::Mat numeric=(ep-em)/(2*h);
            Check(cv::norm(numeric-jacobian.col(col),cv::NORM_INF)<1e-5*std::max(1.,cv::norm(numeric,cv::NORM_INF)),
                  "Bounded stereo Jacobian incorrect at column "+std::to_string(col));
        }
        for (const double sign : {-1.,1.}) {
            TranslationBounds bounds(sign*510.);
            for (const double extreme : {-1000.,1000.}) {
                cv::Mat p=cv::Mat::zeros(6,1,CV_64F); p.rowRange(3,6).setTo(extreme);
                const cv::Mat t=bounds.decode(p);
                Check(bounds.contains(t) && t.at<double>(0)*sign>0 && std::abs(t.at<double>(1))<=10 && std::abs(t.at<double>(2))<=80,
                      "Optical-axis translation bounds failed at an extreme optimizer parameter");
            }
        }
        auto fitted=Solve(pairs,left,right,10);
        Check(fitted.passed,"Mixed-resolution/focal-length stereo failed basic checks");
        Check(cv::norm(fitted.R-R,cv::NORM_INF)<1e-4 && cv::norm(fitted.T-T)<1,"Known stereo extrinsics not recovered");
        Check(cv::norm(left.K,originalK1)==0 && cv::norm(right.K,originalK2)==0 &&
              cv::norm(left.D,originalD1)==0 && cv::norm(right.D,originalD2)==0,"Fixed native K/D were mutated");
        Check(fitted.rms<.05 && fitted.epiP95<.08,"Synthetic residuals unexpectedly high");
        Check(fitted.translationConstraint.converged && !fitted.translationConstraint.atLimit &&
              fitted.translationConstraint.initialT.at<double>(0)<0 && TranslationBounds(-510).contains(fitted.T),
              "Interior negative-X solution violated translation policy");
        const double reportedRms=std::sqrt(cv::norm(fitted.errors,cv::NORM_L2SQR)/(2*fitted.errors.rows));
        Check(std::abs(reportedRms-fitted.rms)<1e-12,"Bounded per-camera residuals disagree with reported RMS");
        const cv::Mat positiveT=(cv::Mat_<double>(3,1)<<490,-4,35);
        const auto positive=Solve(Synthetic(left,right,R,positiveT,root),left,right,10);
        Check(positive.passed && positive.T.at<double>(0)>0 && cv::norm(positive.T-positiveT)<1 &&
              TranslationBounds(510).contains(positive.T),cv::format("Positive-X fit failed: Tx=%.3f error=%.3f rms=%.4f epi=%.4f converged=%d iterations=%d",positive.T.at<double>(0),cv::norm(positive.T-positiveT),positive.rms,positive.epiP95,positive.translationConstraint.converged,positive.translationConstraint.iterations));
        for (const cv::Vec3d& outside : {cv::Vec3d(-650,-4,35),cv::Vec3d(-490,35,35),cv::Vec3d(-490,-4,160),cv::Vec3d(-800,120,300)}) {
            const cv::Mat actual=(cv::Mat_<double>(3,1)<<outside[0],outside[1],outside[2]);
            const auto constrained=Solve(Synthetic(left,right,R,actual,root),left,right,10);
            Check(!TranslationBounds(-510).contains(constrained.translationConstraint.initialT) &&
                  TranslationBounds(-510).contains(constrained.T),"Out-of-range initial T was not constrained");
            Check(constrained.rms>constrained.translationConstraint.initialRms+1e-4 &&
                  cv::norm(constrained.R.t()*constrained.R-cv::Mat::eye(3,3,CV_64F))<1e-9,
                  "Constrained fit hid residual increase or produced an invalid rotation");
            if (outside[0]==-800) Check(!constrained.passed,"Incompatible baseline observations were marked as quality passed");
            const double tx=constrained.T.at<double>(0),ty=constrained.T.at<double>(1),tz=constrained.T.at<double>(2);
            const cv::Mat skew=(cv::Mat_<double>(3,3)<<0.,-tz,ty,tz,0.,-tx,-ty,tx,0.);
            Check(cv::norm(constrained.E-skew*constrained.R)<1e-10 &&
                  cv::norm(constrained.F-right.K.inv().t()*constrained.E*left.K.inv())<1e-12,
                  "E/F were not rebuilt from the bounded R/T");
        }
        Check(fitted.errors.rows==12 && fitted.perPair.size()==12,"Per-pair camera errors not aligned");
        for(std::size_t i=0;i<pairs.size();++i) {
            Check(fitted.pairs[i].pair.number==pairs[i].pair.number && fitted.pairs[i].leftId==pairs[i].leftId,"Pair identity lost");
            const auto recomputed=MeasureEpipolar(pairs[i].pair.left,pairs[i].pair.right,left,right,fitted.F);
            Check(std::abs(recomputed.p95-fitted.perPair[i].p95)<1e-12,"Per-pair epipolar errors not aligned");
        }
        auto shifted=pairs.front().pair.right; for(auto& p:shifted) p.y+=40;
        const auto mismatch=MeasureEpipolar(pairs.front().pair.left,shifted,left,right,fitted.F);
        Check(mismatch.p95>35,"40-pixel input inconsistency was not exposed in native pixel units");
        auto contaminated=pairs; for(auto& p:contaminated[3].pair.right) p.y+=25;
        auto bad=Solve(contaminated,left,right,10);
        Check(!bad.passed && bad.pairs.size()==pairs.size() && bad.epiP95>1,"Bad paired pose was silently dropped or marked good");
        Check(bad.pairs[Worst(bad)].pair.number==4,"Worst-pair removal would target wrong pair");
        auto retained=contaminated; retained.erase(retained.begin()+static_cast<std::ptrdiff_t>(Worst(bad)));
        Check(Solve(retained,left,right,10).passed,"Removing bad pair did not recover the valid solution");
        auto partial=pairs; partial.resize(2); Check(Solve(partial,left,right,10).R.empty(),"Incomplete session yielded valid stereo output");
        partial=pairs; partial.resize(3); Check(!Solve(partial,left,right,10).passed,"Three pairs passed the ten-pair quality policy");

        Observation observation; observation.leftFound=observation.rightFound=true;
        observation.left=pairs[0].pair.left; observation.right=pairs[0].pair.right;
        Stability stable;
        observation.frame.left.timestamp_ms=10000; observation.frame.right.timestamp_ms=10003;
        Check(!stable.update(observation,left,right,.8,5),"One pair was considered stationary");
        observation.frame.left.timestamp_ms=10450; observation.frame.right.timestamp_ms=10453;
        Check(!stable.update(observation,left,right,.8,5),"Two observations bypassed stability gate");
        observation.frame.left.timestamp_ms=10900; observation.frame.right.timestamp_ms=10903;
        Check(stable.update(observation,left,right,.8,5),"Stationary paired board did not become ready");
        for(auto& p:observation.right) p.y+=3;
        observation.frame.left.timestamp_ms=11100; observation.frame.right.timestamp_ms=11103;
        Check(!stable.update(observation,left,right,.8,5),"Motion on only the right side was ignored");
        observation.frame.left.timestamp_ms=11200; observation.frame.right.timestamp_ms=11206;
        Check(!stable.update(observation,left,right,.8,5) && stable.elapsed()==0,"Pair receive-time limit was ignored");
        observation.frame.right.timestamp_ms=11203; observation.rightFound=false;
        Check(!stable.update(observation,left,right,.8,5),"Missing right board passed stability");
        Check(TimestampDifference(std::numeric_limits<std::int64_t>::min(),std::numeric_limits<std::int64_t>::max())==
            std::numeric_limits<std::uint64_t>::max(),"Timestamp difference overflows at signed extremes");
        observation.left=pairs[0].pair.left; observation.right=pairs[0].pair.right; observation.rightFound=true;
        Check(!Diverse(observation,{pairs[0]},left,right),"Repeated paired pose was accepted");
        std::reverse(observation.left.begin(),observation.left.end()); std::reverse(observation.right.begin(),observation.right.end());
        Check(!Diverse(observation,{pairs[0]},left,right),"180-degree detector ordering was treated as a new paired pose");
        for(auto& p:observation.right) p.x+=25;
        Check(Diverse(observation,{pairs[0]},left,right),"Right-side native coordinate motion was ignored");

        // Original images are retained independently of undo/filter operations.
        const cv::Mat rawLeft(left.referenceSize,CV_8UC3,cv::Scalar(65,65,65)),rawRight(right.referenceSize,CV_8UC3,cv::Scalar(85,85,85));
        Check(cv::imwrite(pairs[0].pair.leftPath.string(),rawLeft,{cv::IMWRITE_PNG_COMPRESSION,1}) &&
              cv::imwrite(pairs[0].pair.rightPath.string(),rawRight,{cv::IMWRITE_PNG_COMPRESSION,1}),"Cannot write test raw frames");
        for(std::size_t i=1;i<pairs.size();++i) {
            fs::create_hard_link(pairs[0].pair.leftPath,pairs[i].pair.leftPath);
            fs::create_hard_link(pairs[0].pair.rightPath,pairs[i].pair.rightPath);
        }
        Options options; options.output=root; options.pairMs=5;
        Save(options,left,right,fitted);
        cv::FileStorage report((root/"stereo.yaml").string(),cv::FileStorage::READ);
        cv::Mat loadedR,loadedT,loadedK; report["R"]>>loadedR; report["T_mm"]>>loadedT; report["K_left_reference"]>>loadedK;
        Check(cv::norm(loadedR,fitted.R)<1e-12 && cv::norm(loadedT,fitted.T)<1e-12 && cv::norm(loadedK,left.K)==0,"Export changed extrinsics/coordinate system");
        Check(static_cast<int>(report["left_reference_width"])==5472 && static_cast<int>(report["right_reference_width"])==1440 &&
              report["pairs"].size()==12 && static_cast<int>(report["distance_validated"])==0,"Export lost native dimensions, pairs or validation boundary");
        Check(static_cast<std::string>(report["pairs"][0]["left_frame_id"])=="100" &&
              static_cast<std::string>(report["pairs"][0]["right_frame_id"])=="700","Export paired the wrong frame identities");
        cv::Mat freeT; report["unconstrained_T_mm"]>>freeT;
        Check(cv::norm(freeT-fitted.translationConstraint.initialT)==0 &&
              static_cast<int>(report["translation_constraint"]["enabled"])==1 &&
              static_cast<std::string>(report["translation_constraint"]["coordinate_frame"])=="right_camera_optical_x_right_y_down_z_forward",
              "Export omitted the optical-frame bounds or the unconstrained diagnostic T");
        // Use the just-saved monocular schema as an optional override, and refuse accidental resize/crop mismatch.
        const auto intrinsics=root/"left_intrinsics.yaml";
        { cv::FileStorage file(intrinsics.string(),cv::FileStorage::WRITE);
          file<<"image_width"<<5472<<"image_height"<<3648<<"camera_matrix"<<left.K<<"distortion_coefficients"<<left.D; }
        CameraParams params; params.width=5472; params.height=3648; params.cameraMatrix=left.K; params.distCoeffs=left.D;
        Check(cv::norm(Intrinsics(params,intrinsics).K,left.K)==0,"Monocular override not usable");
        params.width=1440; bool rejected=false;
        try { Intrinsics(params,intrinsics); } catch(const std::exception&) { rejected=true; }
        Check(rejected,"Mismatched intrinsic image size accepted");
        observation.frame.left.image=rawLeft; observation.frame.right.image=rawRight;
        observation.left=pairs[0].pair.left; observation.right=pairs[0].pair.right;
        observation.frame.left.frame_id=100; observation.frame.right.frame_id=700;
        observation.frame.left.timestamp_ms=10000; observation.frame.right.timestamp_ms=10003;
        const auto screen=Dashboard(observation,left,right,options,pairs,fitted,false,true,true,0.8,
            MeasureEpipolar(observation.left,observation.right,left,right,fitted.F),"Updated; move board to another pose",.001,.05);
        Check(cv::imwrite((root/"dashboard.png").string(),screen),"Cannot render stereo dashboard");
        std::cout<<"Mixed native stereo passed: RMS="<<fitted.rms<<" px, epi P95="<<fitted.epiP95
                 <<" px, T error="<<cv::norm(fitted.T-T)<<" mm\nArtifacts: "<<root<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
