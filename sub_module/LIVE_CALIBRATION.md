# 实时单相机标定（demarcate）

`demarcate` 已从目录式离线标定替换为海康 SDK 实时标定。入口仍为 `sub_module/demarcate.cpp`。
复用 OpenCV 4.5.4 官方 interactive-calibration 的角点检测、对象点生成、角度检查、参数自动约束、质量评价和样本筛选；支撑源码与许可证在 `opencv_interactive_calibration/`。

## 编译与启动

在上位机 Ubuntu / WSL Bash、仓库根目录运行：

```bash
cmake --build build --target demarcate -j2
./build/demarcate --side=left
./build/demarcate --side=right
```

默认读取源码目录的 `config/dart.yaml`。`--config=/absolute/path/dart.yaml` 可选择其它完整配置。
`--side` 必须明确填写，只连接选定相机；`input.mode` 不决定标定程序的输入方式，无需修改为 camera。
相机 IP、网口 IP、曝光、增益和尺寸来自所选 cameras 节点。旧 K/D、R/T 不作为标定约束；配置文件不会被覆盖。
镜头焦距、对焦、采集尺寸和硬件偏移须与实际使用状态一致。SDK 当前要求 OffsetX/OffsetY 为零，并核对尺寸，发现不匹配会报错，不自动改硬件 ROI。
关闭会占用同一相机的其它采集程序后运行。

不依赖 YOLO / OpenVINO 的独立构建：

```bash
cmake -S . -B build/live-calibration \
  -DDART_BUILD_MAIN=OFF -DDART_BUILD_YOLO=OFF \
  -DDART_BUILD_HSV_TUNER=OFF -DDART_BUILD_CONTROL_DEMO=OFF \
  -DDART_BUILD_CALIBRATION=ON -DBUILD_TESTING=ON
cmake --build build/live-calibration --target demarcate -j2
./build/live-calibration/demarcate --side=left
```

依赖：MVS SDK（默认 `/opt/MVS`）、OpenCV core/imgproc/imgcodecs/calib3d/highgui/aruco 和可用的图形会话。

## 采样

默认棋盘为 **11×8 内角点，方格边长 30 mm**。更换标定板时明确设置：

```bash
./build/demarcate --side=left --cols=11 --rows=8 --square-mm=30 --hold=0.8 --model=auto
```

自动模式要求完整棋盘在约 0.8 秒内保持稳定；重复姿态被跳过。移动到图像不同区域、改变板的距离和适度倾角，等待自动采样。
第一次至少采集 3 个不同姿态后开始求解；默认质量评价至少 10 张，上限 30 张，超过时按官方误差/覆盖筛选并重新拟合。
边移动边观察角点与覆盖图，**采样时要短暂停稳**。采集、棋盘检测、保存/求解分别在后台运行；检测同一时间只处理一帧，完成后选择最新帧，不积压旧图。后台求解时暂停接受新样本，预览和采集继续运行；实际刷新速度取决于上位机。
主画面标有 `LIVE` 和当前帧号，持续显示最新画面；右下角小窗显示上一次检测的原始画面、角点及其帧号（不随 U 去畸变）。下方显示检测状态、上一轮检测耗时和检测帧距当前的时间。旧角点不会叠加到新的实时画面；保存的原图、角点和帧编号始终属于同一帧。
角点在缩小图中搜索，再回到原分辨率做亚像素精修，保存原始像素坐标；显示缩放不会改变 K/D。

| 按键 | 功能 |
|---|---|
| Space | 请求检测按键后的新帧；该帧棋盘完整且姿态不重复时手动采样 |
| A | 自动 / 手动采样切换 |
| R | 撤销最近一个保留样本并重算（求解完成后可用） |
| D | 清空当前样本并重置（求解完成后可用） |
| S | 保存当前已更新的有效解 |
| U | 切换预览去畸变，不改变检测坐标 |
| Esc / Q / Ctrl+C / 关闭窗口 | 退出；等待已接受样本保存完成，正常退出时保存有效解 |

`--auto=false` 可从手动模式开始。`--min-views=10 --max-views=30` 调整保留数量。

## 模型与质量

- `--model=auto`（默认）：普通五项畸变模型起步，保留官方参数约束自动调整；实际 flags 会显示并记录。
- `--model=full`：固定使用五项 Brown 畸变模型，不自动增加参数约束。
- `--model=k1`：固定 k2、k3 和切向畸变为零，主点自由求解。
- `--model=narrow`：在 k1 模型上再固定主点到图像中心，与旧程序的同名模型保持一致。

界面显示原始分辨率下的 fx/fy/cx/cy、总体及最差单张 RMS、焦距约 95% 统计区间，以及官方样本数、RMS、置信度和覆盖评价。
固定参数的标准差可能为零，这不意味着真实参数毫无误差。质量指示基于当前模型与观测，不能代替独立图片和实物距离验证。

## 输出

默认新建 `CalibrationResults/live_<side>_<timestamp>/`。`--output=/new/directory` 可指定新目录；已有目录会被拒绝。

- `frames/frame_*.png`：已接受样本的未标注、原始分辨率图片。
- `intrinsics.yaml`：原始尺寸、K、5×1 D、棋盘尺寸、模型/实际 flags、标准差、每张误差、保留样本的路径/角点/板位姿和质量检查状态。

`camera_matrix`、`distortion_coefficients`、`image_width`、`image_height` 保持现有 `stereo_calibrate` 内参读取格式。
撤销、清空和自动筛选不会删除已拍原图；`board_poses` 才是当前解实际使用的样本清单。
只保存到本次会话目录，绝不自动写回 dart.yaml；双目外参仍需重新标定。

## 验证

```bash
cmake --build build/live-calibration --target dart_live_calibration_test dart_camera_input_test -j2
ctest --test-dir build/live-calibration -R '^(live_calibration|camera_input)$' --output-on-failure
```

回归检查不打开相机或窗口，覆盖后台阻塞时的非阻塞轮询、单任务无积压、结果帧身份/原图保持、异常传递、原分辨率角点恢复、重复姿态、已知投影的内参恢复、官方筛选后样本/误差对齐、自动/窄视场约束、边缘覆盖索引和输出格式。
实机连接、现场画面刷新速度、采样角度门限和最终精度需在实际相机上验收。
