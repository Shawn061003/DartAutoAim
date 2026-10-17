# 实时双目标定：stereo_calibrate

源文件从 `tests/stereo_calibrate.cpp` 迁移到 `sub_module/stereo_calibrate.cpp`，可执行程序仍为 `stereo_calibrate`。
默认直接连接海康双相机；保留原来的图片目录、ROI 和 `--full-frame` 离线入口。

## 构建与运行

上位机 Ubuntu / WSL Bash 中，在仓库根目录运行：

```bash
cmake --build build --target stereo_calibrate -j2
./build/stereo_calibrate
```

默认从源码目录的 `config/dart.yaml` 读取两侧 SDK 配置、各自 K/D、原生尺寸及 `frame_queue.max_timestamp_diff_ms`。
无需修改 `input.mode`；不加载 YOLO/OpenVINO 模型，也不把 YAML 中已有的 R/T 固定为真值。
两侧均要求零 OffsetX/OffsetY，采集尺寸须符合 cameras 配置。相机之间保持固定，关闭其它占用相机的程序。

单目标定的新结果不会自动写回 dart.yaml。若要用刚生成的内参，显式填写实际路径：

```bash
./build/stereo_calibrate \
  --config=/absolute/path/dart.yaml \
  --left-intrinsics=/absolute/path/left/intrinsics.yaml \
  --right-intrinsics=/absolute/path/right/intrinsics.yaml
```

只指定一侧时另一侧仍使用 YAML 中的参数；控制台会打印两侧来源和 K。覆盖文件的尺寸必须与 SDK 配置严格一致。
内参采用普通 pinhole + 五项 Brown 畸变（k1,k2,p1,p2,k3），不支持把 fisheye 参数直接放入该文件。
默认棋盘为 **11×8 内角点、30 mm 方格**，与原双目标定保持一致。

独立构建（不需要启用主程序及 YOLO）：

```bash
cmake -S . -B build/live-calibration \
  -DDART_BUILD_MAIN=OFF -DDART_BUILD_YOLO=OFF \
  -DDART_BUILD_HSV_TUNER=OFF -DDART_BUILD_CONTROL_DEMO=OFF \
  -DDART_BUILD_CALIBRATION=ON -DBUILD_TESTING=ON
cmake --build build/live-calibration --target stereo_calibrate -j2
./build/live-calibration/stereo_calibrate
```

## 采样及实时界面

- 左右图分别按原始尺寸检测；大图先缩小搜索，再回到原分辨率进行亚像素精修。
- SDK 连续采集，已有队列按接收时间配对；后台只保留最近图对，角点检测和标定求解不会阻塞窗口事件。
- 同一块棋盘须在两侧完整出现；默认停稳 0.8 秒、至少有 3 次连续观测才允许采样。手动采样也遵守停稳限制。
- 相同姿态被跳过。移动棋盘到共同视野中的不同位置，改变距离和适度倾角后再次停稳。
- 延续原程序对近似同向安装双目的角点序号处理：左右行方向相反时将右侧角点序列反转 180 度，逐对记录。这不是任意相机姿态下的棋盘 ID 匹配器。
- 默认至少 10 对才通过数量检查，上限 30 对；达到上限暂停新增，不自动隐藏高残差图对。可撤销最近图对或明确删除最差图对。

界面显示：左右原始分辨率和帧号、棋盘识别状态、接收时间差、停稳时间、样本数量、左右覆盖图、总体重投影 RMS、极线 P95/最大误差、基线，以及相邻两次求解的 R/T 变化量。
R/T 变化是数据集更新后的诊断值，不是测距置信区间。
P95 对每个角点先取左右点到极线距离的最大值，再求分位数，与 GetCandidate 的极线误差定义一致；所有残差为各自原始图像中的像素。

| 按键 | 操作 |
|---|---|
| A | 自动 / 手动采样切换 |
| Space | 请求下一次符合停稳、完整检测、配对时间和姿态多样性条件的观测；未满足时需重新按键 |
| R | 撤销最近一个保留图对并重新求解 |
| X | 删除当前解中误差最高的一个图对并重新求解 |
| D | 清空当前保留图对，重置求解状态 |
| S | 后台保存当前已更新解与中位/最差图对参考图 |
| E / U | 原图角点 / 去畸变并绘制极线预览切换（不是双目校正图） |
| Esc / Q / Ctrl+C / 关闭窗口 | 停止采集，等待已接受图对写完，正常退出时保存有效解 |

保存/求解期间暂停接受新样本与删除操作，采集、检测及窗口事件继续。
至少 3 对开始求解；不足或质量未通过时仍可保存诊断结果，但不会标记成有效距离验证。

## 配对时间

当前相机时间戳是主机接收时间，**不是同步曝光时间**。程序强制两侧棋盘停稳，但这仍不能替代硬触发同步。
默认使用 YAML 中的 5 ms 接收时间差上限。连续 10 秒无可用图对会报错，先看 SDK 日志、连接及两侧是否都在输出。
必要时可显式指定 `--pair-ms=20` 调整接收配对容差；该参数不会把相机变成硬件同步。
其它参数：`--hold=0.8 --min-pairs=10 --max-pairs=30 --auto=true`。

## 求解与输出

分别固定两侧 K/D，通过 `cv::stereoCalibrate(..., CALIB_FIX_INTRINSIC)` 求 R/T/E/F，不加相同焦距约束，也不把输入图片缩放到同一尺寸。
默认新建 `CalibrationResults/stereo_live_<timestamp>/`，或 `--output=/NEW/directory`；拒绝覆盖已有目录。

- `left/leftN.png`、`right/rightN.png`：同一次接受采样的未标注原始图像，保持各自分辨率。
- `stereo.yaml`：两侧 K/D 与尺寸、R、T_mm、E、F_reference、基线、总体及逐对残差、保留图对原始角点、源帧号、接收时间、角点反序和质量检查状态。
- `reference.png`：当前中位和最差图对的去畸变极线局部图与原图，延续原程序参考图方式。

撤销、清空及删除最差图对不会删除已经拍摄的原图；`stereo.yaml` 的 pairs 清单才是当前解使用的图对集合。
`X_right_mm = R * X_left_mm + T_mm`；方格单位为 mm，因此 T_mm 和基线也是 mm。
基本质量条件保持为：至少 min-pairs 对、RMS ≤ 0.5 px、极线 P95 ≤ 1 px。
任何情况下都保留 `distance_validated: 0`，不会自动覆盖 dart.yaml；实际工作距离仍须独立验证。
退出码：0=当前解通过基本质量检查；2=诊断/不完整会话；1=配置、SDK、保存等错误。

## 原离线入口

```bash
./build/stereo_calibrate --offline --full-frame \
  LEFT_IMAGE_DIR RIGHT_IMAGE_DIR NEW_OUTPUT_DIR LEFT_INTRINSICS_YAML RIGHT_INTRINSICS_YAML
```

原先的 `--full-frame ...` 和位置参数调用也可继续使用。离线目录应明确选择要参与计算的图对；实时目录中被撤销的原图仍然存在。

## 无相机回归检查

```bash
cmake --build build/live-calibration --target dart_live_stereo_calibration_test dart_camera_input_test -j2
ctest --test-dir build/live-calibration -R '^(live_stereo_calibration|camera_input)$' --output-on-failure
```

覆盖不同分辨率/焦距的外参恢复、K/D 不变、实时原图角点精修、角点反序、接收时间差与双侧停稳、错误图对暴露与删除、保存格式及双侧界面渲染。
这些检查不连接真实相机。实际 SDK 连接、现场处理速度、棋盘识别和最终距离精度仍需实机验收。
