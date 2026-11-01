# StereoCalibration：采集与离线框选标定

程序 `stereo_calibration` 分两阶段运行。固定使用 **6×6 tag36h11 AprilGrid，标记黑色外框88 mm、间隙26.4 mm、两层黑边、ID 0～35**。正看板面，ID每行从右向左递增，各行从上向下递增；按ID及标记解码角点顺序建立144个对应点。
代码位于本目录；每轮对当前已接受图对分别拟合左右单目 K/D，再固定新内参完成不同原生分辨率的双目求解和 T 约束。

## 构建

仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build --target stereo_calibration -j2
```

独立构建时可以关闭主程序和 YOLO：

```bash
cmake -S . -B build/live-calibration \
  -DDART_BUILD_MAIN=OFF -DDART_BUILD_YOLO=OFF \
  -DDART_BUILD_HSV_TUNER=OFF -DDART_BUILD_CONTROL_DEMO=OFF \
  -DDART_BUILD_CALIBRATION=ON -DBUILD_TESTING=ON
cmake --build build/live-calibration --target stereo_calibration -j2
```

单目标定使用 `calibration`，旧双目标定入口 `stereo_calibrate` 保留。

界面已改为中文，保留 RMS、ROI 等缩写。需要 OpenCV **freetype** 模块和中文字体；自动查找 Noto CJK、文泉驿、Droid 或 WSL 中的 Windows 字体，也可用 `DART_CALIBRATION_FONT=/path/to/font.ttc` 指定。
GTK3 后端的滚轮反向、错误坐标和事件类型已作兼容处理；系统不传递滚轮时可用右键、缩放按钮和列表按钮。

## 1. 采集配对 BMP

```bash
./build/stereo_calibration --collect
# 收帧时间容差不合适时显式设置，例如：
./build/stereo_calibration --collect --pair-ms=20 --interval=1
```

- 从 `config/dart.yaml` 的相机节点读取 SDK 参数，默认配对时间容差同配置中的 `frame_queue.max_timestamp_diff_ms`。
- 一个窗口左右并排实时显示当前配对图像，右栏显示已保存数量、时间差和采集状态。
- 不运行AprilGrid检测、不写视频。默认自动每秒保存一对 BMP；`--interval` 设置间隔，`A` 暂停/恢复，`Space` 请求保存下一对，`Esc/Q/Ctrl+C/关闭窗口` 退出。
- 原生分辨率 BMP 在后台写盘，不在界面线程写图，不积压保存任务；写盘期间采集、配对和预览继续，只保留最近图对。
- 标定板要停稳、对焦清晰、两侧都完整可见；现场不自动判断清晰度，离线可跳过不合格图对。
- **这里配对的是主机收到图像的时间，不是硬件同步曝光。** 配对时检查时间差，并保存左右帧号与各自接收时间；放宽 `--pair-ms` 不能同步曝光。

默认数据目录相对于源码根目录固定为：

```text
sub_module/StereoCalibration/Input/
  left/000001.bmp
  left/000002.bmp
  right/000001.bmp
  right/000002.bmp
  dataset.yaml
```

`--input=/absolute/path/Input` 可以选择另一套数据。为避免覆盖图片、混合不同安装状态，每次采集使用新的或空目录；已有非空数据时程序会拒绝写入。
BMP 文件名成对相同，`dataset.yaml` 记录真实帧号和时间戳。图像保持原始像素，写图失败不会留下半对成功样本。

## 2. 离线框选与标定

```bash
./build/stereo_calibration --calibration
# 直接指定配对图片，无须提前单独拟合或提供内参文件：
./build/stereo_calibration --calibration --input=/absolute/path/Input
```

这个模式不打开相机，也不依赖旧的K/D文件。原生尺寸直接从配对BMP读取；每次接受或移除图对，后台使用同一个选择快照先分别调用calibrateCamera拟合左右五参数内参，再固定新K/D求解双目R/T。三个阶段使用完全相同的图对，不单独筛除某一侧图片。

1. 左右原图在同一个窗口并排显示。按住鼠标左键，分别框选整块AprilGrid，周围留少量白边。
2. 框选结束后立即在后台对选区原生像素运行 AprilTag解码和亚像素角点精修，无全图回退搜索；框选后的坐标会还原到原图坐标。
3. 显示每个标记的ID和四角。左右按ID及角点编号自动对应，不再按方向反转右侧序号，已移除F操作。
4. `Space` / “接受并下一组” 接受并前往下一对；`N` / “跳过并下一组” 跳过模糊、遮挡、不完整或错误配对的图片。
5. 右侧滚动列表显示当前参与标定的图对 ID、每对左右 RMS 和极线 P95；点击列表可回看某一对，重框或移除。
6. 至少 3 对后后台求解；每次增加、移除、重框已接受图对都会更新样本集合并重新计算。默认至少 10 对才通过数量检查，上限 30 对，可用 `--min-pairs`、`--max-pairs` 调整。

鼠标滚轮以光标为中心缩放，右键单击放大、Shift+右键单击缩小，右键拖动平移；适合放大左侧远处的小AprilGrid。所有标注、角点和求解使用原图坐标，显示缩放不改变 K/D。
正在重算时旧指标会清空并显示求解状态，不能把旧数据集的 RMS 当成当前解。换图、重新框选后返回的旧后台检测结果会被丢弃。

| 操作 | 功能 |
|---|---|
| 左键拖动 | 当前一侧框选完整AprilGrid ROI |
| 滚轮 / 右键拖动 | 缩放 / 平移；右侧列表上滚轮用于滚动图对 |
| 右键单击 / Shift+右键单击 | 以点击位置为中心放大 / 缩小 |
| 1 / 2、3 / 4 | 左侧放大/缩小、右侧放大/缩小，也可点击图片下方按钮 |
| + / - | 放大/缩小最近操作的一侧 |
| [ / ]、列表按钮 | 列表上滚 / 下滚，不依赖滚轮 |
| Space | 两侧完整角点检测后接受并前往下一对 |
| N | 跳过当前图对并前往下一对；已接受时从求解集合移除 |
| B / P | 上一对 / 下一对，不改变该图对状态 |
| R | 清空左右 ROI；已接受时先移出求解集合 |
| D | 从求解集合移除当前图对，保留它的标注 |
| U | 撤销本次操作中最近仍有效的一次接受 |
| 0 | 重置两侧显示缩放与平移 |
| S | 后台保存当前已更新解和参考图 |
| Esc / Q / Ctrl+C / 关闭窗口 | 自动保存标注，等待当前计算并保存最终有效解 |

## 标注、指标与输出

- `Input/annotations.yaml` 自动保存框选、角点、接受/跳过状态。重新进入会恢复，并检查原始文件的名称、尺寸信息和修改时间；更换图片后不沿用旧角点。
- 默认新建同级 `Results/calibration_<时间>/`，也可用 `--output=/NEW/directory` 指定新目录。
- left_intrinsics.yaml / right_intrinsics.yaml：本轮选中图片拟合的左右K/D、原生尺寸、单目RMS及逐图角点/位姿，可直接供单目参数读取接口加载。
- `stereo.yaml` 记录最终参与的图对、原图角点、源帧信息、R/T/E/F、重投影与极线误差，以及约束前后的 T/RMS。
- `reference.png` 显示受约束解中位和最差图对的局部极线参考图和原图。
- `stereo_unconstrained.yaml` 单独保存同一批已接受图对的 **无 T 约束**结果：R、T_mm、E、F_reference、基线、总体 RMS、极线误差及逐对左右 RMS，并标记 `translation_constraint.enabled: 0`。它是约束优化前的原始 `stereoCalibrate(CALIB_FIX_INTRINSIC)` 解，两侧 K/D 固定为本轮单目标定的新结果。
- `reference_unconstrained.png` 使用无约束解自己的 F 和误差选择中位/最差图对并绘制参考图。两套结果分别评价质量，均保留 `distance_validated: 0`。
- 按 S 和正常退出自动保存时都会生成以上两套结果；图对相同，质量和误差指标各自计算。
- 支持直接导入配对 BMP：两边文件名相同，或使用 `left1.bmp/right1.bmp`。只使用同 ID 图对；孤立文件会提示并忽略，重复 ID 报错。缺少采集清单时帧时间标记为未知，不冒充同步图对。

显示总体 RMS、极线 P95/最大值、基线、R、T、左右单目RMS、边界命中和质量状态。
先分别拟合左右单目K/D，再固定这两套新内参，联合优化R/T及各图对AprilGrid位姿。沿用右相机光心坐标系（x 右、y 下、z 沿光轴向前）和：

- `450 ≤ |Tx| ≤ 570 mm`，符号来自无约束初始解。
- `|Ty| ≤ 60 mm`；`|Tz| ≤ 160 mm`。

基本质量要求收敛、数量达标、RMS ≤ 0.5 px、极线 P95 ≤ 1 px。始终保留 `distance_validated: 0`，不会写回 dart.yaml。
退出码：0=当前解通过基本检查；2=没有足够样本或质量未过；1=配置、SDK、读写或计算错误。

## 无相机验证

```bash
cmake --build build/live-calibration --target dart_stereo_workflow_test dart_live_stereo_calibration_test dart_live_calibration_test dart_camera_input_test -j2
ctest --test-dir build/live-calibration --output-on-failure
```

覆盖配对 BMP 与时间元数据、拒绝覆盖、原图坐标还原、缩放框选、标注恢复与源图变更、跳过时丢弃旧检测、增删样本后拟合一致性、T 约束和报告。检查不连接真实相机；实机 SDK 连接、画面刷新、鼠标窗口交互与物理距离精度仍需现场验证。

## AprilGrid标注版本

两侧均须完整检测36个ID，不把检测遍历顺序作为左右对应关系。物理坐标单位毫米，新标注采用版本2并保存板规格，旧棋盘格标注会被拒绝；请使用新采集目录重新框选。报告导出板型、编码家族、标记尺寸、间隙、边框宽度和ID布局，不再写棋盘格方格尺寸。双目阶段固定本轮新拟合的内参，保留受约束/无约束两套外参结果。

## 本轮内参与结果一致性

单目内参、双目外参、极线误差、参考图和导出文件都从同一轮Solution快照读取。新选择会清空旧解，后台完成的旧版本不会覆盖当前选择。内参求解失败会报告错误，不回退到旧配置内参。至少3组可开始计算，基本质量仍要求达到min-pairs；请保留足够的位置、尺度和倾角变化，同一组图片自动串联求解不代表精度自动提高。离线入口已移除left-intrinsics/right-intrinsics参数，不自动写回运行配置。
