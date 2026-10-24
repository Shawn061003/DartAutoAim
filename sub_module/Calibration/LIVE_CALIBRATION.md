# 单目标定：采集 BMP 与离线框选 ROI

程序名仍为 `demarcate`，源码位于本目录。流程与双目标定一致：采集阶段只预览、定时抽帧保存，离线阶段人工框选完整棋盘，由 OpenCV 检测角点并求解内参。

## 编译与运行

在仓库根目录运行：

```bash
cmake --build build --target demarcate -j2
# 左相机：先采集，再离线标定
./build/demarcate --collect --side=left
./build/demarcate --calibration --side=left
# 右相机独立处理
./build/demarcate --collect --side=right
./build/demarcate --calibration --side=right
```

默认读取 `config/dart.yaml`。采集只打开 `--side` 指定的相机，读取其 IP、网口、曝光、增益和尺寸，不使用旧内参约束求解，也不受 `input.mode` 影响。
离线模式不打开相机，也不需要读取相机配置；图片直接使用原始尺寸和坐标。
镜头焦距、对焦、采集尺寸和硬件偏移应与实际使用一致。默认棋盘为 **11×8 内角点、30 mm 方格**，可用 `--cols=11 --rows=8 --square-mm=30` 修改。

依赖 MVS SDK、OpenCV 的 core/imgproc/imgcodecs/calib3d/highgui/aruco/**freetype** 模块。
界面使用中文，程序自动查找 Noto CJK、文泉驿、Droid 中文字体或 WSL 下的 Windows 中文字体。
若缺少字体，可安装 `fonts-noto-cjk`，或设置 `DART_CALIBRATION_FONT=/absolute/path/to/font.ttc`。保留 RMS、ROI、fx/fy 等缩写。

## 采集

默认图片路径相对于仓库根目录：

```text
sub_module/Calibration/Input/
  left/000001.bmp
  left/000002.bmp
  left/dataset.yaml
  right/000001.bmp
  right/000002.bmp
  right/dataset.yaml
```

`--input=/new/directory` 可指定当前单相机的图片文件夹。采集仅允许新的或空目录，拒绝覆盖已有数据；需要再录一组时指定新目录。
保存原始分辨率、未标注 BMP，不另存视频。`dataset.yaml` 记录相机侧、原始尺寸、帧号及主机接收时间。
默认每秒抽取一帧，`--interval=1` 调整间隔。采集不运行棋盘检测，不自动判断清晰度，请在保存时停稳标定板。
写盘与取帧在后台运行，预览保持更新；只保留一个写盘任务，不积压旧帧。

| 按键 | 功能 |
|---|---|
| A | 自动保存 / 暂停 |
| 空格 | 保存下一帧 |
| Esc / Q / Ctrl+C / 关闭窗口 | 退出，等待已经启动的保存完成 |

`--auto=false` 可从暂停状态开始。

## 离线框选与标定

```bash
./build/demarcate --calibration --side=left --input=/path/to/BMPs
```

也可导入已有 BMP 文件夹。首次进入逐张框选，重新进入恢复 `annotations.yaml` 中的 ROI、角点和接受/跳过状态；文件修改后不沿用旧标注。

1. 左键拖动框住完整棋盘，包含全部内角点，并留少量白边。
2. 后台只在原始分辨率 ROI 内运行 `findChessboardCornersSB`，失败时尝试传统检测，再做亚像素精修；不进行全图搜索。
3. 检查角点连线确实对应棋盘，按空格接受并下一张；不清晰、不完整的图片按 N 跳过。
4. 至少接受 3 张后后台求解；增加、移除、重框图片后重新求解。默认质量评价至少 10 张，上限 30 张，用 `--min-views` 和 `--max-views` 调整。
5. **左下角始终显示原始传感器坐标下的覆盖网格**。绿色只来自当前已纳入标定的图片角点；移除、重框或跳过已接受图片时立即重建覆盖图。未接受图片不计入覆盖。
6. 右侧列出实际参与的图片和单张 RMS，并显示总体/最差 RMS、fx/fy/cx/cy、焦距统计区间及数量、置信度、覆盖评价。求解期间清空旧指标。

| 操作 | 功能 |
|---|---|
| 左键拖动 | 框选完整棋盘 ROI |
| 滚轮 | 以光标为中心放大 / 缩小；在列表上则滚动列表 |
| 右键单击 / Shift+右键单击 | 以点击位置为中心放大 / 缩小 |
| 右键拖动 | 平移放大后的图片 |
| + / -、缩放按钮 | 放大 / 缩小；0 重置 |
| [ / ]、列表按钮 | 列表上滚 / 下滚；不依赖滚轮 |
| 空格 / N | 接受 / 跳过并下一张 |
| B / P | 上一张 / 下一张，不改变状态 |
| R / D | 清除 ROI / 将当前图片移出标定 |
| U | 撤销本次操作中最近仍有效的一次接受 |
| S | 后台保存当前有效解和界面截图 |
| Esc / Q / Ctrl+C / 关闭窗口 | 保存标注，等待当前求解并保存有效结果 |

GTK3 的滚轮方向、错误坐标和滚轮事件类型已作兼容处理。某些系统不传递完整滚轮事件时，可使用右键、按钮或键盘。
缩放和平移只影响显示，ROI 及角点会还原到原图坐标；过期的后台检测或求解结果不进入当前图片/数据集。

## 模型与输出

- `--model=auto`（默认）：普通五项畸变模型起步，保留 OpenCV 官方参数约束自动调整。
- `--model=full`：使用五项 Brown 畸变模型，不自动增加约束。
- `--model=k1`：固定 k2、k3、切向畸变为零，主点自由求解。
- `--model=narrow`：在 k1 模型上固定主点到图像中心。

默认新建 `sub_module/Calibration/Results/<side>_<时间>/`，或用 `--output=/NEW/directory` 指定新目录。

- 输入目录下 `annotations.yaml`：自动保存全部图片的 ROI、角点、选择状态及源文件校验信息。
- 输出目录下 `intrinsics.yaml`：K、5×1 D、原始尺寸、棋盘参数、实际求解标志、标准差、单张 RMS、实际使用图片的路径/角点/板位姿和质量状态。
- 输出目录下 `review.png`：保存时的中文界面，包含左下角覆盖图。

源图片不会因跳过或移除而删除。`camera_matrix`、`distortion_coefficients`、`image_width`、`image_height` 保持双目标定内参输入格式，可通过双目程序的 `--left-intrinsics` / `--right-intrinsics` 读取。
不自动写回 dart.yaml。RMS 和统计区间不能代替独立数据、实际距离验证。

## 无相机检查

```bash
cmake --build build/live-calibration --target dart_monocular_workflow_test dart_live_calibration_test -j2
ctest --test-dir build/live-calibration -R 'monocular_workflow|live_calibration' --output-on-failure
```

检查覆盖 BMP 原图保持、拒绝覆盖、原始 ROI 坐标、跳过时丢弃旧检测、标注恢复/失效、增删后的覆盖网格、已知内参恢复及报告格式，并直接调用鼠标处理函数检查右键、滚轮、平移和缩放。
真实相机采集、桌面系统传递的鼠标事件和最终精度仍需实机验收。
